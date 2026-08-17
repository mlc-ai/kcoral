import math
import torch
import triton
import triton.language as tl


@triton.jit
def _zero_kernel(out_ptr, n_elements, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = off < n_elements
    tl.store(out_ptr + off, 0.0, mask=mask)


@triton.jit
def _bwd_dv_kernel(
    Q_ptr, K_ptr, dO_ptr, dV_ptr, L_ptr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    stride_lb, stride_lh, stride_ls,
    S, HEAD_SIZE, sm_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """dV[n,d] = sum_m A[m,n] * dO[m,d]. One program per (b,h,kv-tile)."""
    pid = tl.program_id(0)
    num_kv = tl.cdiv(S, BLOCK_N)
    pid_n = pid % num_kv
    pid_bh = pid // num_kv
    b = pid_bh // HEAD_SIZE
    h = pid_bh % HEAD_SIZE

    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    off_d = tl.arange(0, BLOCK_D)
    mask_n = off_n < S
    mask_d = off_d < BLOCK_D

    offset_b_q = b * stride_qb + h * stride_qh
    offset_b_k = b * stride_kb + h * stride_kh
    offset_b_do = b * stride_dob + h * stride_doh
    offset_b_dv = b * stride_dvb + h * stride_dvh
    offset_b_l = b * stride_lb + h * stride_lh

    k_ptrs = K_ptr + offset_b_k + off_n[:, None] * stride_ks + off_d[None, :] * stride_kd
    K_tile = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    acc = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)

    for start_m in range(0, S, BLOCK_M):
        off_m = start_m + tl.arange(0, BLOCK_M)
        mask_m = off_m < S

        q_ptrs = Q_ptr + offset_b_q + off_m[:, None] * stride_qs + off_d[None, :] * stride_qd
        Q_tile = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        l_ptrs = L_ptr + offset_b_l + off_m * stride_ls
        L_vals = tl.load(l_ptrs, mask=mask_m, other=0.0)[:, None]

        scores = tl.dot(Q_tile, K_tile.T) * sm_scale
        attn = tl.exp(scores - L_vals)

        do_ptrs = dO_ptr + offset_b_do + off_m[:, None] * stride_dos + off_d[None, :] * stride_dod
        dO_tile = tl.load(do_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        acc += tl.dot(attn.T, dO_tile)

    dv_out = dV_ptr + offset_b_dv + off_n[:, None] * stride_dvs + off_d[None, :] * stride_dvd
    tl.store(dv_out, acc.to(tl.bfloat16), mask=mask_n[:, None] & mask_d[None, :])


@triton.jit
def _bwd_dqdk_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, dQ_ptr, dK_ptr, L_ptr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_lb, stride_lh, stride_ls,
    S, HEAD_SIZE, sm_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """
    For one (b,h,q-tile): compute dQ direct-store and dK via atomic_add.
    Pass1: compute delta[m] = sum_n A[m,n]*DA[m,n] over all n-tiles.
    Pass2: for each n-tile, compute d_scores, update dQ, atomic-add dK contribution.
    Grid: B * H * cdiv(S, BLOCK_M)
    """
    pid = tl.program_id(0)
    num_qt = tl.cdiv(S, BLOCK_M)
    pid_m = pid % num_qt
    pid_bh = pid // num_qt
    b = pid_bh // HEAD_SIZE
    h = pid_bh % HEAD_SIZE

    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_d = tl.arange(0, BLOCK_D)
    mask_m = off_m < S
    mask_d = off_d < BLOCK_D

    obq = b * stride_qb + h * stride_qh
    obk = b * stride_kb + h * stride_kh
    obv = b * stride_vb + h * stride_vh
    obdo = b * stride_dob + h * stride_doh
    obdq = b * stride_dqb + h * stride_dqh
    obdk = b * stride_dkb + h * stride_dkh
    obl = b * stride_lb + h * stride_lh

    # Load Q tile [BM, BD] -- reused across all n iterations
    q_ptrs = Q_ptr + obq + off_m[:, None] * stride_qs + off_d[None, :] * stride_qd
    Q_tile = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    # Load dO tile [BM, BD]
    do_ptrs = dO_ptr + obdo + off_m[:, None] * stride_dos + off_d[None, :] * stride_dod
    dO_tile = tl.load(do_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    # Load L [BM] -> broadcast to [BM, 1]
    l_ptrs = L_ptr + obl + off_m * stride_ls
    L_vals = tl.load(l_ptrs, mask=mask_m, other=0.0)[:, None]

    # --- Pass 1: delta[m] = sum_n A[m,n]*DA[m,n] ---
    delta = tl.zeros((BLOCK_M, 1), dtype=tl.float32)

    for start_n in range(0, S, BLOCK_N):
        off_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = off_n < S

        k_ptrs = K_ptr + obk + off_n[:, None] * stride_ks + off_d[None, :] * stride_kd
        K_tile = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        v_ptrs = V_ptr + obv + off_n[:, None] * stride_vs + off_d[None, :] * stride_vd
        V_tile = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        scores = tl.dot(Q_tile, K_tile.T) * sm_scale
        attn = tl.exp(scores - L_vals)

        DA = tl.dot(dO_tile, V_tile.T)
        delta += tl.sum(attn * DA, axis=1, keepdims=True)

    # --- Pass 2: accumulate dQ and dK ---
    dQ_acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for start_n in range(0, S, BLOCK_N):
        off_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = off_n < S

        k_ptrs = K_ptr + obk + off_n[:, None] * stride_ks + off_d[None, :] * stride_kd
        K_tile = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        v_ptrs = V_ptr + obv + off_n[:, None] * stride_vs + off_d[None, :] * stride_vd
        V_tile = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        scores = tl.dot(Q_tile, K_tile.T) * sm_scale
        attn = tl.exp(scores - L_vals)

        DA = tl.dot(dO_tile, V_tile.T)
        d_scores = attn * (DA - delta)

        # dQ contribution: d_scores[BM,BN] @ K_tile[BN,BD] -> [BM,BD]
        dQ_acc += tl.dot(d_scores, K_tile) * sm_scale

        # dK contribution: d_scores^T[BN,BM] @ Q_tile[BM,BD] -> [BN,BD]
        dk_contrib = tl.dot(d_scores.T, Q_tile) * sm_scale
        dk_out = dK_ptr + obdk + off_n[:, None] * stride_dks + off_d[None, :] * stride_dkd
        tl.atomic_add(dk_out, dk_contrib.to(tl.bfloat16), mask=(mask_n[:, None] & mask_d[None, :]))

    dq_out = dQ_ptr + obdq + off_m[:, None] * stride_dqs + off_d[None, :] * stride_dqd
    tl.store(dq_out, dQ_acc.to(tl.bfloat16), mask=mask_m[:, None] & mask_d[None, :])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape

    sm_scale = 1.0 / math.sqrt(D)

    if L.ndim == 4:
        L_sq = L.squeeze(-1)
    else:
        L_sq = L

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = 128

    # Zero outputs (dK needs zero init before atomics; dV/dQ overwritten but safe to zero)
    zero_el = max(dQ.numel(), dK.numel(), dV.numel())
    zg = (triton.cdiv(zero_el, 1024),)
    _zero_kernel[zg](dQ, dQ.numel(), BLOCK=1024, num_warps=4)
    _zero_kernel[zg](dK, dK.numel(), BLOCK=1024, num_warps=4)
    _zero_kernel[zg](dV, dV.numel(), BLOCK=1024, num_warps=4)

    strides_Q = [Q.stride(i) for i in range(4)]
    strides_K = [K.stride(i) for i in range(4)]
    strides_V = [V.stride(i) for i in range(4)]
    strides_dO = [dO.stride(i) for i in range(4)]
    strides_L = [L_sq.stride(i) for i in range(3)]
    strides_dQ = [dQ.stride(i) for i in range(4)]
    strides_dK = [dK.stride(i) for i in range(4)]
    strides_dV = [dV.stride(i) for i in range(4)]

    # --- Kernel 1: Compute dV ---
    num_kv = triton.cdiv(S, BLOCK_N)
    dv_grid = (B * H * num_kv,)
    _bwd_dv_kernel[dv_grid](
        Q, K, dO, dV, L_sq,
        *strides_Q, *strides_K, *strides_dO, *strides_dV, *strides_L,
        S, H, sm_scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=4, num_stages=2,
    )

    # --- Kernel 2: Compute dQ and dK ---
    num_qt = triton.cdiv(S, BLOCK_M)
    dqdk_grid = (B * H * num_qt,)
    _bwd_dqdk_kernel[dqdk_grid](
        Q, K, V, dO, dQ, dK, L_sq,
        *strides_Q, *strides_K, *strides_V, *strides_dO, *strides_dQ, *strides_dK, *strides_L,
        S, H, sm_scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=4, num_stages=2,
    )