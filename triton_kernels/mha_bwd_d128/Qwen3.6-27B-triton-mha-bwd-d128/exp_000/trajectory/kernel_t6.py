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
    Compute dQ (direct store) and dK (atomic add) for one (b, h, m-tile).
    Two passes over k-dimension.
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

    q_ptrs = Q_ptr + obq + off_m[:, None] * stride_qs + off_d[None, :] * stride_qd
    Q_tile = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)
    Q_tile = Q_tile.to(tl.float32)

    do_ptrs = dO_ptr + obdo + off_m[:, None] * stride_dos + off_d[None, :] * stride_dod
    dO_tile = tl.load(do_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)
    dO_tile = dO_tile.to(tl.float32)

    l_ptrs = L_ptr + obl + off_m * stride_ls
    L_vals = tl.load(l_ptrs, mask=mask_m, other=0.0)[:, None]

    # Pass 1: accumulate delta[m] = sum_n A[m,n] * DA[m,n]
    delta = tl.zeros((BLOCK_M, 1), dtype=tl.float32)

    for start_n in range(0, S, BLOCK_N):
        off_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = off_n < S

        k_ptrs = K_ptr + obk + off_n[:, None] * stride_ks + off_d[None, :] * stride_kd
        K_t = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        v_ptrs = V_ptr + obv + off_n[:, None] * stride_vs + off_d[None, :] * stride_vd
        V_t = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        scores = tl.dot(Q_tile, K_t.T) * sm_scale
        attn = tl.exp(scores - L_vals)

        DA = tl.dot(dO_tile, V_t.T)
        delta += tl.sum(attn * DA, axis=1, keepdims=True)

    # Pass 2: accumulate dQ and emit dK contributions
    dQ_acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for start_n in range(0, S, BLOCK_N):
        off_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = off_n < S

        k_ptrs = K_ptr + obk + off_n[:, None] * stride_ks + off_d[None, :] * stride_kd
        K_t = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        v_ptrs = V_ptr + obv + off_n[:, None] * stride_vs + off_d[None, :] * stride_vd
        V_t = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        scores = tl.dot(Q_tile, K_t.T) * sm_scale
        attn = tl.exp(scores - L_vals)

        DA = tl.dot(dO_tile, V_t.T)
        d_scores = attn * (DA - delta)

        dQ_acc += tl.dot(d_scores, K_t) * sm_scale

        dk_contrib = tl.dot(d_scores.T, Q_tile) * sm_scale
        dk_ptrs = dK_ptr + obdk + off_n[:, None] * stride_dks + off_d[None, :] * stride_dkd
        mask_k = mask_n[:, None] & mask_d[None, :]
        tl.atomic_add(dk_ptrs, dk_contrib.to(tl.bfloat16), mask=mask_k)

    dq_ptrs = dQ_ptr + obdq + off_m[:, None] * stride_dqs + off_d[None, :] * stride_dqd
    tl.store(dq_ptrs, dQ_acc.to(tl.bfloat16), mask=mask_m[:, None] & mask_d[None, :])


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
    """Compute dV[n,d] = sum_m A[m,n] * dO[m,d]. One program per (b,h,kv-tile)."""
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

    obq = b * stride_qb + h * stride_qh
    obk = b * stride_kb + h * stride_kh
    obdo = b * stride_dob + h * stride_doh
    obdv = b * stride_dvb + h * stride_dvh
    obl = b * stride_lb + h * stride_lh

    k_ptrs = K_ptr + obk + off_n[:, None] * stride_ks + off_d[None, :] * stride_kd
    K_tile = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    acc = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)

    for start_m in range(0, S, BLOCK_M):
        off_m = start_m + tl.arange(0, BLOCK_M)
        mask_m = off_m < S

        q_ptrs = Q_ptr + obq + off_m[:, None] * stride_qs + off_d[None, :] * stride_qd
        Q_tile = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        l_ptrs = L_ptr + obl + off_m * stride_ls
        L_val = tl.load(l_ptrs, mask=mask_m, other=0.0)[:, None]

        scores = tl.dot(Q_tile, K_tile.T) * sm_scale
        attn = tl.exp(scores - L_val)

        do_ptrs = dO_ptr + obdo + off_m[:, None] * stride_dos + off_d[None, :] * stride_dod
        dO_tile = tl.load(do_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        acc += tl.dot(attn.T, dO_tile)

    dv_ptrs = dV_ptr + obdv + off_n[:, None] * stride_dvs + off_d[None, :] * stride_dvd
    tl.store(dv_ptrs, acc.to(tl.bfloat16), mask=mask_n[:, None] & mask_d[None, :])


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

    zero_el = max(dQ.numel(), dK.numel(), dV.numel())
    zg = (triton.cdiv(zero_el, 1024),)
    _zero_kernel[zg](dQ, dQ.numel(), BLOCK=1024, num_warps=4)
    _zero_kernel[zg](dK, dK.numel(), BLOCK=1024, num_warps=4)
    _zero_kernel[zg](dV, dV.numel(), BLOCK=1024, num_warps=4)

    sq = Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3)
    sk = K.stride(0), K.stride(1), K.stride(2), K.stride(3)
    sv = V.stride(0), V.stride(1), V.stride(2), V.stride(3)
    sdo = dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3)
    sl = L_sq.stride(0), L_sq.stride(1), L_sq.stride(2)
    sdq = dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3)
    sdk = dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3)
    sdv = dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3)

    # --- Kernel 1: Compute dV ---
    num_kv = triton.cdiv(S, BLOCK_N)
    dv_grid = (B * H * num_kv,)
    _bwd_dv_kernel[dv_grid](
        Q, K, dO, dV, L_sq,
        *sq, *sk, *sdo, *sdv, *sl,
        S, H, sm_scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=4, num_stages=2,
    )

    # --- Kernel 2: Compute dQ and dK ---
    num_qt = triton.cdiv(S, BLOCK_M)
    dqdk_grid = (B * H * num_qt,)
    _bwd_dqdk_kernel[dqdk_grid](
        Q, K, V, dO, dQ, dK, L_sq,
        *sq, *sk, *sv, *sdo, *sdq, *sdk, *sl,
        S, H, sm_scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=4, num_stages=2,
    )