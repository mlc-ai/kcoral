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
    Q, K, dO, dV, L,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    stride_lb, stride_lh, stride_ls,
    S, HEAD_SIZE, sm_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """
    dV[k,d] = sum_q attn[q,k] * dO[q,d]
    Grid: B * H * cdiv(S, BLOCK_N)  -- one program per kv-position tile
    """
    pid = tl.program_id(0)
    num_kv = tl.cdiv(S, BLOCK_N)
    pid_kv = pid % num_kv
    pid_bh = pid // num_kv

    b = pid_bh // HEAD_SIZE
    h = pid_bh % HEAD_SIZE

    off_n = pid_kv * BLOCK_N + tl.arange(0, BLOCK_N)
    off_d = tl.arange(0, BLOCK_D)
    mask_n = off_n < S
    mask_d = off_d < BLOCK_D

    bhs_q = b * stride_qb + h * stride_qh
    bhs_k = b * stride_kb + h * stride_kh
    bhs_do = b * stride_dob + h * stride_doh
    bhs_dv = b * stride_dvb + h * stride_dvh
    bhs_l = b * stride_lb + h * stride_lh

    k_ptrs = K + bhs_k + off_n[:, None] * stride_ks + off_d[None, :] * stride_kd
    K_tile = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    acc = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)

    num_qt = tl.cdiv(S, BLOCK_M)
    for q_idx in range(num_qt):
        start_m = q_idx * BLOCK_M
        off_m = start_m + tl.arange(0, BLOCK_M)
        mask_m = off_m < S

        q_ptrs = Q + bhs_q + off_m[:, None] * stride_qs + off_d[None, :] * stride_qd
        Q_tile = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        l_ptrs = L + bhs_l + off_m * stride_ls
        L_tile = tl.load(l_ptrs, mask=mask_m, other=float('-inf'))[:, None]

        scores = tl.dot(Q_tile, K_tile.T) * sm_scale
        attn = tl.exp(scores - L_tile)

        do_ptrs = dO + bhs_do + off_m[:, None] * stride_dos + off_d[None, :] * stride_dod
        dO_tile = tl.load(do_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        acc += tl.dot(attn.T, dO_tile)

    dv_ptrs = dV + bhs_dv + off_n[:, None] * stride_dvs + off_d[None, :] * stride_dvd
    tl.store(dv_ptrs, acc.to(tl.bfloat16), mask=mask_n[:, None] & mask_d[None, :])


@triton.jit
def _bwd_dq_kernel(
    Q, K, V, dO, dQ, L,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    stride_lb, stride_lh, stride_ls,
    S, HEAD_SIZE, sm_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """
    Compute dQ for one (b, h, q-tile). Two-pass over k-dimension.
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

    bhs_q = b * stride_qb + h * stride_qh
    bhs_k = b * stride_kb + h * stride_kh
    bhs_v = b * stride_vb + h * stride_vh
    bhs_do = b * stride_dob + h * stride_doh
    bhs_dq = b * stride_dqb + h * stride_dqh
    bhs_l = b * stride_lb + h * stride_lh

    q_ptrs = Q + bhs_q + off_m[:, None] * stride_qs + off_d[None, :] * stride_qd
    Q_tile = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    do_ptrs = dO + bhs_do + off_m[:, None] * stride_dos + off_d[None, :] * stride_dod
    dO_tile = tl.load(do_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    l_ptrs = L + bhs_l + off_m * stride_ls
    L_tile = tl.load(l_ptrs, mask=mask_m, other=float('-inf'))[:, None]

    # Pass 1: compute delta
    delta = tl.zeros((BLOCK_M, 1), dtype=tl.float32)
    num_kt = tl.cdiv(S, BLOCK_N)
    for k_idx in range(num_kt):
        start_n = k_idx * BLOCK_N
        off_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = off_n < S

        k_ptrs = K + bhs_k + off_n[:, None] * stride_ks + off_d[None, :] * stride_kd
        K_tile = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        v_ptrs = V + bhs_v + off_n[:, None] * stride_vs + off_d[None, :] * stride_vd
        V_tile = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        scores = tl.dot(Q_tile, K_tile.T) * sm_scale
        attn = tl.exp(scores - L_tile)

        DA = tl.dot(dO_tile, V_tile.T)
        delta += tl.sum(attn * DA, axis=1, keepdims=True)

    # Pass 2: accumulate dQ
    dQ_acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    for k_idx in range(num_kt):
        start_n = k_idx * BLOCK_N
        off_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = off_n < S

        k_ptrs = K + bhs_k + off_n[:, None] * stride_ks + off_d[None, :] * stride_kd
        K_tile = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        v_ptrs = V + bhs_v + off_n[:, None] * stride_vs + off_d[None, :] * stride_vd
        V_tile = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        scores = tl.dot(Q_tile, K_tile.T) * sm_scale
        attn = tl.exp(scores - L_tile)

        DA = tl.dot(dO_tile, V_tile.T)
        d_scores = attn * (DA - delta)

        dQ_acc += tl.dot(d_scores, K_tile) * sm_scale

    dq_ptrs = dQ + bhs_dq + off_m[:, None] * stride_dqs + off_d[None, :] * stride_dqd
    tl.store(dq_ptrs, dQ_acc.to(tl.bfloat16), mask=mask_m[:, None] & mask_d[None, :])


@triton.jit
def _bwd_dk_kernel(
    Q, K, V, dO, dK, L,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_lb, stride_lh, stride_ls,
    S, HEAD_SIZE, sm_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """
    Compute dK via atomic add. One program per (b, h, k-tile), scans all q-tiles.
    Grid: B * H * cdiv(S, BLOCK_N)
    """
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

    bhs_q = b * stride_qb + h * stride_qh
    bhs_k = b * stride_kb + h * stride_kh
    bhs_v = b * stride_vb + h * stride_vh
    bhs_do = b * stride_dob + h * stride_doh
    bhs_dk = b * stride_dkb + h * stride_dkh
    bhs_l = b * stride_lb + h * stride_lh

    k_ptrs = K + bhs_k + off_n[:, None] * stride_ks + off_d[None, :] * stride_kd
    K_tile = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    dk_acc = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)

    num_qt = tl.cdiv(S, BLOCK_M)
    for q_idx in range(num_qt):
        start_m = q_idx * BLOCK_M
        off_m = start_m + tl.arange(0, BLOCK_M)
        mask_m = off_m < S

        q_ptrs = Q + bhs_q + off_m[:, None] * stride_qs + off_d[None, :] * stride_qd
        Q_tile = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        do_ptrs = dO + bhs_do + off_m[:, None] * stride_dos + off_d[None, :] * stride_dod
        dO_tile = tl.load(do_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        l_ptrs = L + bhs_l + off_m * stride_ls
        L_tile = tl.load(l_ptrs, mask=mask_m, other=float('-inf'))[:, None]

        scores = tl.dot(Q_tile, K_tile.T) * sm_scale
        attn = tl.exp(scores - L_tile)

        v_ptrs = V + bhs_v + off_n[:, None] * stride_vs + off_d[None, :] * stride_vd
        V_tile = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        DA = tl.dot(dO_tile, V_tile.T)

        # For delta, we need to iterate ALL k tiles, but here we only have one.
        # We need to compute the full delta across all k positions.
        # This is the tricky part - each (b,h,q) position needs the sum over ALL k.
        # So this kernel design won't work correctly without computing delta separately.
        # 
        # Skip for now - we'll fix the architecture below.
        pass

    # placeholder store
    dk_ptrs = dK + bhs_dk + off_n[:, None] * stride_dks + off_d[None, :] * stride_dkd
    tl.store(dk_ptrs, dk_acc.to(tl.bfloat16), mask=mask_n[:, None] & mask_d[None, :])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Multi-head attention backward pass for bf16, d=128."""
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    assert D == 128, f"Expected d=128, got {D}"

    sm_scale = 1.0 / math.sqrt(D)

    if L.ndim == 4:
        L_sq = L.squeeze(-1)
    else:
        L_sq = L

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = 128

    # Zero outputs
    zero_grid = (triton.cdiv(dK.numel(), 1024),)
    _zero_kernel[zero_grid](dK, dK.numel(), BLOCK=1024, num_warps=4)
    zero_grid = (triton.cdiv(dV.numel(), 1024),)
    _zero_kernel[zero_grid](dV, dV.numel(), BLOCK=1024, num_warps=4)
    zero_grid = (triton.cdiv(dQ.numel(), 1024),)
    _zero_kernel[zero_grid](dQ, dQ.numel(), BLOCK=1024, num_warps=4)

    # --- Compute dV ---
    num_kv = triton.cdiv(S, BLOCK_N)
    dv_grid = (B * H * num_kv,)
    _bwd_dv_kernel[dv_grid](
        Q, K, dO, dV, L_sq,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        L_sq.stride(0), L_sq.stride(1), L_sq.stride(2),
        S, H, sm_scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=4, num_stages=2,
    )

    # --- Compute dQ ---
    num_qt = triton.cdiv(S, BLOCK_M)
    dq_grid = (B * H * num_qt,)
    _bwd_dq_kernel[dq_grid](
        Q, K, V, dO, dQ, L_sq,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        L_sq.stride(0), L_sq.stride(1), L_sq.stride(2),
        S, H, sm_scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=4, num_stages=2,
    )

    # dK will be computed similarly but requires delta from the global k-sum
    # We'll use a combined kernel approach with atomic adds for dK
    
    # For now, compute dK with a corrected single-kernel approach
    # Launch: B * H * cdiv(S, BLOCK_M) programs
    # Each computes its q-tile contribution to dK, atomically adding
    _bwd_dk_full[
        lambda META: (B * H * triton.cdiv(S, META["BLOCK_M"]),),
    ](
        Q, K, V, dO, dK, L_sq,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        L_sq.stride(0), L_sq.stride(1), L_sq.stride(2),
        S, H, sm_scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=4, num_stages=2,
    )