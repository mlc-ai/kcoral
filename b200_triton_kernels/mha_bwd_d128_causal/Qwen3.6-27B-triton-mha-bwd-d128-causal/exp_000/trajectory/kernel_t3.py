import math

import torch
import triton
import triton.language as tl


@triton.jit
def _dq_bias_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, O_ptr, L_ptr,
    dQ_ptr,
    stride_b, stride_h, stride_s, stride_d,
    stride_l_b, stride_l_h, stride_l_s,
    S, D,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Compute dQ[b,h,:,:] and store bias[b,h,:] at dQ[:,:, :,0].

    Formula:
      dQ[s,:] = scale * (W @ K)[s,:] - scale * B[s] * (P @ K)[s,:]
      where W = P * dP, dP = dO @ V^T, P = exp(scores - L) * causal_mask,
      B[s] = sum_t(W[s,t]), and scores = Q @ K^T * scale.

    Kernel maps: one program per (b, h); tiles over (s, t) pairs covering full S.
    Uses overlap: s-tiles overlap across t-tile iterations (re-loads Q per t-tile),
    which is simple and correct even if less memory-efficient for large S.
    """
    pid_bh = tl.program_id(0)

    # Compute batch, head indices
    pid_b = pid_bh // 48
    pid_h = pid_bh % 48

    bh_offset = pid_b * stride_b + pid_h * stride_h

    q_base = Q_ptr + bh_offset
    k_base = K_ptr + bh_offset
    v_base = V_ptr + bh_offset
    do_base = dO_ptr + bh_offset
    dq_base = dQ_ptr + bh_offset
    l_base = L_ptr + pid_b * stride_l_b + pid_h * stride_l_h

    acc_WK = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_PK = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_bias = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # Iterate over (s, t) tile pairs with stride BLOCK_M in each dimension
    num_tiles = tl.cdiv(S, BLOCK_M)
    for tile_start in range(0, num_tiles * num_tiles):
        s_idx = tile_start // num_tiles
        t_idx = tile_start % num_tiles

        qs = s_idx * BLOCK_M + tl.arange(0, BLOCK_M)
        ts = t_idx * BLOCK_M + tl.arange(0, BLOCK_M)
        ds = tl.arange(0, BLOCK_N)

        mask_qs = qs < S
        mask_ts = ts < S
        mask_ds = ds < D
        mask_qsd = mask_qs[:, None] & mask_ds[None, :]
        mask_tsd = mask_ts[:, None] & mask_ds[None, :]

        Q_s = tl.load(q_base + qs[:, None] * stride_s + ds[None, :] * stride_d,
                       mask=mask_qsd, other=0.0).to(tl.float32)
        K_t = tl.load(k_base + ts[:, None] * stride_s + ds[None, :] * stride_d,
                       mask=mask_tsd, other=0.0).to(tl.float32)
        V_t = tl.load(v_base + ts[:, None] * stride_s + ds[None, :] * stride_d,
                       mask=mask_tsd, other=0.0).to(tl.float32)
        dO_s = tl.load(do_base + qs[:, None] * stride_s + ds[None, :] * stride_d,
                        mask=mask_qsd, other=0.0).to(tl.float32)

        # scores = Q_s @ K_t^T * scale   [BLOCK_M, BLOCK_M]
        scores = tl.dot(Q_s, K_t) * scale

        # Load L for this s-tile
        L_s = tl.load(l_base + qs * stride_l_s, mask=mask_qs, other=0.0)

        # Softmax: P = exp(scores - L[:, None])  [BLOCK_M, BLOCK_M]
        P = tl.exp(scores - L_s[:, None])

        # Apply causal mask: only attend to t <= s
        causal = qs[:, None] >= ts[None, :]
        valid = mask_qs[:, None] & mask_ts[None, :]
        P = tl.where(valid & causal, P, 0.0)

        # dP = dO_s @ V_t^T   [BLOCK_M, BLOCK_M]
        dP = tl.dot(dO_s, V_t)

        # W = P * dP  (attention-weighted gradient)
        W = P * dP

        # Accumulate WK = W @ K, PK = P @ K, and bias = sum_t(W)
        acc_WK = tl.dot(W, K_t, acc=acc_WK)
        acc_PK = tl.dot(P, K_t, acc=acc_PK)
        acc_bias += tl.sum(W, axis=1)

    # dQ = scale * (WK - bias[:, None] * PK)
    dQ_val = scale * (acc_WK - acc_bias[:, None] * acc_PK)

    # Store dQ and bias
    # Bias goes to dQ[:,:, :,0]; rest of dQ to cols 1..
    valid_qs = (qs < S)[:, None]
    valid_ds = (ds < D)[None, :]
    mask_all = valid_qs & valid_ds

    # Store full dQ
    tl.store(dq_base + qs[:, None] * stride_s + ds[None, :] * stride_d,
             dQ_val.to(dtype=tl.bfloat16), mask=mask_all)

    # Overwrite bias info at column 0 (will be read by dK kernel)
    tl.store(dq_base + qs * stride_s, acc_bias.to(dtype=tl.bfloat16), mask=mask_qs)


@triton.jit
def _dk_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, dQ_ptr, dK_ptr,
    stride_b, stride_h, stride_s, stride_d,
    stride_l_b, stride_l_h, stride_l_s,
    S, D,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Compute dK[b,h,:,:] using bias read from dQ[:,:, :,0].

    Formula:
      dK[t,:] = scale * ((W^T @ Q)[t,:] - ((P*B)^T @ Q)[t,:])
      where B[s] = sum_t(W[s,t]) is read from dQ output column 0.
    """
    pid_bh = tl.program_id(0)

    pid_b = pid_bh // 48
    pid_h = pid_bh % 48

    bh_offset = pid_b * stride_b + pid_h * stride_h

    q_base = Q_ptr + bh_offset
    k_base = K_ptr + bh_offset
    v_base = V_ptr + bh_offset
    do_base = dO_ptr + bh_offset
    dk_base = dK_ptr + bh_offset
    dq_base = dQ_ptr + bh_offset
    l_base = L_ptr + pid_b * stride_l_b + pid_h * stride_l_h

    # Read bias from dQ column 0: B[s] for all s
    bias_full = tl.zeros((S,), dtype=tl.float32)

    # Load bias in tiles
    ds_read = tl.arange(0, BLOCK_N)
    mask_d_read = ds_read < D
    for s_tile_start in range(0, tl.cdiv(S, BLOCK_M)):
        qs_read = s_tile_start * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_q_read = qs_read < S
        # Bias is stored at col 0; load one column
        b_vals = tl.load(dq_base + qs_read[:, None] * stride_s + tl.zeros((BLOCK_M, 1), dtype=tl.int64) * stride_d,
                          mask=mask_q_read[:, None], other=0.0).to(tl.float32)[:, 0]
        bias_full = tl.where(mask_q_read[:, None], b_vals, bias_full)

    acc_WQ = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_PBQ = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    num_tiles = tl.cdiv(S, BLOCK_M)
    for tile_start in range(0, num_tiles * num_tiles):
        s_idx = tile_start // num_tiles
        t_idx = tile_start % num_tiles

        qs = s_idx * BLOCK_M + tl.arange(0, BLOCK_M)
        ts = t_idx * BLOCK_M + tl.arange(0, BLOCK_M)
        ds = tl.arange(0, BLOCK_N)

        mask_qs = qs < S
        mask_ts = ts < S
        mask_ds = ds < D
        mask_qsd = mask_qs[:, None] & mask_ds[None, :]
        mask_tsd = mask_ts[:, None] & mask_ds[None, :]

        Q_s = tl.load(q_base + qs[:, None] * stride_s + ds[None, :] * stride_d,
                       mask=mask_qsd, other=0.0).to(tl.float32)
        K_t = tl.load(k_base + ts[:, None] * stride_s + ds[None, :] * stride_d,
                       mask=mask_tsd, other=0.0).to(tl.float32)
        V_t = tl.load(v_base + ts[:, None] * stride_s + ds[None, :] * stride_d,
                       mask=mask_tsd, other=0.0).to(tl.float32)
        dO_s = tl.load(do_base + qs[:, None] * stride_s + ds[None, :] * stride_d,
                        mask=mask_qsd, other=0.0).to(tl.float32)

        scores = tl.dot(Q_s, K_t) * scale

        L_s = tl.load(l_base + qs * stride_l_s, mask=mask_qs, other=0.0)
        P = tl.exp(scores - L_s[:, None])

        causal = qs[:, None] >= ts[None, :]
        valid = mask_qs[:, None] & mask_ts[None, :]
        P = tl.where(valid & causal, P, 0.0)

        dP = tl.dot(dO_s, V_t)
        W = P * dP

        # W^T @ Q  => tl.dot(W.T, Q)
        acc_WQ = tl.dot(tl.trans(W), Q_s, acc=acc_WQ)

        # (P * bias)^T @ Q
        B_s = tl.load(dq_base + qs * stride_s, mask=mask_qs, other=0.0).to(tl.float32)
        PB = P * B_s[:, None]
        acc_PBQ = tl.dot(tl.trans(PB), Q_s, acc=acc_PBQ)

    dK_val = scale * (acc_WQ - acc_PBQ)

    valid_ts = (ts < S)[:, None]
    valid_ds = (ds < D)[None, :]
    mask_out = valid_ts & valid_ds

    tl.store(dk_base + ts[:, None] * stride_s + ds[None, :] * stride_d,
             dK_val.to(dtype=tl.bfloat16), mask=mask_out)


@triton.jit
def _dv_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, dV_ptr,
    stride_b, stride_h, stride_s, stride_d,
    stride_l_b, stride_l_h, stride_l_s,
    S, D,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Compute dV[b,h,:,:] = P^T @ dO (no bias needed)."""
    pid_bh = tl.program_id(0)

    pid_b = pid_bh // 48
    pid_h = pid_bh % 48

    bh_offset = pid_b * stride_b + pid_h * stride_h

    q_base = Q_ptr + bh_offset
    k_base = K_ptr + bh_offset
    do_base = dO_ptr + bh_offset
    dv_base = dV_ptr + bh_offset
    l_base = L_ptr + pid_b * stride_l_b + pid_h * stride_l_h

    acc_dv = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    num_tiles = tl.cdiv(S, BLOCK_M)
    for tile_start in range(0, num_tiles * num_tiles):
        s_idx = tile_start // num_tiles
        t_idx = tile_start % num_tiles

        qs = s_idx * BLOCK_M + tl.arange(0, BLOCK_M)
        ts = t_idx * BLOCK_M + tl.arange(0, BLOCK_M)
        ds = tl.arange(0, BLOCK_N)

        mask_qs = qs < S
        mask_ts = ts < S
        mask_ds = ds < D
        mask_qsd = mask_qs[:, None] & mask_ds[None, :]
        mask_tsd = mask_ts[:, None] & mask_ds[None, :]

        Q_s = tl.load(q_base + qs[:, None] * stride_s + ds[None, :] * stride_d,
                       mask=mask_qsd, other=0.0).to(tl.float32)
        K_t = tl.load(k_base + ts[:, None] * stride_s + ds[None, :] * stride_d,
                       mask=mask_tsd, other=0.0).to(tl.float32)
        dO_s = tl.load(do_base + qs[:, None] * stride_s + ds[None, :] * stride_d,
                        mask=mask_qsd, other=0.0).to(tl.float32)

        scores = tl.dot(Q_s, K_t) * scale

        L_s = tl.load(l_base + qs * stride_l_s, mask=mask_qs, other=0.0)
        P = tl.exp(scores - L_s[:, None])

        causal = qs[:, None] >= ts[None, :]
        valid = mask_qs[:, None] & mask_ts[None, :]
        P = tl.where(valid & causal, P, 0.0)

        # dV += P^T @ dO  => tl.dot(P.T, dO)
        acc_dv = tl.dot(tl.trans(P), dO_s, acc=acc_dv)

    valid_ts = (ts < S)[:, None]
    valid_ds = (ds < D)[None, :]
    mask_out = valid_ts & valid_ds

    tl.store(dv_base + ts[:, None] * stride_s + ds[None, :] * stride_d,
             acc_dv.to(dtype=tl.bfloat16), mask=mask_out)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Causal multi-head attention backward.

    Computes dQ, dK, dV given Q, K, V, O (forward output), dO (upstream grad),
    and L (logsumexp of QK^T/sqrt(d)).

    Args (all CUDA tensors):
        Q:    [B, H, S, d] bfloat16
        K:    [B, H, S, d] bfloat16
        V:    [B, H, S, d] bfloat16
        O:    [B, H, S, d] bfloat16  (forward output)
        dO:   [B, H, S, d] bfloat16  (upstream gradient)
        L:    [B, H, S] float32 or [B, H, S, 1] float32  (logsumexp)
        dQ:   [B, H, S, d] bfloat16  (output, preallocated)
        dK:   [B, H, S, d] bfloat16  (output, preallocated)
        dV:   [B, H, S, d] bfloat16  (output, preallocated)
    """
    torch.cuda.set_device(Q.device)

    B, H, S, D = Q.shape
    assert H == 48, f"Expected H=48, got {H}"

    # Make sure inputs are contiguous for predictable strides
    if not Q.is_contiguous():
        Q = Q.contiguous()
    if not K.is_contiguous():
        K = K.contiguous()
    if not V.is_contiguous():
        V = V.contiguous()
    if not O.is_contiguous():
        O = O.contiguous()
    if not dO.is_contiguous():
        dO = dO.contiguous()
    if not L.is_contiguous():
        L = L.contiguous()

    # Ensure L is 3D [B, H, S]
    if L.dim() == 4:
        L = L.squeeze(-1)

    # Strides in elements (not bytes)
    stride_b = Q.stride(0)
    stride_h = Q.stride(1)
    stride_s = Q.stride(2)
    stride_d = Q.stride(3)

    stride_l_b = L.stride(0)
    stride_l_h = L.stride(1)
    stride_l_s = L.stride(2)

    scale = 1.0 / math.sqrt(float(D))

    BLOCK_M = 64
    BLOCK_N = 64
    num_warps = 4
    num_stages = 2

    grid_bh = (B * H,)

    # Phase 1: Compute dQ and store bias at dQ[:,:, :,0]
    _dq_bias_kernel[grid_bh](
        Q, K, V, dO, O, L, dQ,
        stride_b, stride_h, stride_s, stride_d,
        stride_l_b, stride_l_h, stride_l_s,
        S, D, scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    # Phase 2: Compute dK (reads bias from dQ)
    _dk_kernel[grid_bh](
        Q, K, V, dO, L, dQ, dK,
        stride_b, stride_h, stride_s, stride_d,
        stride_l_b, stride_l_h, stride_l_s,
        S, D, scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    # Phase 3: Compute dV
    _dv_kernel[grid_bh](
        Q, K, V, dO, L, dV,
        stride_b, stride_h, stride_s, stride_d,
        stride_l_b, stride_l_h, stride_l_s,
        S, D, scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )