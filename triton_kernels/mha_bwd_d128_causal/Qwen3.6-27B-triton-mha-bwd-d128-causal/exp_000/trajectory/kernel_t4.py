import math

import torch
import triton
import triton.language as tl


@triton.jit
def _dv_kernel(
    Q_ptr,
    K_ptr,
    dO_ptr,
    L_ptr,
    dV_ptr,
    stride_b, stride_h, stride_s, stride_d,
    stride_l_b, stride_l_h, stride_l_s,
    S, D, scale,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    H: tl.constexpr,
):
    """dV[b,h,t,:] = sum_s P[s,t] * dO[s,:]
    Grid: (B*H, ceil(S/BLOCK_M)). Fix s-tile, loop over t-tiles."""
    pid_bh = tl.program_id(0)
    pid_s = tl.program_id(1)

    pid_b = pid_bh // H
    pid_h = pid_bh % H

    bh_off = pid_b * stride_b + pid_h * stride_h

    q_base = Q_ptr + bh_off
    k_base = K_ptr + bh_off
    v_base = V_ptr + bh_off
    do_base = dO_ptr + bh_off
    dv_base = dV_ptr + bh_off
    l_base = L_ptr + pid_b * stride_l_b + pid_h * stride_l_h

    qs = pid_s * BLOCK_M + tl.arange(0, BLOCK_M)
    ds = tl.arange(0, BLOCK_D)
    mask_qs = qs < S
    mask_ds = ds < D
    mask_qsd = mask_qs[:, None] & mask_ds[None, :]

    Q_s = tl.load(q_base + qs[:, None] * stride_s + ds[None, :] * stride_d,
                  mask=mask_qsd, other=0.0).to(tl.float32)
    dO_s = tl.load(do_base + qs[:, None] * stride_s + ds[None, :] * stride_d,
                   mask=mask_qsd, other=0.0).to(tl.float32)
    L_s = tl.load(l_base + qs * stride_l_s, mask=mask_qs, other=0.0)

    acc_dv = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for t_idx in range(tl.cdiv(S, BLOCK_M)):
        ts = t_idx * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_ts = ts < S
        mask_tsd = mask_ts[:, None] & mask_ds[None, :]

        K_t = tl.load(k_base + ts[:, None] * stride_s + ds[None, :] * stride_d,
                       mask=mask_tsd, other=0.0).to(tl.float32)
        V_t = tl.load(v_base + ts[:, None] * stride_s + ds[None, :] * stride_d,
                       mask=mask_tsd, other=0.0).to(tl.float32)

        scores = tl.dot(Q_s, K_t) * scale
        P = tl.exp(scores - L_s[:, None])

        causal = qs[:, None] >= ts[None, :]
        valid = mask_qs[:, None] & mask_ts[None, :]
        P = tl.where(valid & causal, P, 0.0)

        acc_dv = tl.dot(tl.trans(P), dO_s, acc=acc_dv)

    mask_out = mask_qsd
    tl.store(dv_base + qs[:, None] * stride_s + ds[None, :] * stride_d,
             acc_dv.to(dtype=tl.bfloat16), mask=mask_out)


@triton.jit
def _dq_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    dO_ptr,
    L_ptr,
    dQ_ptr,
    stride_b, stride_h, stride_s, stride_d,
    stride_l_b, stride_l_h, stride_l_s,
    S, D, scale,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    H: tl.constexpr,
):
    """dQ[b,h,s,:] = scale * (WK - bias[:,None]*PK)
    Grid: (B*H, ceil(S/BLOCK_M)). Fix s-tile, loop over t-tiles.
    Bias stored at dQ[:,:,s,0]."""
    pid_bh = tl.program_id(0)
    pid_s = tl.program_id(1)

    pid_b = pid_bh // H
    pid_h = pid_bh % H

    bh_off = pid_b * stride_b + pid_h * stride_h

    q_base = Q_ptr + bh_off
    k_base = K_ptr + bh_off
    v_base = V_ptr + bh_off
    do_base = dO_ptr + bh_off
    dq_base = dQ_ptr + bh_off
    l_base = L_ptr + pid_b * stride_l_b + pid_h * stride_l_h

    qs = pid_s * BLOCK_M + tl.arange(0, BLOCK_M)
    ds = tl.arange(0, BLOCK_D)
    mask_qs = qs < S
    mask_ds = ds < D
    mask_qsd = mask_qs[:, None] & mask_ds[None, :]

    Q_s = tl.load(q_base + qs[:, None] * stride_s + ds[None, :] * stride_d,
                  mask=mask_qsd, other=0.0).to(tl.float32)
    dO_s = tl.load(do_base + qs[:, None] * stride_s + ds[None, :] * stride_d,
                   mask=mask_qsd, other=0.0).to(tl.float32)
    L_s = tl.load(l_base + qs * stride_l_s, mask=mask_qs, other=0.0)

    acc_wK = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    acc_pK = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    acc_bias = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for t_idx in range(tl.cdiv(S, BLOCK_M)):
        ts = t_idx * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_ts = ts < S
        mask_tsd = mask_ts[:, None] & mask_ds[None, :]

        K_t = tl.load(k_base + ts[:, None] * stride_s + ds[None, :] * stride_d,
                       mask=mask_tsd, other=0.0).to(tl.float32)
        V_t = tl.load(v_base + ts[:, None] * stride_s + ds[None, :] * stride_d,
                       mask=mask_tsd, other=0.0).to(tl.float32)

        scores = tl.dot(Q_s, K_t) * scale
        P = tl.exp(scores - L_s[:, None])

        causal = qs[:, None] >= ts[None, :]
        valid = mask_qs[:, None] & mask_ts[None, :]
        P = tl.where(valid & causal, P, 0.0)

        dP = tl.dot(dO_s, V_t)
        W = P * dP

        acc_wK = tl.dot(W, K_t, acc=acc_wK)
        acc_pK = tl.dot(P, K_t, acc=acc_pK)
        acc_bias += tl.sum(W, axis=1)

    dQ_val = scale * (acc_wK - acc_bias[:, None] * acc_pK)
    tl.store(dq_base + qs[:, None] * stride_s + ds[None, :] * stride_d,
             dQ_val.to(dtype=tl.bfloat16), mask=mask_qsd)
    tl.store(dq_base + qs * stride_s, acc_bias.to(dtype=tl.bfloat16), mask=mask_qs)


@triton.jit
def _dk_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    dO_ptr,
    L_ptr,
    dQ_ptr,
    dK_ptr,
    stride_b, stride_h, stride_s, stride_d,
    stride_l_b, stride_l_h, stride_l_s,
    S, D, scale,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    H: tl.constexpr,
):
    """dK[b,h,t,:] = scale * (W^T @ Q - (P*B)^T @ Q)
    Read bias B from dQ[:,:, :,0]."""
    pid_bh = tl.program_id(0)
    pid_s = tl.program_id(1)

    pid_b = pid_bh // H
    pid_h = pid_bh % H

    bh_off = pid_b * stride_b + pid_h * stride_h

    q_base = Q_ptr + bh_off
    k_base = K_ptr + bh_off
    v_base = V_ptr + bh_off
    do_base = dO_ptr + bh_off
    dk_base = dK_ptr + bh_off
    dq_base = dQ_ptr + bh_off
    l_base = L_ptr + pid_b * stride_l_b + pid_h * stride_l_h

    qs = pid_s * BLOCK_M + tl.arange(0, BLOCK_M)
    ds = tl.arange(0, BLOCK_D)
    mask_qs = qs < S
    mask_ds = ds < D
    mask_qsd = mask_qs[:, None] & mask_ds[None, :]

    acc_wQ = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    acc_pbQ = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for t_idx in range(tl.cdiv(S, BLOCK_M)):
        ts = t_idx * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_ts = ts < S
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

        bias_s = tl.load(dq_base + qs * stride_s, mask=mask_qs, other=0.0).to(tl.float32)

        acc_wQ = tl.dot(tl.trans(W), Q_s, acc=acc_wQ)
        PB = P * bias_s[:, None]
        acc_pbQ = tl.dot(tl.trans(PB), Q_s, acc=acc_pbQ)

    dK_val = scale * (acc_wQ - acc_pbQ)
    tl.store(dk_base + qs[:, None] * stride_s + ds[None, :] * stride_d,
             dK_val.to(dtype=tl.bfloat16), mask=mask_qsd)


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
        L:    [B, H, S] or [B, H, S, 1] float32  (logsumexp)
        dQ:   [B, H, S, d] bfloat16  (output, preallocated)
        dK:   [B, H, S, d] bfloat16  (output, preallocated)
        dV:   [B, H, S, d] bfloat16  (output, preallocated)
    """
    torch.cuda.set_device(Q.device)

    B, H, S, D = Q.shape

    # Ensure contiguous for predictable strides
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

    stride_b = Q.stride(0)
    stride_h = Q.stride(1)
    stride_s = Q.stride(2)
    stride_d = Q.stride(3)

    stride_l_b = L.stride(0)
    stride_l_h = L.stride(1)
    stride_l_s = L.stride(2)

    scale = 1.0 / math.sqrt(float(D))

    BLOCK_M = 64
    BLOCK_D = 128
    num_warps = 4
    num_stages = 2

    num_bh = B * H
    num_s_tiles = triton.cdiv(S, BLOCK_M)

    dV_grid = (num_bh, num_s_tiles)

    # Phase 1: Compute dV
    _dv_kernel[dV_grid](
        Q, K, dO, L, dV,
        stride_b, stride_h, stride_s, stride_d,
        stride_l_b, stride_l_h, stride_l_s,
        S, D, scale,
        BLOCK_M=BLOCK_M,
        BLOCK_D=BLOCK_D,
        H=H,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    # Phase 2: Compute dQ and store bias at dQ[:,:, :,0]
    _dq_kernel[dV_grid](
        Q, K, V, dO, L, dQ,
        stride_b, stride_h, stride_s, stride_d,
        stride_l_b, stride_l_h, stride_l_s,
        S, D, scale,
        BLOCK_M=BLOCK_M,
        BLOCK_D=BLOCK_D,
        H=H,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    # Phase 3: Compute dK (reads bias from dQ[:, :, :, 0])
    _dk_kernel[dV_grid](
        Q, K, V, dO, L, dQ, dK,
        stride_b, stride_h, stride_s, stride_d,
        stride_l_b, stride_l_h, stride_l_s,
        S, D, scale,
        BLOCK_M=BLOCK_M,
        BLOCK_D=BLOCK_D,
        H=H,
        num_warps=num_warps,
        num_stages=num_stages,
    )