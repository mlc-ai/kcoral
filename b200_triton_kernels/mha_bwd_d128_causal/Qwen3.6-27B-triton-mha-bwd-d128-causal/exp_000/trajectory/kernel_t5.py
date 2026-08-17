import math

import torch
import triton
import triton.language as tl


@triton.jit
def _dv_kernel(
    Q_ptr, K_ptr, dO_ptr, L_ptr, dV_ptr,
    stride_b, stride_h, stride_s, stride_d,
    stride_l_b, stride_l_h, stride_l_s,
    S, D, scale,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
    H: tl.constexpr,
):
    """dV[t,:] = sum_s P[s,t] * dO[s,:]

    Grid: (B*H, ceil(S/BLOCK_S)).  pid(0) = bh, pid(1) = t_tile.
    """
    pid_bh = tl.program_id(0)
    pid_t = tl.program_id(1)

    pid_b = pid_bh // H
    pid_h = pid_bh % H

    bh_off = pid_b * stride_b + pid_h * stride_h

    q_base = Q_ptr + bh_off
    k_base = K_ptr + bh_off
    do_base = dO_ptr + bh_off
    dv_base = dV_ptr + bh_off
    l_base = L_ptr + pid_b * stride_l_b + pid_h * stride_l_h

    ts = pid_t * BLOCK_S + tl.arange(0, BLOCK_S)
    ds = tl.arange(0, BLOCK_D)
    mask_ts = ts < S
    mask_ds = ds < D
    mask_tsd = mask_ts[:, None] & mask_ds[None, :]

    K_t = tl.load(k_base + ts[:, None] * stride_s + ds[None, :] * stride_d,
                   mask=mask_tsd, other=0.0).to(tl.float32)

    acc_dv = tl.zeros((BLOCK_S, BLOCK_D), dtype=tl.float32)

    num_s_tiles = tl.cdiv(S, BLOCK_S)
    for s_idx in range(num_s_tiles):
        ss = s_idx * BLOCK_S + tl.arange(0, BLOCK_S)
        mask_ss = ss < S
        mask_ssd = mask_ss[:, None] & mask_ds[None, :]

        Q_s = tl.load(q_base + ss[:, None] * stride_s + ds[None, :] * stride_d,
                       mask=mask_ssd, other=0.0).to(tl.float32)
        dO_s = tl.load(do_base + ss[:, None] * stride_s + ds[None, :] * stride_d,
                        mask=mask_ssd, other=0.0).to(tl.float32)
        L_s = tl.load(l_base + ss * stride_l_s, mask=mask_ss, other=0.0)

        scores = tl.dot(Q_s, K_t) * scale
        P = tl.exp(scores - L_s[:, None])

        causal = ss[:, None] >= ts[None, :]
        valid = mask_ss[:, None] & mask_ts[None, :]
        P = tl.where(valid & causal, P, 0.0)

        # dV += P^T @ dO_s  [ts,D] += dot(trans(P)[ts,ss], trans(dO)[D,ss])
        acc_dv = tl.dot(tl.trans(P), tl.trans(dO_s), acc=acc_dv)

    tl.store(dv_base + ts[:, None] * stride_s + ds[None, :] * stride_d,
             acc_dv.to(dtype=tl.bfloat16), mask=mask_tsd)


@triton.jit
def _dq_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, O_ptr, L_ptr, dQ_ptr,
    stride_b, stride_h, stride_s, stride_d,
    stride_l_b, stride_l_h, stride_l_s,
    S, D, scale,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
    H: tl.constexpr,
):
    """dQ[s,:] = scale * (W @ K - bias[:,None] * P @ K)

    Bias B[s] = dO[s,:] @ O[s,:]  (per-row dot, independent of t).

    Grid: (B*H, ceil(S/BLOCK_S)).  pid(0) = bh, pid(1) = s_tile.
    """
    pid_bh = tl.program_id(0)
    pid_s = tl.program_id(1)

    pid_b = pid_bh // H
    pid_h = pid_bh % H

    bh_off = pid_b * stride_b + pid_h * stride_h

    q_base = Q_ptr + bh_off
    k_base = K_ptr + bh_off
    v_base = V_ptr + bh_off
    do_base = dO_ptr + bh_off
    o_base = O_ptr + bh_off
    dq_base = dQ_ptr + bh_off
    l_base = L_ptr + pid_b * stride_l_b + pid_h * stride_l_h

    qs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    ds = tl.arange(0, BLOCK_D)
    mask_qs = qs < S
    mask_ds = ds < D
    mask_qsd = mask_qs[:, None] & mask_ds[None, :]

    Q_s = tl.load(q_base + qs[:, None] * stride_s + ds[None, :] * stride_d,
                  mask=mask_qsd, other=0.0).to(tl.float32)
    dO_s = tl.load(do_base + qs[:, None] * stride_s + ds[None, :] * stride_d,
                   mask=mask_qsd, other=0.0).to(tl.float32)
    O_s = tl.load(o_base + qs[:, None] * stride_s + ds[None, :] * stride_d,
                  mask=mask_qsd, other=0.0).to(tl.float32)
    L_s = tl.load(l_base + qs * stride_l_s, mask=mask_qs, other=0.0)

    bias_s = tl.sum(dO_s * O_s, axis=1)

    acc_wK = tl.zeros((BLOCK_S, BLOCK_D), dtype=tl.float32)
    acc_pK = tl.zeros((BLOCK_S, BLOCK_D), dtype=tl.float32)

    num_t_tiles = tl.cdiv(S, BLOCK_S)
    for t_idx in range(num_t_tiles):
        ts = t_idx * BLOCK_S + tl.arange(0, BLOCK_S)
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

        # W @ K_t  ->  tl.dot(W[ss,ts], trans(K_t)[D,ts]) -> [ss,D]
        acc_wK = tl.dot(W, tl.trans(K_t), acc=acc_wK)
        acc_pK = tl.dot(P, tl.trans(K_t), acc=acc_pK)

    dQ_val = scale * (acc_wK - bias_s[:, None] * acc_pK)
    tl.store(dq_base + qs[:, None] * stride_s + ds[None, :] * stride_d,
             dQ_val.to(dtype=tl.bfloat16), mask=mask_qsd)


@triton.jit
def _dk_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, O_ptr, L_ptr, dK_ptr,
    stride_b, stride_h, stride_s, stride_d,
    stride_l_b, stride_l_h, stride_l_s,
    S, D, scale,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
    H: tl.constexpr,
):
    """dK[t,:] = scale * (W^T @ Q - (P*B)^T @ Q)

    Bias B[s] = dO[s,:] @ O[s,:]  computed per s-tile.

    Grid: (B*H, ceil(S/BLOCK_S)).  pid(0) = bh, pid(1) = t_tile.
    """
    pid_bh = tl.program_id(0)
    pid_t = tl.program_id(1)

    pid_b = pid_bh // H
    pid_h = pid_bh % H

    bh_off = pid_b * stride_b + pid_h * stride_h

    q_base = Q_ptr + bh_off
    k_base = K_ptr + bh_off
    v_base = V_ptr + bh_off
    do_base = dO_ptr + bh_off
    o_base = O_ptr + bh_off
    dk_base = dK_ptr + bh_off
    l_base = L_ptr + pid_b * stride_l_b + pid_h * stride_l_h

    ts = pid_t * BLOCK_S + tl.arange(0, BLOCK_S)
    ds = tl.arange(0, BLOCK_D)
    mask_ts = ts < S
    mask_ds = ds < D
    mask_tsd = mask_ts[:, None] & mask_ds[None, :]

    K_t = tl.load(k_base + ts[:, None] * stride_s + ds[None, :] * stride_d,
                   mask=mask_tsd, other=0.0).to(tl.float32)
    V_t = tl.load(v_base + ts[:, None] * stride_s + ds[None, :] * stride_d,
                   mask=mask_tsd, other=0.0).to(tl.float32)

    acc_wQ = tl.zeros((BLOCK_S, BLOCK_D), dtype=tl.float32)
    acc_pbQ = tl.zeros((BLOCK_S, BLOCK_D), dtype=tl.float32)

    num_s_tiles = tl.cdiv(S, BLOCK_S)
    for s_idx in range(num_s_tiles):
        ss = s_idx * BLOCK_S + tl.arange(0, BLOCK_S)
        mask_ss = ss < S
        mask_ssd = mask_ss[:, None] & mask_ds[None, :]

        Q_s = tl.load(q_base + ss[:, None] * stride_s + ds[None, :] * stride_d,
                       mask=mask_ssd, other=0.0).to(tl.float32)
        dO_s = tl.load(do_base + ss[:, None] * stride_s + ds[None, :] * stride_d,
                        mask=mask_ssd, other=0.0).to(tl.float32)
        O_s = tl.load(o_base + ss[:, None] * stride_s + ds[None, :] * stride_d,
                       mask=mask_ssd, other=0.0).to(tl.float32)
        L_s = tl.load(l_base + ss * stride_l_s, mask=mask_ss, other=0.0)

        bias_s = tl.sum(dO_s * O_s, axis=1)

        scores = tl.dot(Q_s, K_t) * scale
        P = tl.exp(scores - L_s[:, None])

        causal = ss[:, None] >= ts[None, :]
        valid = mask_ss[:, None] & mask_ts[None, :]
        P = tl.where(valid & causal, P, 0.0)

        dP = tl.dot(dO_s, V_t)
        W = P * dP

        # W^T @ Q_s  ->  dot(trans(W)[ts,ss], trans(Q)[D,ss]) -> [ts,D]
        acc_wQ = tl.dot(tl.trans(W), tl.trans(Q_s), acc=acc_wQ)

        PB = P * bias_s[:, None]
        acc_pbQ = tl.dot(tl.trans(PB), tl.trans(Q_s), acc=acc_pbQ)

    dK_val = scale * (acc_wQ - acc_pbQ)
    tl.store(dk_base + ts[:, None] * stride_s + ds[None, :] * stride_d,
             dK_val.to(dtype=tl.bfloat16), mask=mask_tsd)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Causal multi-head attention backward.

    Computes dQ, dK, dV given Q, K, V, O (forward output), dO (upstream grad),
    and L (logsumexp of QK^T/sqrt(d)).
    """
    torch.cuda.set_device(Q.device)

    B, H, S, D = Q.shape

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

    BLOCK_S = 64
    BLOCK_D = 128

    num_bh = B * H
    num_s_tiles = triton.cdiv(S, BLOCK_S)
    grid = (num_bh, num_s_tiles)

    # Phase 1: dV — independent of dQ/dK
    _dv_kernel[grid](
        Q, K, dO, L, dV,
        stride_b, stride_h, stride_s, stride_d,
        stride_l_b, stride_l_h, stride_l_s,
        S, D, scale,
        BLOCK_S=BLOCK_S, BLOCK_D=BLOCK_D, H=H,
        num_warps=4, num_stages=3,
    )

    # Phase 2: dQ — independent of dK
    _dq_kernel[grid](
        Q, K, V, dO, O, L, dQ,
        stride_b, stride_h, stride_s, stride_d,
        stride_l_b, stride_l_h, stride_l_s,
        S, D, scale,
        BLOCK_S=BLOCK_S, BLOCK_D=BLOCK_D, H=H,
        num_warps=4, num_stages=3,
    )

    # Phase 3: dK — independent of others
    _dk_kernel[grid](
        Q, K, V, dO, O, L, dK,
        stride_b, stride_h, stride_s, stride_d,
        stride_l_b, stride_l_h, stride_l_s,
        S, D, scale,
        BLOCK_S=BLOCK_S, BLOCK_D=BLOCK_D, H=H,
        num_warps=4, num_stages=3,
    )