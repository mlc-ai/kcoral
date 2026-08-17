import math

import torch
import triton
import triton.language as tl


@triton.jit
def _dq_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, O_ptr, L_ptr, dQ_ptr,
    stride_b, stride_h, stride_s, stride_d,
    stride_l_b, stride_l_h, stride_l_s,
    S, D, scale,
    TILE_S: tl.constexpr,
    TILE_D: tl.constexpr,
    H: tl.constexpr,
):
    """dQ[b,h,s,:] = scale * (W @ K - bias[:,None] * P @ K).

    Tile strategy: each program owns one (bh, s_tile). Streams over t-tiles.
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

    qs = pid_s * TILE_S + tl.arange(0, TILE_S)
    ds = tl.arange(0, TILE_D)
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

    acc_wK = tl.zeros((TILE_S, TILE_D), dtype=tl.float32)
    acc_pK = tl.zeros((TILE_S, TILE_D), dtype=tl.float32)

    for t_i in range(tl.cdiv(S, TILE_S)):
        ts = t_i * TILE_S + tl.arange(0, TILE_S)
        mask_ts = ts < S
        mask_dts = mask_ds[:, None] & mask_ts[None, :]

        K_dts = tl.load(k_base + ds[:, None] * stride_d + ts[None, :] * stride_s,
                         mask=mask_dts, other=0.0).to(tl.float32)
        V_dts = tl.load(v_base + ds[:, None] * stride_d + ts[None, :] * stride_s,
                         mask=mask_dts, other=0.0).to(tl.float32)

        scores = tl.dot(Q_s, K_dts) * scale

        P = tl.exp(scores - L_s[:, None])

        causal = qs[:, None] >= ts[None, :]
        valid = mask_qs[:, None] & mask_ts[None, :]
        P = tl.where(valid & causal, P, 0.0)

        dP = tl.dot(dO_s, V_dts)
        W = P * dP

        acc_wK = tl.dot(W, K_dts, acc=acc_wK)
        acc_pK = tl.dot(P, K_dts, acc=acc_pK)

    dQ_val = scale * (acc_wK - bias_s[:, None] * acc_pK)
    tl.store(dq_base + qs[:, None] * stride_s + ds[None, :] * stride_d,
             dQ_val.to(dtype=tl.bfloat16), mask=mask_qsd)


@triton.jit
def _dk_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, O_ptr, L_ptr, dK_ptr,
    stride_b, stride_h, stride_s, stride_d,
    stride_l_b, stride_l_h, stride_l_s,
    S, D, scale,
    TILE_S: tl.constexpr,
    TILE_D: tl.constexpr,
    H: tl.constexpr,
):
    """dK[b,h,t,:] = scale * (W^T @ Q - (P*B)^T @ Q).

    Tile strategy: each program owns one (bh, t_tile). Streams over s-tiles.
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

    ts = pid_t * TILE_S + tl.arange(0, TILE_S)
    ds = tl.arange(0, TILE_D)
    mask_ts = ts < S
    mask_ds = ds < D
    mask_tsd = mask_ts[:, None] & mask_ds[None, :]

    K_dt = tl.load(k_base + ds[:, None] * stride_d + ts[None, :] * stride_s,
                    mask=mask_tsd, other=0.0).to(tl.float32)
    V_dt = tl.load(v_base + ds[:, None] * stride_d + ts[None, :] * stride_s,
                    mask=mask_tsd, other=0.0).to(tl.float32)

    acc_wQ = tl.zeros((TILE_S, TILE_D), dtype=tl.float32)
    acc_pbQ = tl.zeros((TILE_S, TILE_D), dtype=tl.float32)

    for s_i in range(tl.cdiv(S, TILE_S)):
        ss = s_i * TILE_S + tl.arange(0, TILE_S)
        mask_ss = ss < S
        mask_ssd = mask_ss[:, None] & mask_ds[None, :]

        Q_sd = tl.load(q_base + ss[:, None] * stride_s + ds[None, :] * stride_d,
                        mask=mask_ssd, other=0.0).to(tl.float32)
        dO_sd = tl.load(do_base + ss[:, None] * stride_s + ds[None, :] * stride_d,
                         mask=mask_ssd, other=0.0).to(tl.float32)
        O_sd = tl.load(o_base + ss[:, None] * stride_s + ds[None, :] * stride_d,
                        mask=mask_ssd, other=0.0).to(tl.float32)
        L_s = tl.load(l_base + ss * stride_l_s, mask=mask_ss, other=0.0)

        scores = tl.dot(Q_sd, K_dt) * scale
        P = tl.exp(scores - L_s[:, None])

        causal = ss[:, None] >= ts[None, :]
        valid = mask_ss[:, None] & mask_ts[None, :]
        P = tl.where(valid & causal, P, 0.0)

        dP = tl.dot(dO_sd, V_dt)
        W = P * dP

        bias_s = tl.sum(dO_sd * O_sd, axis=1)

        PB = P * bias_s[:, None]

        acc_wQ = tl.dot(tl.trans(W), tl.trans(Q_sd), acc=acc_wQ)
        acc_pbQ = tl.dot(tl.trans(PB), tl.trans(Q_sd), acc=acc_pbQ)

    dK_val = scale * (acc_wQ - acc_pbQ)
    tl.store(dk_base + ts[:, None] * stride_s + ds[None, :] * stride_d,
             dK_val.to(dtype=tl.bfloat16), mask=mask_tsd)


@triton.jit
def _dv_kernel(
    Q_ptr, K_ptr, dO_ptr, L_ptr, dV_ptr,
    stride_b, stride_h, stride_s, stride_d,
    stride_l_b, stride_l_h, stride_l_s,
    S, D, scale,
    TILE_S: tl.constexpr,
    TILE_D: tl.constexpr,
    H: tl.constexpr,
):
    """dV[b,h,t,:] = P^T @ dO.

    Tile strategy: each program owns one (bh, t_tile). Streams over s-tiles.
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

    ts = pid_t * TILE_S + tl.arange(0, TILE_S)
    ds = tl.arange(0, TILE_D)
    mask_ts = ts < S
    mask_ds = ds < D
    mask_tsd = mask_ts[:, None] & mask_ds[None, :]

    K_dt = tl.load(k_base + ds[:, None] * stride_d + ts[None, :] * stride_s,
                    mask=mask_tsd, other=0.0).to(tl.float32)

    acc_dv = tl.zeros((TILE_S, TILE_D), dtype=tl.float32)

    for s_i in range(tl.cdiv(S, TILE_S)):
        ss = s_i * TILE_S + tl.arange(0, TILE_S)
        mask_ss = ss < S
        mask_ssd = mask_ss[:, None] & mask_ds[None, :]

        Q_sd = tl.load(q_base + ss[:, None] * stride_s + ds[None, :] * stride_d,
                        mask=mask_ssd, other=0.0).to(tl.float32)
        dO_sd = tl.load(do_base + ss[:, None] * stride_s + ds[None, :] * stride_d,
                         mask=mask_ssd, other=0.0).to(tl.float32)
        L_s = tl.load(l_base + ss * stride_l_s, mask=mask_ss, other=0.0)

        scores = tl.dot(Q_sd, K_dt) * scale
        P = tl.exp(scores - L_s[:, None])

        causal = ss[:, None] >= ts[None, :]
        valid = mask_ss[:, None] & mask_ts[None, :]
        P = tl.where(valid & causal, P, 0.0)

        acc_dv = tl.dot(tl.trans(P), tl.trans(dO_sd), acc=acc_dv)

    tl.store(dv_base + ts[:, None] * stride_s + ds[None, :] * stride_d,
             acc_dv.to(dtype=tl.bfloat16), mask=mask_tsd)


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

    TILE_S = 64
    TILE_D = D

    num_bh = B * H
    num_s_tiles = triton.cdiv(S, TILE_S)

    grid_dq = (num_bh, num_s_tiles)
    grid_dkv = (num_bh, num_s_tiles)

    _dq_kernel[grid_dq](
        Q, K, V, dO, O, L, dQ,
        stride_b, stride_h, stride_s, stride_d,
        stride_l_b, stride_l_h, stride_l_s,
        S, D, scale,
        TILE_S=TILE_S, TILE_D=TILE_D, H=H,
        num_warps=4, num_stages=2,
    )

    _dk_kernel[grid_dkv](
        Q, K, V, dO, O, L, dK,
        stride_b, stride_h, stride_s, stride_d,
        stride_l_b, stride_l_h, stride_l_s,
        S, D, scale,
        TILE_S=TILE_S, TILE_D=TILE_D, H=H,
        num_warps=4, num_stages=2,
    )

    _dv_kernel[grid_dkv](
        Q, K, dO, L, dV,
        stride_b, stride_h, stride_s, stride_d,
        stride_l_b, stride_l_h, stride_l_s,
        S, D, scale,
        TILE_S=TILE_S, TILE_D=TILE_D, H=H,
        num_warps=4, num_stages=2,
    )