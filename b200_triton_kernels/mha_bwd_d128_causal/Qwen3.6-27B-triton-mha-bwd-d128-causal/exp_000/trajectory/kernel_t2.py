import torch
import triton
import triton.language as tl


@triton.jit
def _dq_and_bias_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, O_ptr,
    dQ_ptr, dV_ptr,
    stride_b, stride_h, stride_s, stride_d,
    stride_l_b, stride_l_h, stride_l_s,
    S, D,
    scale,
    BLOCK_S: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr,
    H: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_s = tl.program_id(1)

    pid_b = pid_bh // H
    pid_h = pid_bh % H

    qs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    ds = tl.arange(0, BLOCK_D)

    mask_qs = qs < S
    mask_ds = ds < D
    mask_qs_ds = mask_qs[:, None] & mask_ds[None, :]

    bh_off = pid_b * stride_b + pid_h * stride_h

    q_base = Q_ptr + bh_off
    k_base = K_ptr + bh_off
    v_base = V_ptr + bh_off
    do_base = dO_ptr + bh_off
    o_base = O_ptr + bh_off
    dq_base = dQ_ptr + bh_off
    dv_base = dV_ptr + bh_off
    l_base = L_ptr + pid_b * stride_l_b + pid_h * stride_l_h

    # Load Q, dO, O for this s_block (constant across t-loop)
    Q_s = tl.load(q_base + qs[:, None] * stride_s + ds[None, :] * stride_d,
                  mask=mask_qs_ds, other=0.0)
    dO_s = tl.load(do_base + qs[:, None] * stride_s + ds[None, :] * stride_d,
                   mask=mask_qs_ds, other=0.0)
    O_s = tl.load(o_base + qs[:, None] * stride_s + ds[None, :] * stride_d,
                  mask=mask_qs_ds, other=0.0)

    # Load L for this s_block
    L_s = tl.load(l_base + qs * stride_l_s, mask=mask_qs, other=0.0)

    # Accumulators
    acc_wK = tl.zeros((BLOCK_S, BLOCK_D), dtype=tl.float32)
    acc_PK = tl.zeros((BLOCK_S, BLOCK_D), dtype=tl.float32)
    acc_bias = tl.zeros((BLOCK_S,), dtype=tl.float32)

    num_t_tiles = tl.cdiv(S, BLOCK_T)
    for t_idx in range(0, num_t_tiles):
        ts = t_idx * BLOCK_T + tl.arange(0, BLOCK_T)
        mask_ts = ts < S
        mask_ts_ds = mask_ts[:, None] & mask_ds[None, :]

        K_t = tl.load(k_base + ts[:, None] * stride_s + ds[None, :] * stride_d,
                       mask=mask_ts_ds, other=0.0)
        V_t = tl.load(v_base + ts[:, None] * stride_s + ds[None, :] * stride_d,
                       mask=mask_ts_ds, other=0.0)

        # scores = Q @ K^T * scale  (tl.dot(A,M,K), B,N,K) = A @ B^T
        scores = tl.dot(Q_s, K_t) * scale

        # Causal mask: attend only to positions t <= s
        causal_mask = qs[:, None] >= ts[None, :]
        valid_mask = mask_qs[:, None] & mask_ts[None, :]

        # Softmax probabilities
        P = tl.exp(scores - L_s[:, None])
        P = tl.where(valid_mask & causal_mask, P, 0.0)

        # Attention gradient: dO @ V^T
        dP = tl.dot(dO_s, V_t)

        # Weighted gradient: w = P * dP
        w = P * dP

        K_t_f32 = K_t.to(tl.float32)

        # acc_wK += w @ K (standard matmul) = tl.dot(w, trans(K))
        acc_wK = tl.dot(w, tl.trans(K_t_f32), acc=acc_wK)
        acc_PK = tl.dot(P, tl.trans(K_t_f32), acc=acc_PK)
        acc_bias += tl.sum(w, axis=1)

    # dQ = scale * (acc_wK - bias[:,None] * acc_PK)
    dQ_val = scale * (acc_wK - acc_bias[:, None] * acc_PK)
    tl.store(dq_base + qs[:, None] * stride_s + ds[None, :] * stride_d,
             dQ_val.to(tl.bfloat16), mask=mask_qs_ds)

    # Store bias to dV[b,h,s,0] for dK kernel to read later
    tl.store(dv_base + qs * stride_s, acc_bias.to(tl.bfloat16), mask=mask_qs)


@triton.jit
def _dk_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr,
    dK_ptr, dV_ptr,
    stride_b, stride_h, stride_s, stride_d,
    stride_l_b, stride_l_h, stride_l_s,
    S, D,
    scale,
    BLOCK_S: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr,
    H: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_t = tl.program_id(1)

    pid_b = pid_bh // H
    pid_h = pid_bh % H

    ts = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    ds = tl.arange(0, BLOCK_D)

    mask_ts = ts < S
    mask_ds = ds < D
    mask_ts_ds = mask_ts[:, None] & mask_ds[None, :]

    bh_off = pid_b * stride_b + pid_h * stride_h

    q_base = Q_ptr + bh_off
    k_base = K_ptr + bh_off
    v_base = V_ptr + bh_off
    do_base = dO_ptr + bh_off
    dk_base = dK_ptr + bh_off
    dv_base = dV_ptr + bh_off
    l_base = L_ptr + pid_b * stride_l_b + pid_h * stride_l_h

    # Load K_t and V_t for this t_block (constant across s-loop)
    K_t = tl.load(k_base + ts[:, None] * stride_s + ds[None, :] * stride_d,
                   mask=mask_ts_ds, other=0.0)
    V_t = tl.load(v_base + ts[:, None] * stride_s + ds[None, :] * stride_d,
                   mask=mask_ts_ds, other=0.0)

    acc_wQ = tl.zeros((BLOCK_T, BLOCK_D), dtype=tl.float32)
    acc_bPQ = tl.zeros((BLOCK_T, BLOCK_D), dtype=tl.float32)

    num_s_tiles = tl.cdiv(S, BLOCK_S)
    for s_idx in range(0, num_s_tiles):
        ss = s_idx * BLOCK_S + tl.arange(0, BLOCK_S)
        mask_ss = ss < S
        mask_ss_ds = mask_ss[:, None] & mask_ds[None, :]

        Q_s = tl.load(q_base + ss[:, None] * stride_s + ds[None, :] * stride_d,
                       mask=mask_ss_ds, other=0.0)
        dO_s = tl.load(do_base + ss[:, None] * stride_s + ds[None, :] * stride_d,
                        mask=mask_ss_ds, other=0.0)

        scores = tl.dot(Q_s, K_t) * scale

        L_s = tl.load(l_base + ss * stride_l_s, mask=mask_ss, other=0.0)

        causal_mask = ss[:, None] >= ts[None, :]
        valid_mask = mask_ss[:, None] & mask_ts[None, :]

        P = tl.exp(scores - L_s[:, None])
        P = tl.where(valid_mask & causal_mask, P, 0.0)

        dP = tl.dot(dO_s, V_t)
        w = P * dP

        # Load bias from dV[b,h,s,0]
        bias_s = tl.load(dv_base + ss * stride_s, mask=mask_ss, other=0.0).to(tl.float32)

        Q_s_f32 = Q_s.to(tl.float32)

        # First term: w^T @ Q
        acc_wQ = tl.dot(tl.trans(w), tl.trans(Q_s_f32), acc=acc_wQ)

        # Second term: (P * bias)^T @ Q
        P_adj = P * bias_s[:, None]
        acc_bPQ = tl.dot(tl.trans(P_adj), tl.trans(Q_s_f32), acc=acc_bPQ)

    dK_val = scale * (acc_wQ - acc_bPQ)
    tl.store(dk_base + ts[:, None] * stride_s + ds[None, :] * stride_d,
             dK_val.to(tl.bfloat16), mask=mask_ts_ds)


@triton.jit
def _dv_kernel(
    Q_ptr, K_ptr, dO_ptr, L_ptr,
    dV_ptr,
    stride_b, stride_h, stride_s, stride_d,
    stride_l_b, stride_l_h, stride_l_s,
    S, D,
    scale,
    BLOCK_S: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr,
    H: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_t = tl.program_id(1)

    pid_b = pid_bh // H
    pid_h = pid_bh % H

    ts = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    ds = tl.arange(0, BLOCK_D)

    mask_ts = ts < S
    mask_ds = ds < D
    mask_ts_ds = mask_ts[:, None] & mask_ds[None, :]

    bh_off = pid_b * stride_b + pid_h * stride_h

    q_base = Q_ptr + bh_off
    k_base = K_ptr + bh_off
    do_base = dO_ptr + bh_off
    dv_base = dV_ptr + bh_off
    l_base = L_ptr + pid_b * stride_l_b + pid_h * stride_l_h

    K_t = tl.load(k_base + ts[:, None] * stride_s + ds[None, :] * stride_d,
                   mask=mask_ts_ds, other=0.0)

    acc_dv = tl.zeros((BLOCK_T