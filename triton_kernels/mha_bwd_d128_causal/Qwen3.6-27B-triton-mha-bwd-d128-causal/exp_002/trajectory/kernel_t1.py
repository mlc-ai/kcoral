import torch
import triton
import triton.language as tl
import math


@triton.jit
def _mha_bwd_dq_kernel(
    Q, K, V, dO, L, dQ,
    stride_hq, stride_sq, stride_dq,
    stride_hk, stride_sk, stride_dk,
    stride_hv, stride_sv, stride_dv,
    stride_hdO, stride_sdO, stride_ddO,
    stride_hL, stride_sL,
    stride_hdQ, stride_sdQ, stride_ddQ,
    S,
    B_H,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    # Linear head index
    b_idx = pid_bh % (B_H // 48)
    h_idx = pid_bh // (B_H // 48)

    q_off = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    d_off = tl.arange(0, BLOCK_D)
    q_mask_2d = (q_off[:, None] < S) & (d_off[None, :] < BLOCK_D)
    q_mask_1d = q_off < S

    bh_q = b_idx * stride_hq + h_idx * stride_hq if False else pid_bh * stride_hq
    bh_k = pid_bh * stride_hk
    bh_v = pid_bh * stride_hv
    bh_dO = pid_bh * stride_hdO
    bh_L = pid_bh * stride_hL
    bh_dQ = pid_bh * stride_hdQ

    # Load Q tile [BLOCK_M, BLOCK_D] -> fp32
    q_ptrs = Q + bh_q + q_off[:, None] * stride_sq + d_off[None, :] * stride_dq
    Q_tile = tl.load(q_ptrs, mask=q_mask_2d, other=0.0).to(tl.float32)

    # Load dO tile [BLOCK_M, BLOCK_D] -> fp32
    dO_ptrs = dO + bh_dO + q_off[:, None] * stride_sdO + d_off[None, :] * stride_ddO
    dO_tile = tl.load(dO_ptrs, mask=q_mask_2d, other=0.0).to(tl.float32)

    # Load L row [BLOCK_M] fp32
    l_ptrs = L + bh_L + q_off * stride_sL
    L_row = tl.load(l_ptrs, mask=q_mask_1d, other=0.0)

    # Accumulator for dQ [BLOCK_M, BLOCK_D]
    dQ_acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    n_base = tl.arange(0, BLOCK_N)

    for start_n in range(0, S, BLOCK_N):
        n_off = start_n + n_base
        n_mask_2d = (n_off[:, None] < S) & (d_off[None, :] < BLOCK_D)

        # Load K tile [BLOCK_N, BLOCK_D] -> fp32
        k_ptrs = K + bh_k + n_off[:, None] * stride_sk + d_off[None, :] * stride_dk
        K_tile = tl.load(k_ptrs, mask=n_mask_2d, other=0.0).to(tl.float32)

        # Load V tile [BLOCK_N, BLOCK_D] -> fp32
        v_ptrs = V + bh_v + n_off[:, None] * stride_sv + d_off[None, :] * stride_dv
        V_tile = tl.load(v_ptrs, mask=n_mask_2d, other=0.0).to(tl.float32)

        # Causal mask: key position <= query position
        causal = n_off[None, :] <= q_off[:, None]
        causal_and_bounds = causal & (n_off[None, :] < S)

        # Compute attention scores [BLOCK_M, BLOCK_N]
        scores = tl.dot(Q_tile, K_tile.T) * scale

        # Apply causal mask and reconstruct attention weights
        masked_scores = tl.where(causal_and_bounds, scores, -float('inf'))
        P = tl.exp(masked_scores - L_row[:, None])
        # Zero out invalid causal positions explicitly
        P = tl.where(causal_and_bounds, P, 0.0)

        # D = dO @ V.T [BLOCK_M, BLOCK_N]
        D_block = tl.dot(dO_tile, V_tile.T)

        # Softmax backward correction: delta = sum(P * D_block, axis=1)
        delta = tl.sum(P * D_block, axis=1)[:, None]
        d_scores = P * (D_block - delta)

        # Accumulate dQ += d_scores @ K
        dQ_acc = tl.dot(d_scores, K_tile, acc=dQ_acc)

    # Store dQ: multiply by scale for chain rule (ds/dQ = K/sqrt(d))
    dQ_ptrs = dQ + bh_dQ + q_off[:, None] * stride_sdQ + d_off[None, :] * stride_ddQ
    tl.store(dQ_ptrs, (dQ_acc * scale).to(tl.bfloat16), mask=q_mask_2d)


@triton.jit
def _mha_bwd_dkv_kernel(
    Q, K, V, dO, L, dV, dK,
    stride_hq, stride_sq, stride_dq,
    stride_hk, stride_sk, stride_dk,
    stride_hv, stride_sv, stride_dv,
    stride_hdO, stride_sdO, stride_ddO,
    stride_hL, stride_sL,
    stride_hdV, stride_sdV, stride_ddV,
    stride_hdK, stride_sdK, stride_ddK,
    S,
    B_H,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)

    n_off = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    d_off = tl.arange(0, BLOCK_D)
    n_mask_2d = (n_off[:, None] < S) & (d_off[None, :] < BLOCK_D)

    bh_q = pid_bh * stride_hq
    bh_k = pid_bh * stride_hk
    bh_v = pid_bh * stride_hv
    bh_dO = pid_bh * stride_hdO
    bh_L = pid_bh * stride_hL
    bh_dV = pid_bh * stride_hdV
    bh_dK = pid_bh * stride_hdK

    # Preload K tile [BLOCK_N, BLOCK_D] -> fp32
    k_ptrs = K + bh_k + n_off[:, None] * stride_sk + d_off[None, :] * stride_dk
    K_tile = tl.load(k_ptrs, mask=n_mask_2d, other=0.0).to(tl.float32)

    # Preload V tile [BLOCK_N, BLOCK_D] -> fp32
    v_ptrs = V + bh_v + n_off[:, None] * stride_sv + d_off[None, :] * stride_dv
    V_tile = tl.load(v_ptrs, mask=n_mask_2d, other=0.0).to(tl.float32)

    dV_acc = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)
    dK_acc = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)

    m_base = tl.arange(0, BLOCK_M)

    for start_m in range(0, S, BLOCK_M):
        m_off = start_m + m_base
        m_mask_2d = (m_off[:, None] < S) & (d_off[None, :] < BLOCK_D)

        # Load Q tile [BLOCK_M, BLOCK_D] -> fp32
        q_ptrs = Q + bh_q + m_off[:, None] * stride_sq + d_off[None, :] * stride_dq
        Q_tile = tl.load(q_ptrs, mask=m_mask_2d, other=0.0).to(tl.float32)

        # Load dO tile [BLOCK_M, BLOCK_D] -> fp32
        dO_ptrs = dO + bh_dO + m_off[:, None] * stride_sdO + d_off[None, :] * stride_ddO
        dO_tile = tl.load(dO_ptrs, mask=m_mask_2d, other=0.0).to(tl.float32)

        # Load L row [BLOCK_M]
        l_ptrs = L + bh_L + m_off * stride_sL
        L_row = tl.load(l_ptrs, mask=(m_off < S), other=0.0)

        # Causal mask: query position >= key position
        causal = m_off[:, None] >= n_off[None, :]
        causal_and_bounds = causal & (m_off[:, None] < S)

        # Compute attention scores and weights
        scores = tl.dot(Q_tile, K_tile.T) * scale
        masked_scores = tl.where(causal_and_bounds, scores, -float('inf'))
        P = tl.exp(masked_scores - L_row[:, None])
        P = tl.where(causal_and_bounds, P, 0.0)

        # dV += P.T @ dO
        dV_acc = tl.dot(P.T, dO_tile, acc=dV_acc)

        # dK contribution via d_scores
        D_block = tl.dot(dO_tile, V_tile.T)
        delta = tl.sum(P * D_block, axis=1)[:, None]
        d_scores = P * (D_block - delta)
        dK_acc = tl.dot(d_scores.T, Q_tile, acc=dK_acc)

    # Store dV
    dV_ptrs = dV + bh_dV + n_off[:, None] * stride_sdV + d_off[None, :] * stride_ddV
    tl.store(dV_ptrs, dV_acc.to(tl.bfloat16), mask=n_mask_2d)

    # Store dK: multiply by scale for chain rule
    dK_ptrs = dK + bh_dK + n_off[:, None] * stride_sdK + d_off[None, :] * stride_ddK
    tl.store(dK_ptrs, (dK_acc * scale).to(tl.bfloat16), mask=n_mask_2d)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Causal multi-head attention backward.
    
    Computes dQ, dK, dV given forward inputs and upstream gradient.
    Uses two-kernel approach: one for dQ, one for dK+dV.
    """
    torch.cuda.set_device(Q.device)

    B, H, S, D = Q.shape
    num_bh = B * H
    scale = 1.0 / math.sqrt(D)

    # Flatten B,H into single batch-head index for grid dimension
    # Use separate strides for B and H to handle contiguous layout [B,H,S,D]
    h_stride_q = Q.stride(1)
    h_stride_k = K.stride(1)
    h_stride_v = V.stride(1)
    h_stride_dO = dO.stride(1)
    h_stride_dq = dQ.stride(1)
    h_stride_dv = dV.stride(1)
    h_stride_dk = dK.stride(1)

    s_q, d_q = Q.stride(2), Q.stride(3)
    s_k, d_k = K.stride(2), K.stride(3)
    s_v, d_v = V.stride(2), V.stride(3)
    s_dO, d_dO = dO.stride(2), dO.stride(3)
    s_dq, d_dq = dQ.stride(2), dQ.stride(3)
    s_dv, d_dv = dV.stride(2), dV.stride(3)
    s_dk, d_dk = dK.stride(2), dK.stride(3)

    # L strides: handle both [B,H,S] and [B,H,S,1]
    h_L = L.stride(1)
    s_L = L.stride(2)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = D

    grid_dq = (triton.cdiv(S, BLOCK_M), num_bh)
    _mha_bwd_dq_kernel[grid_dq](
        Q, K, V, dO, L, dQ,
        h_stride_q, s_q, d_q,
        h_stride_k, s_k, d_k,
        h_stride_v, s_v, d_v,
        h_stride_dO, s_dO, d_dO,
        h_L, s_L,
        h_stride_dq, s_dq, d_dq,
        S,
        num_bh,
        scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=4, num_stages=2,
    )

    grid_dkv = (triton.cdiv(S, BLOCK_N), num_bh)
    _mha_bwd_dkv_kernel[grid_dkv](
        Q, K, V, dO, L, dV, dK,
        h_stride_q, s_q, d_q,
        h_stride_k, s_k, d_k,
        h_stride_v, s_v, d_v,
        h_stride_dO, s_dO, d_dO,
        h_L, s_L,
        h_stride_dv, s_dv, d_dv,
        h_stride_dk, s_dk, d_dk,
        S,
        num_bh,
        scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=4, num_stages=2,
    )