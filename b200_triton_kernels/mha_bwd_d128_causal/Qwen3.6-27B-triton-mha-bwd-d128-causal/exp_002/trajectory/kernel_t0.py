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
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute dQ for causal MHA backward."""
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    q_off = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    d_off = tl.arange(0, BLOCK_D)
    q_mask = q_off[:, None] < S

    bh_off_Q = pid_bh * stride_hq
    bh_off_K = pid_bh * stride_hk
    bh_off_V = pid_bh * stride_hv
    bh_off_dO = pid_bh * stride_hdO
    bh_off_L = pid_bh * stride_hL
    bh_off_dQ = pid_bh * stride_hdQ

    # Load Q tile [BLOCK_M, BLOCK_D]
    q_ptrs = Q + bh_off_Q + q_off[:, None] * stride_sq + d_off[None, :] * stride_dq
    Q_tile = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # Load dO tile [BLOCK_M, BLOCK_D]
    dO_ptrs = dO + bh_off_dO + q_off[:, None] * stride_sdO + d_off[None, :] * stride_ddO
    dO_tile = tl.load(dO_ptrs, mask=q_mask, other=0.0)

    # Load L row [BLOCK_M]
    l_ptrs = L + bh_off_L + q_off * stride_sL
    L_row = tl.load(l_ptrs, mask=q_off < S, other=0.0)

    # Accumulator for dQ [BLOCK_M, BLOCK_D]
    dQ_acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    n_base = tl.arange(0, BLOCK_N)

    for start_n in range(0, S, BLOCK_N):
        n_off = start_n + n_base
        n_mask = n_off[:, None] < S

        # Load K tile [BLOCK_N, BLOCK_D]
        k_ptrs = K + bh_off_K + n_off[:, None] * stride_sk + d_off[None, :] * stride_dk
        K_tile = tl.load(k_ptrs, mask=n_mask, other=0.0)

        # Load V tile [BLOCK_N, BLOCK_D]
        v_ptrs = V + bh_off_V + n_off[:, None] * stride_sv + d_off[None, :] * stride_dv
        V_tile = tl.load(v_ptrs, mask=n_mask, other=0.0)

        # Causal mask: key position <= query position
        causal = n_off[None, :] <= q_off[:, None]

        # Compute attention scores [BLOCK_M, BLOCK_N]
        scores = tl.dot(Q_tile, K_tile.T) * scale

        # Reconstruct attention weights from stored logsumexp
        masked_scores = tl.where(causal, scores, -1.0e20)
        P = tl.exp(masked_scores - L_row[:, None])

        # D = dO @ V.T [BLOCK_M, BLOCK_N]
        D_block = tl.dot(dO_tile, V_tile.T)

        # Softmax backward correction
        delta = tl.sum(P * D_block, axis=1)[:, None]
        d_scores = P * (D_block - delta)

        # Accumulate dQ += d_scores @ K
        dQ_acc = tl.dot(d_scores, K_tile, acc=dQ_acc)

    # Store dQ (multiply by scale for chain rule: d_scores/d_score * d_score/d_Q = K * scale)
    dQ_ptrs = dQ + bh_off_dQ + q_off[:, None] * stride_sdQ + d_off[None, :] * stride_ddQ
    tl.store(dQ_ptrs, (dQ_acc * scale).to(tl.bfloat16), mask=q_mask)


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
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute dV and dK for causal MHA backward."""
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)

    n_off = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    d_off = tl.arange(0, BLOCK_D)
    n_mask = n_off[:, None] < S

    bh_off_Q = pid_bh * stride_hq
    bh_off_K = pid_bh * stride_hk
    bh_off_V = pid_bh * stride_hv
    bh_off_dO = pid_bh * stride_hdO
    bh_off_L = pid_bh * stride_hL
    bh_off_dV = pid_bh * stride_hdV
    bh_off_dK = pid_bh * stride_hdK

    # Preload K tile [BLOCK_N, BLOCK_D] (constant across inner loop)
    k_ptrs = K + bh_off_K + n_off[:, None] * stride_sk + d_off[None, :] * stride_dk
    K_tile = tl.load(k_ptrs, mask=n_mask, other=0.0)

    # Preload V tile [BLOCK_N, BLOCK_D] (constant across inner loop)
    v_ptrs = V + bh_off_V + n_off[:, None] * stride_sv + d_off[None, :] * stride_dv
    V_tile = tl.load(v_ptrs, mask=n_mask, other=0.0)

    dV_acc = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)
    dK_acc = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)

    m_base = tl.arange(0, BLOCK_M)

    for start_m in range(0, S, BLOCK_M):
        m_off = start_m + m_base
        m_mask = m_off[:, None] < S

        # Load Q tile [BLOCK_M, BLOCK_D]
        q_ptrs = Q + bh_off_Q + m_off[:, None] * stride_sq + d_off[None, :] * stride_dq
        Q_tile = tl.load(q_ptrs, mask=m_mask, other=0.0)

        # Load dO tile [BLOCK_M, BLOCK_D]
        dO_ptrs = dO + bh_off_dO + m_off[:, None] * stride_sdO + d_off[None, :] * stride_ddO
        dO_tile = tl.load(dO_ptrs, mask=m_mask, other=0.0)

        # Load L row [BLOCK_M]
        l_ptrs = L + bh_off_L + m_off * stride_sL
        L_row = tl.load(l_ptrs, mask=m_off < S, other=0.0)

        # Causal mask: query position >= key position
        causal = m_off[:, None] >= n_off[None, :]

        # Compute attention scores and weights
        scores = tl.dot(Q_tile, K_tile.T) * scale
        masked_scores = tl.where(causal, scores, -1.0e20)
        P = tl.exp(masked_scores - L_row[:, None])

        # dV += P.T @ dO
        dV_acc = tl.dot(P.T, dO_tile, acc=dV_acc)

        # dK contribution
        D_block = tl.dot(dO_tile, V_tile.T)
        delta = tl.sum(P * D_block, axis=1)[:, None]
        d_scores = P * (D_block - delta)
        dK_acc = tl.dot(d_scores.T, Q_tile, acc=dK_acc)

    # Store dV
    dV_ptrs = dV + bh_off_dV + n_off[:, None] * stride_sdV + d_off[None, :] * stride_ddV
    tl.store(dV_ptrs, dV_acc.to(tl.bfloat16), mask=n_mask)

    # Store dK (multiply by scale for chain rule)
    dK_ptrs = dK + bh_off_dK + n_off[:, None] * stride_sdK + d_off[None, :] * stride_ddK
    tl.store(dK_ptrs, (dK_acc * scale).to(tl.bfloat16), mask=n_mask)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Causal multi-head attention backward.

    Computes gradients dQ, dK, dV given:
    - Q, K, V: forward attention inputs [B, H, S, D] bf16
    - O: forward output (unused in backward derivation)
    - dO: upstream gradient [B, H, S, D] bf16
    - L: log-sum-exp of scaled scores [B, H, S] or [B, H, S, 1] fp32

    Algorithm:
    1. Reconstruct attention weights P = exp(QK^T/sqrt(d) - L) with causal mask
    2. Compute dV = P^T @ dO
    3. Compute d_scores = P .* (dO @ V^T - mean(P .* (dO @ V^T)))
    4. Compute dQ = d_scores @ K / sqrt(d)
    5. Compute dK = d_scores^T @ Q / sqrt(d)
    """
    torch.cuda.set_device(Q.device)

    B, H, S, D = Q.shape
    num_bh = B * H
    scale = 1.0 / math.sqrt(D)

    # Extract strides from tensors
    s_q, d_q = Q.stride(2), Q.stride(3)
    s_k, d_k = K.stride(2), K.stride(3)
    s_v, d_v = V.stride(2), V.stride(3)
    s_dO, d_dO = dO.stride(2), dO.stride(3)
    s_dq, d_dq = dQ.stride(2), dQ.stride(3)
    s_dv, d_dv = dV.stride(2), dV.stride(3)
    s_dk, d_dk = dK.stride(2), dK.stride(3)

    h_q = Q.stride(1)
    h_k = K.stride(1)
    h_v = V.stride(1)
    h_dO = dO.stride(1)
    h_dq = dQ.stride(1)
    h_dv = dV.stride(1)
    h_dk = dK.stride(1)

    # L strides: works for both [B,H,S] and [B,H,S,1]
    h_L = L.stride(1)
    s_L = L.stride(2)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = D

    # Launch dQ kernel: tile over query blocks for L2-friendly iteration order
    grid_dq = (triton.cdiv(S, BLOCK_M), num_bh)
    _mha_bwd_dq_kernel[grid_dq](
        Q, K, V, dO, L, dQ,
        h_q, s_q, d_q,
        h_k, s_k, d_k,
        h_v, s_v, d_v,
        h_dO, s_dO, d_dO,
        h_L, s_L,
        h_dq, s_dq, d_dq,
        S,
        scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=4, num_stages=2,
    )

    # Launch dKV kernel: tile over key blocks
    grid_dkv = (triton.cdiv(S, BLOCK_N), num_bh)
    _mha_bwd_dkv_kernel[grid_dkv](
        Q, K, V, dO, L, dV, dK,
        h_q, s_q, d_q,
        h_k, s_k, d_k,
        h_v, s_v, d_v,
        h_dO, s_dO, d_dO,
        h_L, s_L,
        h_dv, s_dv, d_dv,
        h_dk, s_dk, d_dk,
        S,
        scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=4, num_stages=2,
    )