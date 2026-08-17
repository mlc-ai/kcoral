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
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    m_off = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    d_off = tl.arange(0, BLOCK_D)
    m_mask_2d = (m_off[:, None] < S) & (d_off[None, :] < BLOCK_D)

    # Load Q tile -> fp32
    q_ptrs = Q + pid_bh * stride_hq + m_off[:, None] * stride_sq + d_off[None, :] * stride_dq
    Q_tile = tl.load(q_ptrs, mask=m_mask_2d, other=0.0).to(tl.float32)

    # Load dO tile -> fp32
    do_ptrs = dO + pid_bh * stride_hdO + m_off[:, None] * stride_sdO + d_off[None, :] * stride_ddO
    dO_tile = tl.load(do_ptrs, mask=m_mask_2d, other=0.0).to(tl.float32)

    # Load L row -> fp32
    l_ptrs = L + pid_bh * stride_hL + m_off * stride_sL
    L_row = tl.load(l_ptrs, mask=(m_off < S), other=0.0)

    dQ_acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    n_base = tl.arange(0, BLOCK_N)

    for start_n in range(0, S, BLOCK_N):
        n_off = start_n + n_base
        n_mask_2d = (n_off[:, None] < S) & (d_off[None, :] < BLOCK_D)

        # Load K, V tiles -> fp32
        k_ptrs = K + pid_bh * stride_hk + n_off[:, None] * stride_sk + d_off[None, :] * stride_dk
        K_tile = tl.load(k_ptrs, mask=n_mask_2d, other=0.0).to(tl.float32)
        v_ptrs = V + pid_bh * stride_hv + n_off[:, None] * stride_sv + d_off[None, :] * stride_dv
        V_tile = tl.load(v_ptrs, mask=n_mask_2d, other=0.0).to(tl.float32)

        # Causal mask: key pos <= query pos
        causal = n_off[None, :] <= m_off[:, None]

        # Compute attention weights
        scores = tl.dot(Q_tile, K_tile.T) * scale
        masked_scores = tl.where(causal, scores, -float('inf'))
        att = tl.where(causal, tl.exp(masked_scores - L_row[:, None]), 0.0)

        # Backward through attention
        D_block = tl.dot(dO_tile, V_tile.T)
        delta = tl.sum(att * D_block, axis=1)[:, None]
        d_scores = att * (D_block - delta)

        # dQ contribution: d_scores @ K * scale (chain rule: dscores/dQ = K * scale)
        dQ_acc = tl.dot(d_scores, K_tile, acc=dQ_acc)

    # Apply scale factor for the chain rule
    dQ_out = dQ_acc * scale
    tl.store(dQ + pid_bh * stride_hdQ + m_off[:, None] * stride_sdQ + d_off[None, :] * stride_ddQ,
             dQ_out.to(tl.bfloat16), mask=m_mask_2d)


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
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)

    n_off = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    d_off = tl.arange(0, BLOCK_D)
    n_mask_2d = (n_off[:, None] < S) & (d_off[None, :] < BLOCK_D)

    # Preload K, V tiles (constant across m-loop) -> fp32
    k_ptrs = K + pid_bh * stride_hk + n_off[:, None] * stride_sk + d_off[None, :] * stride_dk
    K_tile = tl.load(k_ptrs, mask=n_mask_2d, other=0.0).to(tl.float32)
    v_ptrs = V + pid_bh * stride_hv + n_off[:, None] * stride_sv + d_off[None, :] * stride_dv
    V_tile = tl.load(v_ptrs, mask=n_mask_2d, other=0.0).to(tl.float32)

    dV_acc = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    dK_acc = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    m_base = tl.arange(0, BLOCK_M)

    for start_m in range(0, S, BLOCK_M):
        m_off = start_m + m_base
        m_mask_2d = (m_off[:, None] < S) & (d_off[None, :] < BLOCK_D)

        # Load Q, dO tiles -> fp32
        q_ptrs = Q + pid_bh * stride_hq + m_off[:, None] * stride_sq + d_off[None, :] * stride_dq
        Q_tile = tl.load(q_ptrs, mask=m_mask_2d, other=0.0).to(tl.float32)
        do_ptrs = dO + pid_bh * stride_hdO + m_off[:, None] * stride_sdO + d_off[None, :] * stride_ddO
        dO_tile = tl.load(do_ptrs, mask=m_mask_2d, other=0.0).to(tl.float32)

        # Load L row
        l_ptrs = L + pid_bh * stride_hL + m_off * stride_sL
        L_row = tl.load(l_ptrs, mask=(m_off < S), other=0.0)

        # Causal mask: query pos >= key pos
        causal = m_off[:, None] >= n_off[None, :]

        # Recompute attention weights
        scores = tl.dot(Q_tile, K_tile.T) * scale
        masked_scores = tl.where(causal, scores, -float('inf'))
        att = tl.where(causal, tl.exp(masked_scores - L_row[:, None]), 0.0)

        # dV += att.T @ dO
        dV_acc = tl.dot(att.T, dO_tile, acc=dV_acc)

        # dK contribution via d_scores = att * (dO @ V^T - delta)
        D_block = tl.dot(dO_tile, V_tile.T)
        delta = tl.sum(att * D_block, axis=1)[:, None]
        d_scores = att * (D_block - delta)

        # dK += d_scores^T @ Q * scale (chain rule)
        dK_acc = tl.dot(d_scores.T, Q_tile, acc=dK_acc)

    # Store dV (no extra scale)
    tl.store(dV + pid_bh * stride_hdV + n_off[:, None] * stride_sdV + d_off[None, :] * stride_ddV,
             dV_acc.to(tl.bfloat16), mask=n_mask_2d)

    # Store dK with scale factor for chain rule
    tl.store(dK + pid_bh * stride_hdK + n_off[:, None] * stride_sdK + d_off[None, :] * stride_ddK,
             (dK_acc * scale).to(tl.bfloat16), mask=n_mask_2d)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Causal multi-head attention backward.
    
    Computes gradients dQ, dK, dV given forward inputs Q, K, V,
    forward output O, upstream gradient dO, and log-sum-exp L.
    """
    torch.cuda.set_device(Q.device)

    B, H, S, D = Q.shape
    num_bh = B * H
    scale = 1.0 / math.sqrt(D)

    # Use contiguous B,H,S,D layout strides
    h_stride_q, s_stride_q, d_stride_q = Q.stride(1), Q.stride(2), Q.stride(3)
    h_stride_k, s_stride_k, d_stride_k = K.stride(1), K.stride(2), K.stride(3)
    h_stride_v, s_stride_v, d_stride_v = V.stride(1), V.stride(2), V.stride(3)
    h_stride_dO, s_stride_dO, d_stride_dO = dO.stride(1), dO.stride(2), dO.stride(3)
    h_stride_L, s_stride_L = L.stride(1), L.stride(2)
    h_stride_dq, s_stride_dq, d_stride_dq = dQ.stride(1), dQ.stride(2), dQ.stride(3)
    h_stride_dv, s_stride_dv, d_stride_dv = dV.stride(1), dV.stride(2), dV.stride(3)
    h_stride_dk, s_stride_dk, d_stride_dk = dK.stride(1), dK.stride(2), dK.stride(3)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = D

    # Launch dQ kernel: tile over query position blocks
    grid_dq = (triton.cdiv(S, BLOCK_M), num_bh)
    _mha_bwd_dq_kernel[grid_dq](
        Q, K, V, dO, L, dQ,
        h_stride_q, s_stride_q, d_stride_q,
        h_stride_k, s_stride_k, d_stride_k,
        h_stride_v, s_stride_v, d_stride_v,
        h_stride_dO, s_stride_dO, d_stride_dO,
        h_stride_L, s_stride_L,
        h_stride_dq, s_stride_dq, d_stride_dq,
        S,
        scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=4, num_stages=2,
    )

    # Launch dKV kernel: tile over key position blocks
    grid_dkv = (triton.cdiv(S, BLOCK_N), num_bh)
    _mha_bwd_dkv_kernel[grid_dkv](
        Q, K, V, dO, L, dV, dK,
        h_stride_q, s_stride_q, d_stride_q,
        h_stride_k, s_stride_k, d_stride_k,
        h_stride_v, s_stride_v, d_stride_v,
        h_stride_dO, s_stride_dO, d_stride_dO,
        h_stride_L, s_stride_L,
        h_stride_dv, s_stride_dv, d_stride_dv,
        h_stride_dk, s_stride_dk, d_stride_dk,
        S,
        scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=4, num_stages=2,
    )