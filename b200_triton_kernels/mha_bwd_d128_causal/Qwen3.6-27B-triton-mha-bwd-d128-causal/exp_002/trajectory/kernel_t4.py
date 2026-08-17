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
    m_mask_qd = m_off[:, None] < S
    m_mask_1d = m_off < S

    # Load Q tile [BLOCK_M, BLOCK_D] -> fp32
    q_ptrs = Q + pid_bh * stride_hq + m_off[:, None] * stride_sq + d_off[None, :] * stride_dq
    Q_tile = tl.load(q_ptrs, mask=m_mask_qd, other=0.0).to(tl.float32)

    # Load dO tile [BLOCK_M, BLOCK_D] -> fp32
    do_ptrs = dO + pid_bh * stride_hdO + m_off[:, None] * stride_sdO + d_off[None, :] * stride_ddO
    dO_tile = tl.load(do_ptrs, mask=m_mask_qd, other=0.0).to(tl.float32)

    # Load L row [BLOCK_M] -> fp32
    l_ptrs = L + pid_bh * stride_hL + m_off * stride_sL
    L_row = tl.load(l_ptrs, mask=m_mask_1d, other=0.0)

    dQ_acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    n_base = tl.arange(0, BLOCK_N)

    for start_n in range(0, S, BLOCK_N):
        n_off = start_n + n_base
        n_mask_2d = (n_off[:, None] < S)

        # Load K tile [BLOCK_N, BLOCK_D] -> fp32
        k_ptrs = K + pid_bh * stride_hk + n_off[:, None] * stride_sk + d_off[None, :] * stride_dk
        K_tile = tl.load(k_ptrs, mask=n_mask_2d, other=0.0).to(tl.float32)

        # Load V tile [BLOCK_N, BLOCK_D] -> fp32
        v_ptrs = V + pid_bh * stride_hv + n_off[:, None] * stride_sv + d_off[None, :] * stride_dv
        V_tile = tl.load(v_ptrs, mask=n_mask_2d, other=0.0).to(tl.float32)

        # Causal mask: key position <= query position  (shape [BLOCK_M, BLOCK_N])
        causal = (n_off[None, :] <= m_off[:, None]).to(tl.int1)

        # Compute raw scores with scale already applied
        scores = tl.dot(Q_tile, K_tile.T) * scale

        # Apply causal mask: -inf for invalid, scores for valid
        masked_scores = tl.where(causal, scores, -float('inf'))
        
        # Compute attention weights: exp(score - L) for valid, 0 for invalid
        att = tl.where(causal, tl.exp(masked_scores - L_row[:, None]), 0.0)

        # Compute D_block = dO @ V.T  [BLOCK_M, BLOCK_N]
        D_block = tl.dot(dO_tile, V_tile.T)

        # Softmax backward: d_scores = att * (D_block - delta)
        # delta[i] = sum_j(att[i,j] * D_block[i,j])
        delta = tl.sum(att * D_block, axis=1)[:, None]
        d_scores = att * (D_block - delta)

        # Accumulate dQ: dQ += d_scores @ K
        dQ_acc = tl.dot(d_scores, K_tile, acc=dQ_acc)

    # Chain rule: multiply by scale (dscores/dQ = K/sqrt(d), and scale = 1/sqrt(d))
    dQ_out = dQ_acc * scale
    
    dq_ptrs = dQ + pid_bh * stride_hdQ + m_off[:, None] * stride_sdQ + d_off[None, :] * stride_ddQ
    tl.store(dq_ptrs, dQ_out.to(tl.bfloat16), mask=m_mask_qd)


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
    n_mask_kd = n_off[:, None] < S

    bh_q = pid_bh * stride_hq
    bh_k = pid_bh * stride_hk
    bh_v = pid_bh * stride_hv
    bh_dO = pid_bh * stride_hdO
    bh_L = pid_bh * stride_hL
    bh_dV = pid_bh * stride_hdV
    bh_dK = pid_bh * stride_hdK

    # Preload K, V tiles [BLOCK_N, BLOCK_D] -> fp32
    K_tile = tl.load(K + bh_k + n_off[:, None] * stride_sk + d_off[None, :] * stride_dk,
                     mask=n_mask_kd, other=0.0).to(tl.float32)
    V_tile = tl.load(V + bh_v + n_off[:, None] * stride_sv + d_off[None, :] * stride_dv,
                     mask=n_mask_kd, other=0.0).to(tl.float32)

    dV_acc = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    dK_acc = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    m_base = tl.arange(0, BLOCK_M)

    for start_m in range(0, S, BLOCK_M):
        m_off = start_m + m_base
        m_mask_qd = (m_off[:, None] < S)

        # Load Q tile [BLOCK_M, BLOCK_D] -> fp32
        Q_tile = tl.load(Q + bh_q + m_off[:, None] * stride_sq + d_off[None, :] * stride_dq,
                         mask=m_mask_qd, other=0.0).to(tl.float32)

        # Load dO tile [BLOCK_M, BLOCK_D] -> fp32
        dO_tile = tl.load(dO + bh_dO + m_off[:, None] * stride_sdO + d_off[None, :] * stride_ddO,
                          mask=m_mask_qd, other=0.0).to(tl.float32)

        # Load L row [BLOCK_M] -> fp32
        L_row = tl.load(L + bh_L + m_off * stride_sL, mask=(m_off < S), other=0.0)

        # Causal mask: query position >= key position  (shape [BLOCK_M, BLOCK_N])
        causal = (m_off[:, None] >= n_off[None, :]).to(tl.int1)

        # Recompute attention weights
        scores = tl.dot(Q_tile, K_tile.T) * scale
        masked_scores = tl.where(causal, scores, -float('inf'))
        att = tl.where(causal, tl.exp(masked_scores - L_row[:, None]), 0.0)

        # dV += att.T @ dO
        dV_acc = tl.dot(att.T, dO_tile, acc=dV_acc)

        # Compute d_scores for dK
        D_block = tl.dot(dO_tile, V_tile.T)
        delta = tl.sum(att * D_block, axis=1)[:, None]
        d_scores = att * (D_block - delta)

        # dK += d_scores^T @ Q
        dK_acc = tl.dot(d_scores.T, Q_tile, acc=dK_acc)

    # Store dV (no extra scale needed)
    tl.store(dV + bh_dV + n_off[:, None] * stride_sdV + d_off[None, :] * stride_ddV,
             dV_acc.to(tl.bfloat16), mask=n_mask_kd)

    # Store dK with scale factor for chain rule
    tl.store(dK + bh_dK + n_off[:, None] * stride_sdK + d_off[None, :] * stride_ddK,
             (dK_acc * scale).to(tl.bfloat16), mask=n_mask_kd)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Causal multi-head attention backward.
    
    Given Q, K, V, O (forward output), dO (upstream grad), L (log-sum-exp),
    computes dQ, dK, dV gradients.
    
    Uses two-kernel decomposition:
    1. dQ kernel: iterates over query blocks, accumulates dQ via d_scores @ K
    2. dKV kernel: iterates over key blocks, accumulates dV via P^T @ dO 
                   and dK via d_scores^T @ Q
    """
    torch.cuda.set_device(Q.device)

    B, H, S, D = Q.shape
    num_bh = B * H
    scale = 1.0 / math.sqrt(D)

    # Extract strides
    hq, sq, dq = Q.stride(1), Q.stride(2), Q.stride(3)
    hk, sk, dk = K.stride(1), K.stride(2), K.stride(3)
    hv, sv, dv = V.stride(1), V.stride(2), V.stride(3)
    hdO, sdO, ddO = dO.stride(1), dO.stride(2), dO.stride(3)
    hL, sL = L.stride(1), L.stride(2)
    hdq, sdq, ddq = dQ.stride(1), dQ.stride(2), dQ.stride(3)
    hdv, sdv, ddv = dV.stride(1), dV.stride(2), dV.stride(3)
    hdk, sdk, ddk = dK.stride(1), dK.stride(2), dK.stride(3)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = D

    # Launch dQ kernel
    grid_dq = (triton.cdiv(S, BLOCK_M), num_bh)
    _mha_bwd_dq_kernel[grid_dq](
        Q, K, V, dO, L, dQ,
        hq, sq, dq, hk, sk, dk, hv, sv, dv,
        hdO, sdO, ddO, hL, sL,
        hdq, sdq, ddq,
        S, scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=4, num_stages=2,
    )

    # Launch dKV kernel
    grid_dkv = (triton.cdiv(S, BLOCK_N), num_bh)
    _mha_bwd_dkv_kernel[grid_dkv](
        Q, K, V, dO, L, dV, dK,
        hq, sq, dq, hk, sk, dk, hv, sv, dv,
        hdO, sdO, ddO, hL, sL,
        hdv, sdv, ddv, hdk, sdk, ddk,
        S, scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=4, num_stages=2,
    )