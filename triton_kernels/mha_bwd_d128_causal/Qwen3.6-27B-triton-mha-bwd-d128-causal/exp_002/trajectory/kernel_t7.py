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
    q_mask = m_off[:, None] < S

    bh_q = pid_bh * stride_hq
    bh_k = pid_bh * stride_hk
    bh_v = pid_bh * stride_hv
    bh_dO = pid_bh * stride_hdO
    bh_L = pid_bh * stride_hL
    bh_dQ = pid_bh * stride_hdQ

    # Load Q [BLOCK_M, BLOCK_D] -> fp32
    Q_tile = tl.load(bh_q + m_off[:, None] * stride_sq + d_off[None, :] * stride_dq + Q,
                     mask=q_mask, other=0.0).to(tl.float32)

    # Load dO [BLOCK_M, BLOCK_D] -> fp32
    dO_tile = tl.load(bh_dO + m_off[:, None] * stride_sdO + d_off[None, :] * stride_ddO + dO,
                      mask=q_mask, other=0.0).to(tl.float32)

    # Load L [BLOCK_M] -> fp32
    L_row = tl.load(bh_L + m_off * stride_sL + L, mask=(m_off < S), other=0.0)

    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    n_base = tl.arange(0, BLOCK_N)

    for start_n in range(0, S, BLOCK_N):
        n_off = start_n + n_base
        k_mask = n_off[:, None] < S

        K_tile = tl.load(bh_k + n_off[:, None] * stride_sk + d_off[None, :] * stride_dk + K,
                         mask=k_mask, other=0.0).to(tl.float32)
        V_tile = tl.load(bh_v + n_off[:, None] * stride_sv + d_off[None, :] * stride_dv + V,
                         mask=k_mask, other=0.0).to(tl.float32)

        # Causal: key pos <= query pos AND within bounds
        causal = (n_off[None, :] <= m_off[:, None]) & (n_off[None, :] < S)

        scores = tl.dot(Q_tile, K_tile.T) * scale
        att = tl.where(causal, tl.exp(scores - L_row[:, None]), 0.0)

        dOVt = tl.dot(dO_tile, V_tile.T)
        delta = tl.sum(att * dOVt, axis=1)[:, None]
        d_scores = att * (dOVt - delta)

        acc = tl.dot(d_scores, K_tile, acc=acc)

    # Multiply by scale in fp32, THEN convert to bf16
    tl.store(bh_dQ + m_off[:, None] * stride_sdQ + d_off[None, :] * stride_ddQ + dQ,
             (acc * scale).to(tl.bfloat16), mask=q_mask)


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
    k_mask = n_off[:, None] < S

    bh_q = pid_bh * stride_hq
    bh_k = pid_bh * stride_hk
    bh_v = pid_bh * stride_hv
    bh_dO = pid_bh * stride_hdO
    bh_L = pid_bh * stride_hL
    bh_dV = pid_bh * stride_hdV
    bh_dK = pid_bh * stride_hdK

    K_tile = tl.load(bh_k + n_off[:, None] * stride_sk + d_off[None, :] * stride_dk + K,
                     mask=k_mask, other=0.0).to(tl.float32)
    V_tile = tl.load(bh_v + n_off[:, None] * stride_sv + d_off[None, :] * stride_dv + V,
                     mask=k_mask, other=0.0).to(tl.float32)

    acc_dV = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    acc_dK = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    m_base = tl.arange(0, BLOCK_M)

    for start_m in range(0, S, BLOCK_M):
        m_off = start_m + m_base
        q_mask = m_off[:, None] < S

        Q_tile = tl.load(bh_q + m_off[:, None] * stride_sq + d_off[None, :] * stride_dq + Q,
                         mask=q_mask, other=0.0).to(tl.float32)
        dO_tile = tl.load(bh_dO + m_off[:, None] * stride_sdO + d_off[None, :] * stride_ddO + dO,
                          mask=q_mask, other=0.0).to(tl.float32)

        L_row = tl.load(bh_L + m_off * stride_sL + L, mask=(m_off < S), other=0.0)

        # Causal: query pos >= key pos AND within bounds
        causal = (m_off[:, None] >= n_off[None, :]) & (m_off[:, None] < S)

        scores = tl.dot(Q_tile, K_tile.T) * scale
        att = tl.where(causal, tl.exp(scores - L_row[:, None]), 0.0)

        acc_dV = tl.dot(att.T, dO_tile, acc=acc_dV)

        dOVt = tl.dot(dO_tile, V_tile.T)
        delta = tl.sum(att * dOVt, axis=1)[:, None]
        d_scores = att * (dOVt - delta)
        acc_dK = tl.dot(d_scores.T, Q_tile, acc=acc_dK)

    tl.store(bh_dV + n_off[:, None] * stride_sdV + d_off[None, :] * stride_ddV + dV,
             acc_dV.to(tl.bfloat16), mask=k_mask)

    # Multiply by scale in fp32, THEN convert to bf16
    tl.store(bh_dK + n_off[:, None] * stride_sdK + d_off[None, :] * stride_ddK + dK,
             (acc_dK * scale).to(tl.bfloat16), mask=k_mask)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Causal multi-head attention backward."""
    torch.cuda.set_device(Q.device)

    B, H, S, D = Q.shape
    num_bh = B * H
    scale = 1.0 / math.sqrt(D)

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

    grid_dq = (triton.cdiv(S, BLOCK_M), num_bh)
    _mha_bwd_dq_kernel[grid_dq](
        Q, K, V, dO, L, dQ,
        hq, sq, dq, hk, sk, dk, hv, sv, dv,
        hdO, sdO, ddO, hL, sL,
        hdq, sdq, ddq, S, scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=4, num_stages=2,
    )

    grid_dkv = (triton.cdiv(S, BLOCK_N), num_bh)
    _mha_bwd_dkv_kernel[grid_dkv](
        Q, K, V, dO, L, dV, dK,
        hq, sq, dq, hk, sk, dk, hv, sv, dv,
        hdO, sdO, ddO, hL, sL,
        hdv, sdv, ddv, hdk, sdk, ddk, S, scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=4, num_stages=2,
    )