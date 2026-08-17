import torch
import triton
import triton.language as tl
import math


@triton.jit
def _mha_bwd_dq_kernel(
    Q, K, V, dO, L, dQ,
    stride_bq, stride_hq, stride_sq, stride_dq,
    stride_bk, stride_hk, stride_sk, stride_dk,
    stride_bv, stride_hv, stride_sv, stride_dv,
    stride_bdO, stride_hdO, stride_sdO, stride_ddO,
    stride_bL, stride_hL, stride_sL,
    stride_bdQ, stride_hdQ, stride_sdQ, stride_ddQ,
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

    bh_q = pid_bh * stride_hq
    bh_k = pid_bh * stride_hk
    bh_v = pid_bh * stride_hv
    bh_dO = pid_bh * stride_hdO
    bh_L = pid_bh * stride_hL
    bh_dQ = pid_bh * stride_hdQ

    Q_tile = tl.load(Q + bh_q + m_off[:, None] * stride_sq + d_off[None, :] * stride_dq,
                     mask=m_mask_2d, other=0.0).to(tl.float32)
    dO_tile = tl.load(dO + bh_dO + m_off[:, None] * stride_sdO + d_off[None, :] * stride_ddO,
                      mask=m_mask_2d, other=0.0).to(tl.float32)
    L_row = tl.load(L + bh_L + m_off * stride_sL, mask=(m_off < S), other=0.0)[:, None]

    dQ_acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    n_base = tl.arange(0, BLOCK_N)

    for start_n in range(0, S, BLOCK_N):
        n_off = start_n + n_base
        n_limit = min(S, start_n + BLOCK_N)
        n_mask_1d = n_off < S
        n_mask_2d = n_mask_1d[:, None] & (d_off[None, :] < BLOCK_D)

        K_tile = tl.load(K + bh_k + n_off[:, None] * stride_sk + d_off[None, :] * stride_dk,
                         mask=n_mask_2d, other=0.0).to(tl.float32)
        V_tile = tl.load(V + bh_v + n_off[:, None] * stride_sv + d_off[None, :] * stride_dv,
                         mask=n_mask_2d, other=0.0).to(tl.float32)

        # Causal: k <= q (key pos <= query pos)
        causal = (n_off[None, :] <= m_off[:, None]) & n_mask_1d[None, :].to(tl.int1)

        scores = tl.dot(Q_tile, K_tile.T) * scale
        att = tl.where(causal, tl.exp(scores - L_row), 0.0)

        D_block = tl.dot(dO_tile, V_tile.T)
        delta = tl.sum(att * D_block, axis=1)[:, None]
        d_scores = att * (D_block - delta)

        dQ_acc += tl.dot(d_scores, K_tile)

    dQ_acc *= scale
    tl.store(dQ + bh_dQ + m_off[:, None] * stride_sdQ + d_off[None, :] * stride_ddQ,
             dQ_acc.to(tl.bfloat16), mask=m_mask_2d)


@triton.jit
def _mha_bwd_dkv_kernel(
    Q, K, V, dO, L, dV, dK,
    stride_bq, stride_hq, stride_sq, stride_dq,
    stride_bk, stride_hk, stride_sk, stride_dk,
    stride_bv, stride_hv, stride_sv, stride_dv,
    stride_bdO, stride_hdO, stride_sdO, stride_ddO,
    stride_bL, stride_hL, stride_sL,
    stride_bdV, stride_hdV, stride_sdV, stride_ddV,
    stride_bdK, stride_hdK, stride_sdK, stride_ddK,
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

    bh_q = pid_bh * stride_hq
    bh_k = pid_bh * stride_hk
    bh_v = pid_bh * stride_hv
    bh_dO = pid_bh * stride_hdO
    bh_L = pid_bh * stride_hL
    bh_dV = pid_bh * stride_hdV
    bh_dK = pid_bh * stride_hdK

    K_tile = tl.load(K + bh_k + n_off[:, None] * stride_sk + d_off[None, :] * stride_dk,
                     mask=n_mask_2d, other=0.0).to(tl.float32)
    V_tile = tl.load(V + bh_v + n_off[:, None] * stride_sv + d_off[None, :] * stride_dv,
                     mask=n_mask_2d, other=0.0).to(tl.float32)

    dV_acc = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    dK_acc = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    m_base = tl.arange(0, BLOCK_M)

    for start_m in range(0, S, BLOCK_M):
        m_off = start_m + m_base
        m_mask_2d = (m_off[:, None] < S) & (d_off[None, :] < BLOCK_D)

        Q_tile = tl.load(Q + bh_q + m_off[:, None] * stride_sq + d_off[None, :] * stride_dq,
                         mask=m_mask_2d, other=0.0).to(tl.float32)
        dO_tile = tl.load(dO + bh_dO + m_off[:, None] * stride_sdO + d_off[None, :] * stride_ddO,
                          mask=m_mask_2d, other=0.0).to(tl.float32)
        L_row = tl.load(L + bh_L + m_off * stride_sL, mask=(m_off < S), other=0.0)[:, None]

        # Causal: q >= k (query pos >= key pos)
        causal = (m_off[:, None] >= n_off[None, :]) & (m_off[:, None] < S)

        scores = tl.dot(Q_tile, K_tile.T) * scale
        att = tl.where(causal, tl.exp(scores - L_row), 0.0)

        dV_acc += tl.dot(att.T, dO_tile)

        D_block = tl.dot(dO_tile, V_tile.T)
        delta = tl.sum(att * D_block, axis=1)[:, None]
        d_scores = att * (D_block - delta)
        dK_acc += tl.dot(d_scores.T, Q_tile)

    dK_acc *= scale
    tl.store(dV + bh_dV + n_off[:, None] * stride_sdV + d_off[None, :] * stride_ddV,
             dV_acc.to(tl.bfloat16), mask=n_mask_2d)
    tl.store(dK + bh_dK + n_off[:, None] * stride_sdK + d_off[None, :] * stride_ddK,
             dK_acc.to(tl.bfloat16), mask=n_mask_2d)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Causal multi-head attention backward."""
    torch.cuda.set_device(Q.device)

    B, H, S, D = Q.shape
    num_bh = B * H
    scale = 1.0 / math.sqrt(D)

    # Strides
    bq, hq, sq, dq_ = Q.stride()
    bk, hk, sk, dk_ = K.stride()
    bv, hv, sv, dv_ = V.stride()
    bdO, hdO, sdO, ddO_ = dO.stride()
    bL, hL, sL = L.stride(0), L.stride(1), L.stride(2)
    bdQ, hdQ, sdQ, ddQ_ = dQ.stride()
    bdV, hdV, sdV, ddV_ = dV.stride()
    bdK, hdK, sdK, ddK_ = dK.stride()

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = D

    grid = (triton.cdiv(S, BLOCK_M), num_bh)
    _mha_bwd_dq_kernel[grid](
        Q, K, V, dO, L, dQ,
        bq, hq, sq, dq_, bk, hk, sk, dk_,
        bv, hv, sv, dv_, bdO, hdO, sdO, ddO_,
        bL, hL, sL, bdQ, hdQ, sdQ, ddQ_,
        S, scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=4, num_stages=2,
    )

    grid = (triton.cdiv(S, BLOCK_N), num_bh)
    _mha_bwd_dkv_kernel[grid](
        Q, K, V, dO, L, dV, dK,
        bq, hq, sq, dq_, bk, hk, sk, dk_,
        bv, hv, sv, dv_, bdO, hdO, sdO, ddO_,
        bL, hL, sL, bdV, hdV, sdV, ddV_,
        bdK, hdK, sdK, ddK_,
        S, scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=4, num_stages=2,
    )