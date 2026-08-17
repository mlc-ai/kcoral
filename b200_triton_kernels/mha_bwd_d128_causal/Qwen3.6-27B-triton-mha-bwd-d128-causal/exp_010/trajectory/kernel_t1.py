import math
import torch
import triton
import triton.language as tl


@triton.jit
def _dKV_kernel(
    Q, K, V, dO, O, L, dK, dV,
    B, H, S, D_dim, scale,
    stride_Qb, stride_Qh, stride_Qs, stride_Qd,
    stride_Kb, stride_Kh, stride_Ks, stride_Kd,
    stride_Vb, stride_Vh, stride_Vs, stride_Vd,
    stride_dOb, stride_dOh, stride_dOs, stride_dOd,
    stride_Ob, stride_Oh, stride_Os, stride_Od,
    stride_dKb, stride_dKh, stride_dKs, stride_dKd,
    stride_dVb, stride_dVh, stride_dVs, stride_dVd,
    stride_Lb, stride_Lh, stride_Ls,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """Compute dK and dV. Each program owns one (batch, head, kv-tile)."""
    pid_bh = tl.program_id(0)
    pid_n = tl.program_id(1)

    bid_b = pid_bh // H
    bid_h = pid_bh % H

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    # --- Load Q, dO, O once (they depend only on Q-axis, cached across K/V loop) ---
    q_base = Q + bid_b * stride_Qb + bid_h * stride_Qh
    dO_base = dO + bid_b * stride_dOb + bid_h * stride_dOh
    o_base = O + bid_b * stride_Ob + bid_h * stride_Oh

    mask_m_d = (offs_m[:, None] < S) & (offs_d[None, :] < D_dim)

    tile_q = tl.load(q_base + offs_m[:, None] * stride_Qs + offs_d[None, :] * stride_Qd,
                     mask=mask_m_d, other=0.0).to(tl.float32)
    tile_dO = tl.load(dO_base + offs_m[:, None] * stride_dOs + offs_d[None, :] * stride_dOd,
                      mask=mask_m_d, other=0.0).to(tl.float32)
    tile_O = tl.load(o_base + offs_m[:, None] * stride_Os + offs_d[None, :] * stride_Od,
                     mask=mask_m_d, other=0.0).to(tl.float32)

    # D = rowsum(dO * O) for each query row
    row_dot = tl.sum(tile_dO * tile_O, axis=1)

    l_base = L + bid_b * stride_Lb + bid_h * stride_Lh
    tile_l = tl.load(l_base + offs_m * stride_Ls, mask=(offs_m < S), other=0.0).to(tl.float32)

    acc_dk = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    acc_dv = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)

    for nid in range(tl.cdiv(S, BLOCK_N)):
        offs_n_cur = nid * BLOCK_N + tl.arange(0, BLOCK_N)

        mask_n_d = (offs_n_cur[:, None] < S) & (offs_d[None, :] < D_dim)
        mask_causal = (offs_m[:, None] >= offs_n_cur[None, :])
        valid_mn = mask_causal & (offs_m[:, None] < S) & (offs_n_cur[None, :] < S)

        k_base = K + bid_b * stride_Kb + bid_h * stride_Kh
        tile_k = tl.load(k_base + offs_n_cur[:, None] * stride_Ks + offs_d[None, :] * stride_Kd,
                         mask=mask_n_d, other=0.0).to(tl.float32)

        v_base = V + bid_b * stride_Vb + bid_h * stride_Vh
        tile_v = tl.load(v_base + offs_n_cur[:, None] * stride_Vs + offs_d[None, :] * stride_Vd,
                         mask=mask_n_d, other=0.0).to(tl.float32)

        # S = Q @ K^T  ->  [BLOCK_M, BLOCK_N]
        tile_s = tl.dot(tile_q, tile_k.T)

        # Recover P = exp(S * scale - L)  with causal masking
        s_safe = tl.where(valid_mn, tile_s, -1e5)
        m_new = tl.max(s_safe, axis=1, keepdims=True)
        p = tl.exp(s_safe * scale - m_new) * tl.exp(m_new - tile_l[:, None])
        p = tl.where(valid_mn, p, 0.0)

        # dS = P * (dO @ V^T - D)  with causal masking
        dp_v = tl.dot(tile_dO, tile_v.T)
        ds = tl.where(valid_mn, (dp_v - row_dot[:, None]) * scale, 0.0)

        # Gradients: dK += dS^T @ Q,  dV += P^T @ dO
        acc_dk += tl.dot(ds.T, tile_q)
        acc_dv += tl.dot(p.T, tile_dO)

    # Store dK
    out_dtype = tl.bfloat16
    dk_base = dK + bid_b * stride_dKb + bid_h * stride_dKh
    dk_ptrs = dk_base + offs_n[:, None] * stride_dKs + offs_d[None, :] * stride_dKd
    tl.store(dk_ptrs, acc_dk.to(out_dtype), mask=(offs_n[:, None] < S) & (offs_d[None, :] < D_dim))

    # Store dV
    dv_base = dV + bid_b * stride_dVb + bid_h * stride_dVh
    dv_ptrs = dv_base + offs_n[:, None] * stride_dVs + offs_d[None, :] * stride_dVd
    tl.store(dv_ptrs, acc_dv.to(out_dtype), mask=(offs_n[:, None] < S) & (offs_d[None, :] < D_dim))


@triton.jit
def _dQ_kernel(
    Q, K, V, dO, O, L, dQ,
    B, H, S, D_dim, scale,
    stride_Qb, stride_Qh, stride_Qs, stride_Qd,
    stride_Kb, stride_Kh, stride_Ks, stride_Kd,
    stride_Vb, stride_Vh, stride_Vs, stride_Vd,
    stride_dOb, stride_dOh, stride_dOs, stride_dOd,
    stride_Ob, stride_Oh, stride_Os, stride_Od,
    stride_dQb, stride_dQh, stride_dQs, stride_dQd,
    stride_Lb, stride_Lh, stride_Ls,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """Compute dQ. Each program owns one (batch, head, q-tile)."""
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    bid_b = pid_bh // H
    bid_h = pid_bh % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    mask_m_d = (offs_m[:, None] < S) & (offs_d[None, :] < D_dim)

    q_base = Q + bid_b * stride_Qb + bid_h * stride_Qh
    tile_q = tl.load(q_base + offs_m[:, None] * stride_Qs + offs_d[None, :] * stride_Qd,
                     mask=mask_m_d, other=0.0).to(tl.float32)

    dO_base = dO + bid_b * stride_dOb + bid_h * stride_dOh
    tile_dO = tl.load(dO_base + offs_m[:, None] * stride_dOs + offs_d[None, :] * stride_dOd,
                      mask=mask_m_d, other=0.0).to(tl.float32)

    o_base = O + bid_b * stride_Ob + bid_h * stride_Oh
    tile_O = tl.load(o_base + offs_m[:, None] * stride_Os + offs_d[None, :] * stride_Od,
                     mask=mask_m_d, other=0.0).to(tl.float32)

    row_dot = tl.sum(tile_dO * tile_O, axis=1)

    l_base = L + bid_b * stride_Lb + bid_h * stride_Lh
    tile_l = tl.load(l_base + offs_m * stride_Ls, mask=(offs_m < S), other=0.0).to(tl.float32)

    acc_dq = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for nid in range(tl.cdiv(S, BLOCK_N)):
        offs_n_cur = nid * BLOCK_N + tl.arange(0, BLOCK_N)

        mask_n_d = (offs_n_cur[:, None] < S) & (offs_d[None, :] < D_dim)
        valid_mn = (offs_m[:, None] >= offs_n_cur[None, :]) & (offs_m[:, None] < S) & (offs_n_cur[None, :] < S)

        k_base = K + bid_b * stride_Kb + bid_h * stride_Kh
        tile_k = tl.load(k_base + offs_n_cur[:, None] * stride_Ks + offs_d[None, :] * stride_Kd,
                         mask=mask_n_d, other=0.0).to(tl.float32)

        v_base = V + bid_b * stride_Vb + bid_h * stride_Vh
        tile_v = tl.load(v_base + offs_n_cur[:, None] * stride_Vs + offs_d[None, :] * stride_Vd,
                         mask=mask_n_d, other=0.0).to(tl.float32)

        tile_s = tl.dot(tile_q, tile_k.T)

        s_safe = tl.where(valid_mn, tile_s, -1e5)
        m_new = tl.max(s_safe, axis=1, keepdims=True)
        p = tl.exp(s_safe * scale - m_new) * tl.exp(m_new - tile_l[:, None])
        p = tl.where(valid_mn, p, 0.0)

        dp_v = tl.dot(tile_dO, tile_v.T)
        ds = tl.where(valid_mn, (dp_v - row_dot[:, None]) * scale, 0.0)

        acc_dq += tl.dot(ds, tile_k)

    # Store dQ
    dq_base = dQ + bid_b * stride_dQb + bid_h * stride_dQh
    dq_ptrs = dq_base + offs_m[:, None] * stride_dQs + offs_d[None, :] * stride_dQd
    tl.store(dq_ptrs, acc_dq.to(tl.bfloat16), mask=mask_m_d)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Causal multi-head attention backward pass.
    
    Computes dQ, dK, dV given Q, K, V, forward output O, upstream gradient dO, and log-sum-exp L.
    Uses two-kernel FlashAttention-style approach:
      1. dKV kernel: each program fixes one (B,H,kv-tile), reduces over all q-positions
      2. dQ kernel: each program fixes one (B,H,q-tile), reduces over all kv-positions
    No extra allocations; D=rowsum(dO*O) is computed inline.
    """
    torch.cuda.set_device(Q.device)

    B, H, S, D_dim = Q.shape
    attn_scale = 1.0 / math.sqrt(D_dim)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = 128

    # Zero-initialize dK and dV (modified in-place, no allocation)
    dK.zero_()
    dV.zero_()

    # Strides (all tensors share the same layout [B, H, S, D] or [B, H, S])
    sq = Q.stride(); sk = K.stride(); sv = V.stride()
    sdO = dO.stride(); sO = O.stride()
    sdQ = dQ.stride(); sdK = dK.stride(); sdV = dV.stride()
    sL = L.stride()

    BH = B * H

    # Kernel 1: dK and dV
    grid_dKV = (BH, triton.cdiv(S, BLOCK_N))
    _dKV_kernel[grid_dKV](
        Q, K, V, dO, O, L, dK, dV,
        B, H, S, D_dim, attn_scale,
        sq[0], sq[1], sq[2], sq[3],
        sk[0], sk[1], sk[2], sk[3],
        sv[0], sv[1], sv[2], sv[3],
        sdO[0], sdO[1], sdO[2], sdO[3],
        sO[0], sO[1], sO[2], sO[3],
        sdK[0], sdK[1], sdK[2], sdK[3],
        sdV[0], sdV[1], sdV[2], sdV[3],
        sL[0], sL[1], sL[2],
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=8, num_stages=3,
    )

    # Kernel 2: dQ
    grid_dQ = (BH, triton.cdiv(S, BLOCK_M))
    _dQ_kernel[grid_dQ](
        Q, K, V, dO, O, L, dQ,
        B, H, S, D_dim, attn_scale,
        sq[0], sq[1], sq[2], sq[3],
        sk[0], sk[1], sk[2], sk[3],
        sv[0], sv[1], sv[2], sv[3],
        sdO[0], sdO[1], sdO[2], sdO[3],
        sO[0], sO[1], sO[2], sO[3],
        sdQ[0], sdQ[1], sdQ[2], sdQ[3],
        sL[0], sL[1], sL[2],
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=8, num_stages=3,
    )