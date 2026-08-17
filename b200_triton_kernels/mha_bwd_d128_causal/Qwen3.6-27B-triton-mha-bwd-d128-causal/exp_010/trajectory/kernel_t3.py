import math
import torch
import triton
import triton.language as tl


@triton.jit
def _dKV_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, O_ptr, L_ptr, dK_ptr, dV_ptr,
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
    """Compute dK and dV.
    
    Each program owns (bid_b, bid_h, pid_n) -> one KV tile.
    Loops over ALL query blocks, accumulating into dK and dV.
    """
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)

    bid_b = pid_bh // H
    bid_h = pid_bh % H

    # Fixed KV tile for this program
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    mask_n = offs_n < S
    mask_d = offs_d < D_dim

    k_base = K_ptr + bid_b * stride_Kb + bid_h * stride_Kh
    v_base = V_ptr + bid_b * stride_Vb + bid_h * stride_Vh

    tile_k = tl.load(k_base + offs_n[:, None] * stride_Ks + offs_d[None, :] * stride_Kd,
                     mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
    tile_v = tl.load(v_base + offs_n[:, None] * stride_Vs + offs_d[None, :] * stride_Vd,
                     mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    acc_dk = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    acc_dv = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)

    num_q_tiles = tl.cdiv(S, BLOCK_M)

    for mid in range(num_q_tiles):
        offs_m = mid * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S
        causal_mask = (offs_m[:, None] >= offs_n[None, :])

        valid_mn = causal_mask & mask_m[:, None] & mask_n[None, :]

        q_base_t = Q_ptr + bid_b * stride_Qb + bid_h * stride_Qh
        tile_q = tl.load(q_base_t + offs_m[:, None] * stride_Qs + offs_d[None, :] * stride_Qd,
                         mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        dO_base_t = dO_ptr + bid_b * stride_dOb + bid_h * stride_dOh
        tile_dO = tl.load(dO_base_t + offs_m[:, None] * stride_dOs + offs_d[None, :] * stride_dOd,
                          mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        o_base_t = O_ptr + bid_b * stride_Ob + bid_h * stride_Oh
        tile_o = tl.load(o_base_t + offs_m[:, None] * stride_Os + offs_d[None, :] * stride_Od,
                         mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        row_dot = tl.sum(tile_dO * tile_o, axis=1)

        l_base_t = L_ptr + bid_b * stride_Lb + bid_h * stride_Lh
        tile_l = tl.load(l_base_t + offs_m * stride_Ls, mask=mask_m, other=0.0).to(tl.float32)

        s_tile = tl.dot(tile_q, tile_k.T)

        s_safe = tl.where(valid_mn, s_tile, -1e5)
        m_new = tl.max(s_safe, axis=1, keepdims=True)
        p = tl.exp(s_safe * scale - m_new)
        p = p * tl.exp(m_new - tile_l[:, None])
        p = tl.where(valid_mn, p, 0.0)

        dp_v = tl.dot(tile_dO, tile_v.T)
        ds = tl.where(valid_mn, (dp_v - row_dot[:, None]) * scale, 0.0)

        acc_dk += tl.dot(ds.T, tile_q)
        acc_dv += tl.dot(p.T, tile_dO)

    dk_base = dK_ptr + bid_b * stride_dKb + bid_h * stride_dKh
    dv_base = dV_ptr + bid_b * stride_dVb + bid_h * stride_dVh

    tl.store(dk_base + offs_n[:, None] * stride_dKs + offs_d[None, :] * stride_dKd,
             acc_dk.to(tl.bfloat16), mask=mask_n[:, None] & mask_d[None, :])
    tl.store(dv_base + offs_n[:, None] * stride_dVs + offs_d[None, :] * stride_dVd,
             acc_dv.to(tl.bfloat16), mask=mask_n[:, None] & mask_d[None, :])


@triton.jit
def _dQ_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, O_ptr, L_ptr, dQ_ptr,
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
    """Compute dQ.
    
    Each program owns (bid_b, bid_h, pid_m) -> one Q tile.
    Loops over ALL KV blocks, accumulating into dQ.
    """
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    bid_b = pid_bh // H
    bid_h = pid_bh % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    mask_m = offs_m < S
    mask_d = offs_d < D_dim

    q_base = Q_ptr + bid_b * stride_Qb + bid_h * stride_Qh
    tile_q = tl.load(q_base + offs_m[:, None] * stride_Qs + offs_d[None, :] * stride_Qd,
                     mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    dO_base = dO_ptr + bid_b * stride_dOb + bid_h * stride_dOh
    tile_dO = tl.load(dO_base + offs_m[:, None] * stride_dOs + offs_d[None, :] * stride_dOd,
                      mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    o_base = O_ptr + bid_b * stride_Ob + bid_h * stride_Oh
    tile_o = tl.load(o_base + offs_m[:, None] * stride_Os + offs_d[None, :] * stride_Od,
                     mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    row_dot = tl.sum(tile_dO * tile_o, axis=1)

    l_base = L_ptr + bid_b * stride_Lb + bid_h * stride_Lh
    tile_l = tl.load(l_base + offs_m * stride_Ls, mask=mask_m, other=0.0).to(tl.float32)

    acc_dq = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    num_kv_tiles = tl.cdiv(S, BLOCK_N)
    for nid in range(num_kv_tiles):
        offs_n = nid * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S
        causal_mask = (offs_m[:, None] >= offs_n[None, :])
        valid_mn = causal_mask & mask_m[:, None] & mask_n[None, :]

        k_base_t = K_ptr + bid_b * stride_Kb + bid_h * stride_Kh
        tile_k = tl.load(k_base_t + offs_n[:, None] * stride_Ks + offs_d[None, :] * stride_Kd,
                         mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        v_base_t = V_ptr + bid_b * stride_Vb + bid_h * stride_Vh
        tile_v = tl.load(v_base_t + offs_n[:, None] * stride_Vs + offs_d[None, :] * stride_Vd,
                         mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        s_tile = tl.dot(tile_q, tile_k.T)

        s_safe = tl.where(valid_mn, s_tile, -1e5)
        m_new = tl.max(s_safe, axis=1, keepdims=True)
        p = tl.exp(s_safe * scale - m_new)
        p = p * tl.exp(m_new - tile_l[:, None])
        p = tl.where(valid_mn, p, 0.0)

        dp_v = tl.dot(tile_dO, tile_v.T)
        ds = tl.where(valid_mn, (dp_v - row_dot[:, None]) * scale, 0.0)

        acc_dq += tl.dot(ds, tile_k)

    dq_base = dQ_ptr + bid_b * stride_dQb + bid_h * stride_dQh
    tl.store(dq_base + offs_m[:, None] * stride_dQs + offs_d[None, :] * stride_dQd,
             acc_dq.to(tl.bfloat16), mask=mask_m[:, None] & mask_d[None, :])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Causal multi-head attention backward pass.
    
    Uses two-kernel FlashAttention-style approach:
      Kernel 1 (dKV): grid over (kv_blocks, B*H). Each CTA loads one KV tile,
                       loops over ALL Q tiles, accumulates dK and dV for its KV block.
      Kernel 2 (dQ):  grid over (q_blocks, B*H). Each CTA loads one Q tile,
                      loops over ALL KV tiles, accumulates dQ for its Q block.
    No extra tensor allocations; D=rowsum(dO*O) computed inline per query tile.
    """
    torch.cuda.set_device(Q.device)

    B, H, S, D_dim = Q.shape
    attn_scale = 1.0 / math.sqrt(D_dim)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = 128

    sq = Q.stride()
    sk = K.stride()
    sv = V.stride()
    sdO = dO.stride()
    sO = O.stride()
    sdQ = dQ.stride()
    sdK = dK.stride()
    sdV = dV.stride()
    sL = L.stride()

    BH = B * H
    num_kv_blocks = triton.cdiv(S, BLOCK_N)
    num_q_blocks = triton.cdiv(S, BLOCK_M)

    # Initialize outputs to zero
    dK.zero_()
    dV.zero_()

    # Kernel 1: dK and dV
    grid_dKV = (num_kv_blocks, BH)
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
    grid_dQ = (num_q_blocks, BH)
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