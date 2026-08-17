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
    """Compute dK and dV for causal attention backward.
    
    Grid axis 0: (B*H) -> batch/head index
    Grid axis 1: kv-block index
    
    Each CTA owns one (batch, head, kv-block). It loops over ALL q-blocks,
    recomputing softmax to get P, then accumulates dK and dV.
    """
    pid_bh = tl.program_id(0)
    pid_n = tl.program_id(1)

    bid_b = pid_bh // H
    bid_h = pid_bh % H

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    n_mask = offs_n < S
    d_mask = offs_d < D_dim
    nd_mask = n_mask[:, None] & d_mask[None, :]

    # Load fixed KV tile
    k_off = K_ptr + bid_b * stride_Kb + bid_h * stride_Kh + \
            offs_n[:, None] * stride_Ks + offs_d[None, :] * stride_Kd
    v_off = V_ptr + bid_b * stride_Vb + bid_h * stride_Vh + \
            offs_n[:, None] * stride_Vs + offs_d[None, :] * stride_Vd
    
    tile_k = tl.load(k_off, mask=nd_mask, other=0.0).to(tl.float32)
    tile_v = tl.load(v_off, mask=nd_mask, other=0.0).to(tl.float32)

    acc_dk = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    acc_dv = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)

    nqblocks = tl.cdiv(S, BLOCK_M)

    for qb in range(nqblocks):
        offs_m = qb * BLOCK_M + tl.arange(0, BLOCK_M)
        m_mask = offs_m < S
        md_mask = m_mask[:, None] & d_mask[None, :]
        
        causal = (offs_m[:, None] >= offs_n[None, :])
        mn_mask = causal & m_mask[:, None] & n_mask[None, :]

        # Load Q, dO, O tiles
        q_off = Q_ptr + bid_b * stride_Qb + bid_h * stride_Qh + \
                offs_m[:, None] * stride_Qs + offs_d[None, :] * stride_Qd
        do_off = dO_ptr + bid_b * stride_dOb + bid_h * stride_dOh + \
                 offs_m[:, None] * stride_dOs + offs_d[None, :] * stride_dOd
        o_off = O_ptr + bid_b * stride_Ob + bid_h * stride_Oh + \
                offs_m[:, None] * stride_Os + offs_d[None, :] * stride_Od
        
        tile_q = tl.load(q_off, mask=md_mask, other=0.0).to(tl.float32)
        tile_do = tl.load(do_off, mask=md_mask, other=0.0).to(tl.float32)
        tile_o = tl.load(o_off, mask=md_mask, other=0.0).to(tl.float32)

        # D[row] = sum(dO * O, dim=d)
        row_dot = tl.sum(tile_do * tile_o, axis=1)

        # Load logsumexp
        l_off = L_ptr + bid_b * stride_Lb + bid_h * stride_Lh + offs_m * stride_Ls
        tile_l = tl.load(l_off, mask=m_mask, other=1e38).to(tl.float32)

        # Compute attention scores
        s_mat = tl.dot(tile_q, tile_k.T)
        
        # For masked positions, use a large negative number so exp -> 0
        NEG_INF = -1e38
        s_m = tl.where(mn_mask, s_mat, NEG_INF)
        m_row = tl.max(s_m, axis=1, keepdims=True)
        
        p_exp = tl.exp(s_m * scale - m_row)
        p_exp = p_exp * tl.exp(m_row - tile_l[:, None])
        p_val = tl.where(mn_mask, p_exp, 0.0)

        # dS computation
        dpv = tl.dot(tile_do, tile_v.T)
        ds = tl.where(mn_mask, (dpv - row_dot[:, None]) * scale, 0.0)

        acc_dk += tl.dot(ds.T, tile_q)
        acc_dv += tl.dot(p_val.T, tile_do)

    # Store results
    dk_off = dK_ptr + bid_b * stride_dKb + bid_h * stride_dKh + \
             offs_n[:, None] * stride_dKs + offs_d[None, :] * stride_dKd
    dv_off = dV_ptr + bid_b * stride_dVb + bid_h * stride_dVh + \
             offs_n[:, None] * stride_dVs + offs_d[None, :] * stride_dVd

    tl.store(dk_off, acc_dk.to(tl.bfloat16), mask=nd_mask)
    tl.store(dv_off, acc_dv.to(tl.bfloat16), mask=nd_mask)


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
    """Compute dQ for causal attention backward."""
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    bid_b = pid_bh // H
    bid_h = pid_bh % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    m_mask = offs_m < S
    d_mask = offs_d < D_dim
    md_mask = m_mask[:, None] & d_mask[None, :]

    # Load Q, dO, O tiles (fixed for this program)
    q_base = Q_ptr + bid_b * stride_Qb + bid_h * stride_Qh
    do_base = dO_ptr + bid_b * stride_dOb + bid_h * stride_dOh
    o_base = O_ptr + bid_b * stride_Ob + bid_h * stride_Oh

    tile_q = tl.load(q_base + offs_m[:, None] * stride_Qs + offs_d[None, :] * stride_Qd,
                     mask=md_mask, other=0.0).to(tl.float32)
    tile_do = tl.load(do_base + offs_m[:, None] * stride_dOs + offs_d[None, :] * stride_dOd,
                      mask=md_mask, other=0.0).to(tl.float32)
    tile_o = tl.load(o_base + offs_m[:, None] * stride_Os + offs_d[None, :] * stride_Od,
                     mask=md_mask, other=0.0).to(tl.float32)

    row_dot = tl.sum(tile_do * tile_o, axis=1)

    l_base = L_ptr + bid_b * stride_Lb + bid_h * stride_Lh
    tile_l = tl.load(l_base + offs_m * stride_Ls, mask=m_mask, other=1e38).to(tl.float32)

    acc_dq = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    nkblocks = tl.cdiv(S, BLOCK_N)
    for kb in range(nkblocks):
        offs_n = kb * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < S
        nd_mask = n_mask[:, None] & d_mask[None, :]
        
        causal = (offs_m[:, None] >= offs_n[None, :])
        mn_mask = causal & m_mask[:, None] & n_mask[None, :]

        k_base_t = K_ptr + bid_b * stride_Kb + bid_h * stride_Kh
        v_base_t = V_ptr + bid_b * stride_Vb + bid_h * stride_Vh
        
        tile_k = tl.load(k_base_t + offs_n[:, None] * stride_Ks + offs_d[None, :] * stride_Kd,
                         mask=nd_mask, other=0.0).to(tl.float32)
        tile_v = tl.load(v_base_t + offs_n[:, None] * stride_Vs + offs_d[None, :] * stride_Vd,
                         mask=nd_mask, other=0.0).to(tl.float32)

        s_mat = tl.dot(tile_q, tile_k.T)
        
        NEG_INF = -1e38
        s_m = tl.where(mn_mask, s_mat, NEG_INF)
        m_row = tl.max(s_m, axis=1, keepdims=True)
        
        p_exp = tl.exp(s_m * scale - m_row)
        p_exp = p_exp * tl.exp(m_row - tile_l[:, None])
        p_val = tl.where(mn_mask, p_exp, 0.0)

        dpv = tl.dot(tile_do, tile_v.T)
        ds = tl.where(mn_mask, (dpv - row_dot[:, None]) * scale, 0.0)

        acc_dq += tl.dot(ds, tile_k)

    dq_off = dQ_ptr + bid_b * stride_dQb + bid_h * stride_dQh + \
             offs_m[:, None] * stride_dQs + offs_d[None, :] * stride_dQd
    tl.store(dq_off, acc_dq.to(tl.bfloat16), mask=md_mask)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Causal multi-head attention backward pass.
    
    Two-kernel strategy following FlashAttention:
      Kernel 1 (dKV): grid=(B*H, ceil(S/BLOCK_N)). Each CTA fixes one KV block,
         loops over all Q blocks, accumulates into dK/dV. Non-overlapping writes.
      Kernel 2 (dQ): grid=(ceil(S/BLOCK_M), B*H). Each CTA fixes one Q block,
         loops over all KV blocks, accumulates into dQ. Non-overlapping writes.
    
    Numerical approach: recover P = exp(S/sqrt(d) - L) inline using stored logsumexp L.
    No extra tensor allocations needed.
    """
    torch.cuda.set_device(Q.device)

    B, H, S, D_dim = Q.shape
    assert K.shape == V.shape == O.shape == dO.shape == dQ.shape == dK.shape == dV.shape
    assert L.shape == (B, H, S)

    scale = 1.0 / math.sqrt(D_dim)

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
    nkblocks = triton.cdiv(S, BLOCK_N)
    nqblocks = triton.cdiv(S, BLOCK_M)

    # Initialize outputs
    dK.zero_()
    dV.zero_()

    # Kernel 1: dK and dV
    grid_dkv = (BH, nkblocks)
    _dKV_kernel[grid_dkv](
        Q, K, V, dO, O, L, dK, dV,
        B, H, S, D_dim, scale,
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
    grid_dq = (nqblocks, BH)
    _dQ_kernel[grid_dq](
        Q, K, V, dO, O, L, dQ,
        B, H, S, D_dim, scale,
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