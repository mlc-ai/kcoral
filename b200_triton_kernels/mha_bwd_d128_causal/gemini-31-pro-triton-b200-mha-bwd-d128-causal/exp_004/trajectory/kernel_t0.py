import math
import torch
import triton
import triton.language as tl

@triton.jit
def mha_bwd_dq_kernel(
    Q, K, V, O, dO, L, dQ,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    H, S, scale, LOG2_E,
    d: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    q_tile = tl.program_id(0)
    batch_head_id = tl.program_id(1)
    
    batch_id = batch_head_id // H
    head_id = batch_head_id % H
    
    # Advance pointers to the batch and head
    Q += batch_id * stride_qb + head_id * stride_qh
    K += batch_id * stride_kb + head_id * stride_kh
    V += batch_id * stride_vb + head_id * stride_vh
    O += batch_id * stride_ob + head_id * stride_oh
    dO += batch_id * stride_dob + head_id * stride_doh
    L += batch_id * stride_lb + head_id * stride_lh
    dQ += batch_id * stride_dqb + head_id * stride_dqh
    
    offs_m = q_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, d)
    
    mask_m = offs_m < S
    
    q_ptrs = Q + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    o_ptrs = O + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    do_ptrs = dO + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod
    
    # Load resident query tile, corresponding outputs, and upstream gradients
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
    do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
    lse = tl.load(L + offs_m * stride_ls, mask=mask_m, other=0.0)
    
    # Precompute rowwise delta needed for backward scaling
    delta = tl.sum(do.to(tl.float32) * o.to(tl.float32), axis=1)
    dq = tl.zeros((BLOCK_M, d), tl.float32)
    
    # Causal loop bound
    max_kv_tile = tl.minimum(tl.cdiv(S, BLOCK_N), (q_tile * BLOCK_M + BLOCK_M + BLOCK_N - 1) // BLOCK_N)
    
    for kv_tile in range(0, max_kv_tile):
        offs_n = kv_tile * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S
        
        k_ptrs = K + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        v_ptrs = V + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
        
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        
        # Recompute scores
        scores = tl.dot(q, k.T) * scale
        valid_score = (offs_m[:, None] >= offs_n[None, :]) & mask_m[:, None] & mask_n[None, :]
        scores = tl.where(valid_score, scores, float("-inf"))
        
        # Reconstruct probability matrix, observing scale
        p = tl.math.exp2((scores - lse[:, None]) * LOG2_E)
        dp = tl.dot(do, v.T)
        
        # Form dS gradient
        ds = p * (dp - delta[:, None]) * scale
        ds = tl.where(valid_score, ds, 0.0)
        
        # Accumulate strictly owned query gradient
        dq += tl.dot(ds.to(tl.bfloat16), k)
        
    dq_ptrs = dQ + offs_m[:, None] * stride_dqs + offs_d[None, :] * stride_dqd
    tl.store(dq_ptrs, dq.to(tl.bfloat16), mask=mask_m[:, None])


@triton.jit
def mha_bwd_dkdv_kernel(
    Q, K, V, O, dO, L, dK, dV,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    H, S, scale, LOG2_E,
    d: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    kv_tile = tl.program_id(0)
    batch_head_id = tl.program_id(1)
    
    batch_id = batch_head_id // H
    head_id = batch_head_id % H
    
    # Advance pointers to the batch and head
    Q += batch_id * stride_qb + head_id * stride_qh
    K += batch_id * stride_kb + head_id * stride_kh
    V += batch_id * stride_vb + head_id * stride_vh
    O += batch_id * stride_ob + head_id * stride_oh
    dO += batch_id * stride_dob + head_id * stride_doh
    L += batch_id * stride_lb + head_id * stride_lh
    dK += batch_id * stride_dkb + head_id * stride_dkh
    dV += batch_id * stride_dvb + head_id * stride_dvh
    
    offs_n = kv_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, d)
    
    mask_n = offs_n < S
    
    k_ptrs = K + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = V + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
    
    # Load resident KV tile
    k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
    
    dk = tl.zeros((BLOCK_N, d), tl.float32)
    dv = tl.zeros((BLOCK_N, d), tl.float32)
    
    # Causal loop bound tracking forward sequence progression
    min_q_tile = (kv_tile * BLOCK_N) // BLOCK_M
    max_q_tile = tl.cdiv(S, BLOCK_M)
    
    for q_tile in range(min_q_tile, max_q_tile):
        offs_m = q_tile * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S
        
        q_ptrs = Q + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
        o_ptrs = O + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
        do_ptrs = dO + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod
        
        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
        o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
        do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
        lse = tl.load(L + offs_m * stride_ls, mask=mask_m, other=0.0)
        
        # Calculate dynamic inline rowwise delta mapping to query space
        delta = tl.sum(do.to(tl.float32) * o.to(tl.float32), axis=1)
        
        # Compute scores_t natively transposed
        scores_t = tl.dot(k, q.T) * scale
        valid_score_t = (offs_m[None, :] >= offs_n[:, None]) & mask_m[None, :] & mask_n[:, None]
        scores_t = tl.where(valid_score_t, scores_t, float("-inf"))
        
        # Reconstruct structurally transposed probabilities
        p_t = tl.math.exp2((scores_t - lse[None, :]) * LOG2_E)
        
        # Accumulate strictly owned dV
        dv += tl.dot(p_t.to(tl.bfloat16), do)
        
        dp_t = tl.dot(v, do.T)
        ds_t = p_t * (dp_t - delta[None, :]) * scale
        ds_t = tl.where(valid_score_t, ds_t, 0.0)
        
        # Accumulate strictly owned dK
        dk += tl.dot(ds_t.to(tl.bfloat16), q)
        
    dk_ptrs = dK + offs_n[:, None] * stride_dks + offs_d[None, :] * stride_dkd
    dv_ptrs = dV + offs_n[:, None] * stride_dvs + offs_d[None, :] * stride_dvd
    
    tl.store(dk_ptrs, dk.to(tl.bfloat16), mask=mask_n[:, None])
    tl.store(dv_ptrs, dv.to(tl.bfloat16), mask=mask_n[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    
    BLOCK_M = 64
    BLOCK_N = 64
    
    num_q_tiles = triton.cdiv(S, BLOCK_M)
    num_kv_tiles = triton.cdiv(S, BLOCK_N)
    
    scale = 1.0 / math.sqrt(d)
    LOG2_E = 1.4426950408889634
    
    if num_q_tiles > 0:
        grid_dq = (num_q_tiles, B * H)
        mha_bwd_dq_kernel[grid_dq](
            Q, K, V, O, dO, L, dQ,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
            L.stride(0), L.stride(1), L.stride(2),
            dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
            H, S, scale, LOG2_E,
            d=d, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            num_warps=4, num_stages=3
        )
        
        grid_dkdv = (num_kv_tiles, B * H)
        mha_bwd_dkdv_kernel[grid_dkdv](
            Q, K, V, O, dO, L, dK, dV,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
            L.stride(0), L.stride(1), L.stride(2),
            dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
            dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
            H, S, scale, LOG2_E,
            d=d, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            num_warps=4, num_stages=3
        )