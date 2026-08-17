import torch
import triton
import triton.language as tl

def get_dq_configs():
    return [
        # Limit stages dynamically to remain beneath Blackwell's 228 KiB SM budget per CTA
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=2, num_warps=8),
    ]

def get_dkdv_configs():
    return [
        # O, dO, and Q are loaded per inner loop logic. Maintain smaller blocking constraints to stay resident
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
    ]

@triton.autotune(configs=get_dq_configs(), key=['seqlen'])
@triton.jit
def bwd_dq_kernel(
    Q, K, V, O, dO, dQ, L,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    stride_lb, stride_lh, stride_ls,
    seqlen, softmax_scale,
    H,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    q_start = pid_m * BLOCK_M
    if q_start >= seqlen:
        return
        
    pid_b = pid_bh // H
    pid_h = pid_bh % H
    
    offs_m = q_start + tl.arange(0, BLOCK_M)
    mask_m = offs_m < seqlen
    offs_d = tl.arange(0, HEAD_DIM)
    
    # Base pointers setup
    q_ptrs = Q + pid_b * stride_qb + pid_h * stride_qh + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    do_ptrs = dO + pid_b * stride_dob + pid_h * stride_doh + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod
    o_ptrs = O + pid_b * stride_ob + pid_h * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    l_ptrs = L + pid_b * stride_lb + pid_h * stride_lh + offs_m * stride_ls
    
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
    o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
    lse = tl.load(l_ptrs, mask=mask_m, other=0.0)
    
    # Outer precomputed delta property isolated once
    delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
    dq = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)
    
    q_end = tl.minimum(q_start + BLOCK_M, seqlen)
    num_kv_tiles = tl.cdiv(q_end, BLOCK_N)
    num_unmasked_kv = q_start // BLOCK_N
    
    offs_n = tl.arange(0, BLOCK_N)
    k_ptrs = K + pid_b * stride_kb + pid_h * stride_kh + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = V + pid_b * stride_vb + pid_h * stride_vh + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
    
    RCP_LN2 = 1.4426950408889634
    
    # 1. Fully-unmasked Loop Stream (Guaranteed within causal window)
    curr_n_base = 0
    for kv_tile in range(0, num_unmasked_kv):
        curr_n = curr_n_base + offs_n
        mask_n = curr_n < seqlen
        
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        
        scores = tl.dot(q, k.T) * softmax_scale
        
        # Bypass causal_mask ALU checks
        valid_mask = mask_n[None, :] & mask_m[:, None]
        scores = tl.where(valid_mask, scores, float('-inf'))
        p = tl.math.exp2((scores - lse[:, None]) * RCP_LN2)
        
        dp = tl.dot(do, v.T)
        ds = p * (dp - delta[:, None]) * softmax_scale
        ds = tl.where(valid_mask, ds, 0.0)
        
        dq += tl.dot(ds.to(tl.bfloat16), k)
        
        curr_n_base += BLOCK_N
        k_ptrs += BLOCK_N * stride_ks
        v_ptrs += BLOCK_N * stride_vs
        
    # 2. Causally-masked Overlapped Loop Stream 
    curr_n_base = num_unmasked_kv * BLOCK_N
    for kv_tile in range(num_unmasked_kv, num_kv_tiles):
        curr_n = curr_n_base + offs_n
        mask_n = curr_n < seqlen
        
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        
        scores = tl.dot(q, k.T) * softmax_scale
        
        causal_mask = offs_m[:, None] >= curr_n[None, :]
        valid_mask = causal_mask & mask_n[None, :] & mask_m[:, None]
        
        scores = tl.where(valid_mask, scores, float('-inf'))
        p = tl.math.exp2((scores - lse[:, None]) * RCP_LN2)
        
        dp = tl.dot(do, v.T)
        ds = p * (dp - delta[:, None]) * softmax_scale
        ds = tl.where(valid_mask, ds, 0.0)
        
        dq += tl.dot(ds.to(tl.bfloat16), k)
        
        curr_n_base += BLOCK_N
        k_ptrs += BLOCK_N * stride_ks
        v_ptrs += BLOCK_N * stride_vs
        
    dq_ptrs = dQ + pid_b * stride_dqb + pid_h * stride_dqh + offs_m[:, None] * stride_dqs + offs_d[None, :] * stride_dqd
    tl.store(dq_ptrs, dq.to(tl.bfloat16), mask=mask_m[:, None])

@triton.autotune(configs=get_dkdv_configs(), key=['seqlen'])
@triton.jit
def bwd_dk_dv_kernel(
    Q, K, V, O, dO, dK, dV, L,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    stride_lb, stride_lh, stride_ls,
    seqlen, softmax_scale,
    H,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    kv_start = pid_n * BLOCK_N
    if kv_start >= seqlen:
        return
        
    pid_b = pid_bh // H
    pid_h = pid_bh % H
    
    offs_n = kv_start + tl.arange(0, BLOCK_N)
    mask_n = offs_n < seqlen
    offs_d = tl.arange(0, HEAD_DIM)
    
    # Base pointers setup
    k_ptrs = K + pid_b * stride_kb + pid_h * stride_kh + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = V + pid_b * stride_vb + pid_h * stride_vh + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
    
    k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
    
    dk = tl.zeros((BLOCK_N, HEAD_DIM), tl.float32)
    dv = tl.zeros((BLOCK_N, HEAD_DIM), tl.float32)
    
    kv_end = tl.minimum(kv_start + BLOCK_N, seqlen)
    q_start_tile = kv_start // BLOCK_M
    num_q_tiles = tl.cdiv(seqlen, BLOCK_M)
    
    first_unmasked_q = tl.cdiv(kv_end, BLOCK_M)
    offs_m = tl.arange(0, BLOCK_M)
    
    q_ptrs = Q + pid_b * stride_qb + pid_h * stride_qh + (q_start_tile * BLOCK_M + offs_m)[:, None] * stride_qs + offs_d[None, :] * stride_qd
    do_ptrs = dO + pid_b * stride_dob + pid_h * stride_doh + (q_start_tile * BLOCK_M + offs_m)[:, None] * stride_dos + offs_d[None, :] * stride_dod
    o_ptrs = O + pid_b * stride_ob + pid_h * stride_oh + (q_start_tile * BLOCK_M + offs_m)[:, None] * stride_os + offs_d[None, :] * stride_od
    l_ptrs = L + pid_b * stride_lb + pid_h * stride_lh + (q_start_tile * BLOCK_M + offs_m) * stride_ls
    
    RCP_LN2 = 1.4426950408889634
    
    # 1. Causally-masked Overlapped Loop Stream 
    curr_m_base = q_start_tile * BLOCK_M
    for q_tile in range(q_start_tile, first_unmasked_q):
        curr_m = curr_m_base + offs_m
        mask_m = curr_m < seqlen
        
        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
        do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
        o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
        lse = tl.load(l_ptrs, mask=mask_m, other=0.0)
        
        delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
        scores_t = tl.dot(k, q.T) * softmax_scale
        
        causal_mask_t = curr_m[None, :] >= offs_n[:, None]
        valid_mask_t = causal_mask_t & mask_m[None, :] & mask_n[:, None]
        
        scores_t = tl.where(valid_mask_t, scores_t, float('-inf'))
        p_t = tl.math.exp2((scores_t - lse[None, :]) * RCP_LN2)
        
        dv += tl.dot(p_t.to(tl.bfloat16), do)
        
        dp_t = tl.dot(v, do.T)
        ds_t = p_t * (dp_t - delta[None, :]) * softmax_scale
        ds_t = tl.where(valid_mask_t, ds_t, 0.0)
        
        dk += tl.dot(ds_t.to(tl.bfloat16), q)
        
        curr_m_base += BLOCK_M
        q_ptrs += BLOCK_M * stride_qs
        do_ptrs += BLOCK_M * stride_dos
        o_ptrs += BLOCK_M * stride_os
        l_ptrs += BLOCK_M * stride_ls
        
    # 2. Fully-unmasked Loop Stream (Guaranteed within causal window)
    for q_tile in range(first_unmasked_q, num_q_tiles):
        curr_m = curr_m_base + offs_m
        mask_m = curr_m < seqlen
        
        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
        do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
        o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
        lse = tl.load(l_ptrs, mask=mask_m, other=0.0)
        
        delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
        scores_t = tl.dot(k, q.T) * softmax_scale
        
        # Bypass causal_mask ALU checks
        valid_mask_t = mask_m[None, :] & mask_n[:, None]
        scores_t = tl.where(valid_mask_t, scores_t, float('-inf'))
        p_t = tl.math.exp2((scores_t - lse[None, :]) * RCP_LN2)
        
        dv += tl.dot(p_t.to(tl.bfloat16), do)
        
        dp_t = tl.dot(v, do.T)
        ds_t = p_t * (dp_t - delta[None, :]) * softmax_scale
        ds_t = tl.where(valid_mask_t, ds_t, 0.0)
        
        dk += tl.dot(ds_t.to(tl.bfloat16), q)
        
        curr_m_base += BLOCK_M
        q_ptrs += BLOCK_M * stride_qs
        do_ptrs += BLOCK_M * stride_dos
        o_ptrs += BLOCK_M * stride_os
        l_ptrs += BLOCK_M * stride_ls
        
    dk_ptrs = dK + pid_b * stride_dkb + pid_h * stride_dkh + offs_n[:, None] * stride_dks + offs_d[None, :] * stride_dkd
    dv_ptrs = dV + pid_b * stride_dvb + pid_h * stride_dvh + offs_n[:, None] * stride_dvs + offs_d[None, :] * stride_dvd
    
    tl.store(dk_ptrs, dk.to(tl.bfloat16), mask=mask_n[:, None])
    tl.store(dv_ptrs, dv.to(tl.bfloat16), mask=mask_n[:, None])

def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    softmax_scale = 1.0 / (D ** 0.5)
    
    # 1. Independent Ownership Segment 1: Produce dQ mapping queries exclusively over corresponding KV items
    grid_dq = lambda META: (
        triton.cdiv(S, META['BLOCK_M']),
        B * H,
    )
    bwd_dq_kernel[grid_dq](
        Q, K, V, O, dO, dQ, L,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        S, softmax_scale,
        H, HEAD_DIM=D,
    )
    
    # 2. Independent Ownership Segment 2: Produce dK,dV grouping corresponding active Query sets exclusively mapped over Keys
    grid_dkdv = lambda META: (
        triton.cdiv(S, META['BLOCK_N']),
        B * H,
    )
    bwd_dk_dv_kernel[grid_dkdv](
        Q, K, V, O, dO, dK, dV, L,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        S, softmax_scale,
        H, HEAD_DIM=D,
    )