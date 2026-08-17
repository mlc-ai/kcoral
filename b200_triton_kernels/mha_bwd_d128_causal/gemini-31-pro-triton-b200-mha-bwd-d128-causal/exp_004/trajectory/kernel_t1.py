import math
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def mha_bwd_dq_kernel(
    q_desc, k_desc, v_desc, o_desc, do_desc, dq_desc,
    L_ptr, stride_lb, stride_lh, stride_ls,
    S, scale, LOG2_E, H,
    d: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    q_tile = tl.program_id(0)
    batch_head_id = tl.program_id(1)
    
    batch_id = batch_head_id // H
    head_id = batch_head_id % H
    
    # TMA logically loads bounded padded regions and lowers 4D shapes seamlessly
    q = q_desc.load([batch_id, head_id, q_tile * BLOCK_M, 0])
    q = tl.reshape(q, (BLOCK_M, d))
    
    o = o_desc.load([batch_id, head_id, q_tile * BLOCK_M, 0])
    o = tl.reshape(o, (BLOCK_M, d))
    
    do = do_desc.load([batch_id, head_id, q_tile * BLOCK_M, 0])
    do = tl.reshape(do, (BLOCK_M, d))
    
    offs_m = q_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < S
    
    # L has stride shapes that usually violate descriptor 16-byte alignment invariants.
    # Therefore, strictly rely on conventional pointer loading.
    lse_ptrs = L_ptr + batch_id * stride_lb + head_id * stride_lh + offs_m * stride_ls
    lse = tl.load(lse_ptrs, mask=mask_m, other=0.0)
    
    # Precompute rowwise delta logically needed for backward scaling
    delta = tl.sum(do.to(tl.float32) * o.to(tl.float32), axis=1)
    dq = tl.zeros((BLOCK_M, d), tl.float32)
    
    # Causal loop boundary definition avoiding non-contributing forward regions
    max_kv_tile = tl.minimum(tl.cdiv(S, BLOCK_N), (q_tile * BLOCK_M + BLOCK_M + BLOCK_N - 1) // BLOCK_N)
    
    for kv_tile in tl.range(0, max_kv_tile, num_stages=3):
        k = k_desc.load([batch_id, head_id, kv_tile * BLOCK_N, 0])
        k = tl.reshape(k, (BLOCK_N, d))
        
        v = v_desc.load([batch_id, head_id, kv_tile * BLOCK_N, 0])
        v = tl.reshape(v, (BLOCK_N, d))
        
        offs_n = kv_tile * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S
        
        # Recompute scores logically
        scores = tl.dot(q, k.T) * scale
        
        # Strictly mask out noncausal and out-of-bounds trailing sequences
        valid_score = (offs_m[:, None] >= offs_n[None, :]) & mask_m[:, None] & mask_n[None, :]
        scores = tl.where(valid_score, scores, float("-inf"))
        
        # Reconstruct probability matrix maintaining precision numerically 
        p = tl.math.exp2((scores - lse[:, None]) * LOG2_E)
        dp = tl.dot(do, v.T)
        
        # Form dS upstream local gradient
        ds = p * (dp - delta[:, None]) * scale
        ds = tl.where(valid_score, ds, 0.0)
        
        # Accumulate strictly owned query local gradient
        dq += tl.dot(ds.to(tl.bfloat16), k)
        
    dq = tl.reshape(dq.to(tl.bfloat16), (1, 1, BLOCK_M, d))
    dq_desc.store([batch_id, head_id, q_tile * BLOCK_M, 0], dq)


@triton.jit
def mha_bwd_dkdv_kernel(
    q_desc, k_desc, v_desc, o_desc, do_desc, dk_desc, dv_desc,
    L_ptr, stride_lb, stride_lh, stride_ls,
    S, scale, LOG2_E, H,
    d: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr
):
    kv_tile = tl.program_id(0)
    batch_head_id = tl.program_id(1)
    
    batch_id = batch_head_id // H
    head_id = batch_head_id % H
    
    k = k_desc.load([batch_id, head_id, kv_tile * BLOCK_N, 0])
    k = tl.reshape(k, (BLOCK_N, d))
    
    v = v_desc.load([batch_id, head_id, kv_tile * BLOCK_N, 0])
    v = tl.reshape(v, (BLOCK_N, d))
    
    dk = tl.zeros((BLOCK_N, d), tl.float32)
    dv = tl.zeros((BLOCK_N, d), tl.float32)
    
    offs_n = kv_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < S
    
    # Causal sequence start offset tracking forward sequence progression implicitly
    min_q_tile = (kv_tile * BLOCK_N) // BLOCK_M
    max_q_tile = tl.cdiv(S, BLOCK_M)
    
    for q_tile in tl.range(min_q_tile, max_q_tile, num_stages=2):
        q = q_desc.load([batch_id, head_id, q_tile * BLOCK_M, 0])
        q = tl.reshape(q, (BLOCK_M, d))
        
        o = o_desc.load([batch_id, head_id, q_tile * BLOCK_M, 0])
        o = tl.reshape(o, (BLOCK_M, d))
        
        do = do_desc.load([batch_id, head_id, q_tile * BLOCK_M, 0])
        do = tl.reshape(do, (BLOCK_M, d))
        
        offs_m = q_tile * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S
        
        lse_ptrs = L_ptr + batch_id * stride_lb + head_id * stride_lh + offs_m * stride_ls
        lse = tl.load(lse_ptrs, mask=mask_m, other=0.0)
        
        # Inline dynamic rowwise delta compilation mapping 
        delta = tl.sum(do.to(tl.float32) * o.to(tl.float32), axis=1)
        
        # Compute scores natively transposed avoiding excessive transpose overhead
        scores_t = tl.dot(k, q.T) * scale
        
        # Invert conditional mask logic for strictly structural orientation
        valid_score_t = (offs_m[None, :] >= offs_n[:, None]) & mask_m[None, :] & mask_n[:, None]
        scores_t = tl.where(valid_score_t, scores_t, float("-inf"))
        
        p_t = tl.math.exp2((scores_t - lse[None, :]) * LOG2_E)
        
        # Accumulate strictly owned partial local gradient dV explicitly mapped
        dv += tl.dot(p_t.to(tl.bfloat16), do)
        
        dp_t = tl.dot(v, do.T)
        ds_t = p_t * (dp_t - delta[None, :]) * scale
        ds_t = tl.where(valid_score_t, ds_t, 0.0)
        
        # Accumulate strictly owned partial local gradient dK explicitly mapped
        dk += tl.dot(ds_t.to(tl.bfloat16), q)
        
    dk = tl.reshape(dk.to(tl.bfloat16), (1, 1, BLOCK_N, d))
    dk_desc.store([batch_id, head_id, kv_tile * BLOCK_N, 0], dk)
    
    dv = tl.reshape(dv.to(tl.bfloat16), (1, 1, BLOCK_N, d))
    dv_desc.store([batch_id, head_id, kv_tile * BLOCK_N, 0], dv)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    
    # Strictly calculated and statically verified blocking profiles
    BLOCK_M_DQ = 128
    BLOCK_N_DQ = 64
    
    BLOCK_N_DKDV = 64
    BLOCK_M_DKDV = 128
    
    num_q_tiles = triton.cdiv(S, BLOCK_M_DQ)
    num_kv_tiles = triton.cdiv(S, BLOCK_N_DKDV)
    
    scale = 1.0 / math.sqrt(d)
    LOG2_E = 1.4426950408889634
    
    # TMA descriptors effectively represent native memory topologies synchronously 
    q_desc_dq = TensorDescriptor.from_tensor(Q, [1, 1, BLOCK_M_DQ, d])
    k_desc_dq = TensorDescriptor.from_tensor(K, [1, 1, BLOCK_N_DQ, d])
    v_desc_dq = TensorDescriptor.from_tensor(V, [1, 1, BLOCK_N_DQ, d])
    o_desc_dq = TensorDescriptor.from_tensor(O, [1, 1, BLOCK_M_DQ, d])
    do_desc_dq = TensorDescriptor.from_tensor(dO, [1, 1, BLOCK_M_DQ, d])
    dq_desc = TensorDescriptor.from_tensor(dQ, [1, 1, BLOCK_M_DQ, d])
    
    q_desc_dkdv = TensorDescriptor.from_tensor(Q, [1, 1, BLOCK_M_DKDV, d])
    k_desc_dkdv = TensorDescriptor.from_tensor(K, [1, 1, BLOCK_N_DKDV, d])
    v_desc_dkdv = TensorDescriptor.from_tensor(V, [1, 1, BLOCK_N_DKDV, d])
    o_desc_dkdv = TensorDescriptor.from_tensor(O, [1, 1, BLOCK_M_DKDV, d])
    do_desc_dkdv = TensorDescriptor.from_tensor(dO, [1, 1, BLOCK_M_DKDV, d])
    dk_desc = TensorDescriptor.from_tensor(dK, [1, 1, BLOCK_N_DKDV, d])
    dv_desc = TensorDescriptor.from_tensor(dV, [1, 1, BLOCK_N_DKDV, d])
    
    if num_q_tiles > 0:
        grid_dq = (num_q_tiles, B * H)
        mha_bwd_dq_kernel[grid_dq](
            q_desc_dq, k_desc_dq, v_desc_dq, o_desc_dq, do_desc_dq, dq_desc,
            L, L.stride(0), L.stride(1), L.stride(2),
            S, scale, LOG2_E, H,
            d=d, BLOCK_M=BLOCK_M_DQ, BLOCK_N=BLOCK_N_DQ,
            num_warps=8, num_stages=3
        )
        
        grid_dkdv = (num_kv_tiles, B * H)
        mha_bwd_dkdv_kernel[grid_dkdv](
            q_desc_dkdv, k_desc_dkdv, v_desc_dkdv, o_desc_dkdv, do_desc_dkdv, dk_desc, dv_desc,
            L, L.stride(0), L.stride(1), L.stride(2),
            S, scale, LOG2_E, H,
            d=d, BLOCK_N=BLOCK_N_DKDV, BLOCK_M=BLOCK_M_DKDV,
            num_warps=8, num_stages=2
        )
        
    return dQ, dK, dV