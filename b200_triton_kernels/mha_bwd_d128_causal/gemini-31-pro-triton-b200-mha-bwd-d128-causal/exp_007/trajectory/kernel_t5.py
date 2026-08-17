import math
import torch
import triton
import triton.language as tl

# Configure Triton's tensor descriptor allocator for Blackwell TMA
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "PIPELINE_STAGES": 3}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "PIPELINE_STAGES": 3}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "PIPELINE_STAGES": 4}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "PIPELINE_STAGES": 4}, num_warps=8, num_stages=4),
    ],
    key=["S"],
)
@triton.jit
def bwd_dq_kernel(
    Q, K, V, O, dO, L, dQ,
    stride_qb, stride_qh, stride_qs,
    stride_kb, stride_kh, stride_ks,
    stride_vb, stride_vh, stride_vs,
    stride_ob, stride_oh, stride_os,
    stride_dob, stride_doh, stride_dos,
    stride_lb, stride_lh, stride_ls,
    stride_dqb, stride_dqh, stride_dqs,
    scale, S, H,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr, PIPELINE_STAGES: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    batch_idx = pid_bh // H
    head_idx = pid_bh % H
    
    q_start = pid_m * BLOCK_M
    
    q_base = Q + batch_idx * stride_qb + head_idx * stride_qh
    do_base = dO + batch_idx * stride_dob + head_idx * stride_doh
    o_base = O + batch_idx * stride_ob + head_idx * stride_oh
    k_base = K + batch_idx * stride_kb + head_idx * stride_kh
    v_base = V + batch_idx * stride_vb + head_idx * stride_vh
    dq_base = dQ + batch_idx * stride_dqb + head_idx * stride_dqh
    
    q_desc = tl.make_tensor_descriptor(q_base, shape=[S, BLOCK_D], strides=[stride_qs, 1], block_shape=[BLOCK_M, BLOCK_D], padding_option="zero")
    do_desc = tl.make_tensor_descriptor(do_base, shape=[S, BLOCK_D], strides=[stride_dos, 1], block_shape=[BLOCK_M, BLOCK_D], padding_option="zero")
    o_desc = tl.make_tensor_descriptor(o_base, shape=[S, BLOCK_D], strides=[stride_os, 1], block_shape=[BLOCK_M, BLOCK_D], padding_option="zero")
    k_desc = tl.make_tensor_descriptor(k_base, shape=[S, BLOCK_D], strides=[stride_ks, 1], block_shape=[BLOCK_N, BLOCK_D], padding_option="zero")
    v_desc = tl.make_tensor_descriptor(v_base, shape=[S, BLOCK_D], strides=[stride_vs, 1], block_shape=[BLOCK_N, BLOCK_D], padding_option="zero")
    
    q = q_desc.load([q_start, 0])
    do = do_desc.load([q_start, 0])
    o = o_desc.load([q_start, 0])
    
    offs_m = q_start + tl.arange(0, BLOCK_M)
    mask_m = offs_m < S
    lse_base = L + batch_idx * stride_lb + head_idx * stride_lh
    lse = tl.load(lse_base + offs_m * stride_ls, mask=mask_m, other=0.0)
    
    delta = tl.sum(do.to(tl.float32) * o.to(tl.float32), axis=1)
    dq = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    
    kv_dense_iters = q_start // BLOCK_N
    
    # Dense loop: Perfectly valid region, no causal masking needed
    for kv_idx in tl.range(0, kv_dense_iters, num_stages=PIPELINE_STAGES):
        kv_start = kv_idx * BLOCK_N
        k = k_desc.load([kv_start, 0])
        v = v_desc.load([kv_start, 0])
        
        scores = tl.dot(q, k.T) * scale
        p = tl.exp(scores - lse[:, None])
        
        dp = tl.dot(do, v.T)
        ds = p * (dp - delta[:, None]) * scale
        
        dq += tl.dot(ds.to(q.dtype), k)
        
    kv_max = tl.minimum(S, q_start + BLOCK_M)
    
    # Masked loop: Crosses the causal boundary
    for kv_idx in tl.range(kv_dense_iters, tl.cdiv(kv_max, BLOCK_N), num_stages=PIPELINE_STAGES):
        kv_start = kv_idx * BLOCK_N
        k = k_desc.load([kv_start, 0])
        v = v_desc.load([kv_start, 0])
        
        offs_n = kv_start + tl.arange(0, BLOCK_N)
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        
        scores = tl.dot(q, k.T) * scale
        scores = tl.where(causal_mask, scores, float('-inf'))
        p = tl.exp(scores - lse[:, None])
        
        dp = tl.dot(do, v.T)
        ds = p * (dp - delta[:, None]) * scale
        
        dq += tl.dot(ds.to(q.dtype), k)
        
    dq_desc = tl.make_tensor_descriptor(dq_base, shape=[S, BLOCK_D], strides=[stride_dqs, 1], block_shape=[BLOCK_M, BLOCK_D], padding_option="zero")
    dq_desc.store([q_start, 0], dq.to(q.dtype))

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "PIPELINE_STAGES": 3}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "PIPELINE_STAGES": 3}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "PIPELINE_STAGES": 4}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "PIPELINE_STAGES": 4}, num_warps=8, num_stages=4),
    ],
    key=["S"],
)
@triton.jit
def bwd_dkdv_kernel(
    Q, K, V, O, dO, L, dK, dV,
    stride_qb, stride_qh, stride_qs,
    stride_kb, stride_kh, stride_ks,
    stride_vb, stride_vh, stride_vs,
    stride_ob, stride_oh, stride_os,
    stride_dob, stride_doh, stride_dos,
    stride_lb, stride_lh, stride_ls,
    stride_dkb, stride_dkh, stride_dks,
    stride_dvb, stride_dvh, stride_dvs,
    scale, S, H,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr, PIPELINE_STAGES: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    batch_idx = pid_bh // H
    head_idx = pid_bh % H
    
    kv_start = pid_n * BLOCK_N
    
    q_base = Q + batch_idx * stride_qb + head_idx * stride_qh
    do_base = dO + batch_idx * stride_dob + head_idx * stride_doh
    o_base = O + batch_idx * stride_ob + head_idx * stride_oh
    k_base = K + batch_idx * stride_kb + head_idx * stride_kh
    v_base = V + batch_idx * stride_vb + head_idx * stride_vh
    dk_base = dK + batch_idx * stride_dkb + head_idx * stride_dkh
    dv_base = dV + batch_idx * stride_dvb + head_idx * stride_dvh
    
    k_desc = tl.make_tensor_descriptor(k_base, shape=[S, BLOCK_D], strides=[stride_ks, 1], block_shape=[BLOCK_N, BLOCK_D], padding_option="zero")
    v_desc = tl.make_tensor_descriptor(v_base, shape=[S, BLOCK_D], strides=[stride_vs, 1], block_shape=[BLOCK_N, BLOCK_D], padding_option="zero")
    q_desc = tl.make_tensor_descriptor(q_base, shape=[S, BLOCK_D], strides=[stride_qs, 1], block_shape=[BLOCK_M, BLOCK_D], padding_option="zero")
    do_desc = tl.make_tensor_descriptor(do_base, shape=[S, BLOCK_D], strides=[stride_dos, 1], block_shape=[BLOCK_M, BLOCK_D], padding_option="zero")
    o_desc = tl.make_tensor_descriptor(o_base, shape=[S, BLOCK_D], strides=[stride_os, 1], block_shape=[BLOCK_M, BLOCK_D], padding_option="zero")
    
    k = k_desc.load([kv_start, 0])
    v = v_desc.load([kv_start, 0])
    
    dk = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)
    dv = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)
    
    offs_n = kv_start + tl.arange(0, BLOCK_N)
    
    q_start_idx = kv_start // BLOCK_M
    q_dense_start_idx = (kv_start + BLOCK_N + BLOCK_M - 1) // BLOCK_M
    num_q_tiles = tl.cdiv(S, BLOCK_M)
    q_dense_start_idx = tl.minimum(q_dense_start_idx, num_q_tiles)
    
    lse_base = L + batch_idx * stride_lb + head_idx * stride_lh
    
    # Masked loop: Crosses the causal boundary
    for q_idx in tl.range(q_start_idx, q_dense_start_idx, num_stages=PIPELINE_STAGES):
        q_start = q_idx * BLOCK_M
        
        q = q_desc.load([q_start, 0])
        do = do_desc.load([q_start, 0])
        o = o_desc.load([q_start, 0])
        
        offs_m = q_start + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S
        lse = tl.load(lse_base + offs_m * stride_ls, mask=mask_m, other=0.0)
        
        delta = tl.sum(do.to(tl.float32) * o.to(tl.float32), axis=1)
        
        scores_t = tl.dot(k, q.T) * scale
        
        causal_mask_t = offs_n[:, None] <= offs_m[None, :]
        scores_t = tl.where(causal_mask_t, scores_t, float('-inf'))
        p_t = tl.exp(scores_t - lse[None, :])
        
        dv += tl.dot(p_t.to(q.dtype), do)
        dp_t = tl.dot(v, do.T)
        ds_t = p_t * (dp_t - delta[None, :]) * scale
        dk += tl.dot(ds_t.to(q.dtype), q)
        
    # Dense loop: Perfectly valid region, no causal masking needed
    for q_idx in tl.range(q_dense_start_idx, num_q_tiles, num_stages=PIPELINE_STAGES):
        q_start = q_idx * BLOCK_M
        
        q = q_desc.load([q_start, 0])
        do = do_desc.load([q_start, 0])
        o = o_desc.load([q_start, 0])
        
        offs_m = q_start + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S
        lse = tl.load(lse_base + offs_m * stride_ls, mask=mask_m, other=0.0)
        
        delta = tl.sum(do.to(tl.float32) * o.to(tl.float32), axis=1)
        
        scores_t = tl.dot(k, q.T) * scale
        p_t = tl.exp(scores_t - lse[None, :])
        
        dv += tl.dot(p_t.to(q.dtype), do)
        dp_t = tl.dot(v, do.T)
        ds_t = p_t * (dp_t - delta[None, :]) * scale
        dk += tl.dot(ds_t.to(q.dtype), q)
        
    dk_desc = tl.make_tensor_descriptor(dk_base, shape=[S, BLOCK_D], strides=[stride_dks, 1], block_shape=[BLOCK_N, BLOCK_D], padding_option="zero")
    dv_desc = tl.make_tensor_descriptor(dv_base, shape=[S, BLOCK_D], strides=[stride_dvs, 1], block_shape=[BLOCK_N, BLOCK_D], padding_option="zero")
    
    dk_desc.store([kv_start, 0], dk.to(k.dtype))
    dv_desc.store([kv_start, 0], dv.to(v.dtype))

def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes the causal multi-head attention backward pass.
    Writes the gradients to the preallocated dQ, dK, and dV tensors.
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    
    # Launch dQ owner kernel (iterates over K/V chunks).
    grid_dq = lambda META: (
        triton.cdiv(S, META['BLOCK_M']),
        B * H
    )
    bwd_dq_kernel[grid_dq](
        Q, K, V, O, dO, L, dQ,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        dO.stride(0), dO.stride(1), dO.stride(2),
        L.stride(0), L.stride(1), L.stride(2),
        dQ.stride(0), dQ.stride(1), dQ.stride(2),
        scale, S, H,
        BLOCK_D=d,
    )
    
    # Launch dK/dV owner kernel (iterates over Q chunks).
    grid_dkdv = lambda META: (
        triton.cdiv(S, META['BLOCK_N']),
        B * H
    )
    bwd_dkdv_kernel[grid_dkdv](
        Q, K, V, O, dO, L, dK, dV,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        dO.stride(0), dO.stride(1), dO.stride(2),
        L.stride(0), L.stride(1), L.stride(2),
        dK.stride(0), dK.stride(1), dK.stride(2),
        dV.stride(0), dV.stride(1), dV.stride(2),
        scale, S, H,
        BLOCK_D=d,
    )