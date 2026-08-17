import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

# Configs restrict max `num_stages` sizes to guarantee that shared memory doesn't 
# spill past the 228 KiB limit per streaming multiprocessor (SM) on Blackwell GPUs.
def get_dq_configs():
    return [
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
    ]

def get_dkdv_configs():
    return [
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
    ]

def bwd_dq_pre_hook(kwargs):
    BLOCK_M = kwargs["BLOCK_M"]
    BLOCK_N = kwargs["BLOCK_N"]
    HEAD_DIM = kwargs["HEAD_DIM"]
    
    # Pre-build TMA Descriptors avoiding device-sided recreation overhead
    kwargs["q_desc"] = TensorDescriptor.from_tensor(kwargs["Q"], [1, 1, BLOCK_M, HEAD_DIM])
    kwargs["k_desc"] = TensorDescriptor.from_tensor(kwargs["K"], [1, 1, BLOCK_N, HEAD_DIM])
    kwargs["v_desc"] = TensorDescriptor.from_tensor(kwargs["V"], [1, 1, BLOCK_N, HEAD_DIM])
    kwargs["do_desc"] = TensorDescriptor.from_tensor(kwargs["dO"], [1, 1, BLOCK_M, HEAD_DIM])
    kwargs["dq_desc"] = TensorDescriptor.from_tensor(kwargs["dQ"], [1, 1, BLOCK_M, HEAD_DIM])
    kwargs["o_desc"] = TensorDescriptor.from_tensor(kwargs["O"], [1, 1, BLOCK_M, HEAD_DIM])

def bwd_dkdv_pre_hook(kwargs):
    BLOCK_M = kwargs["BLOCK_M"]
    BLOCK_N = kwargs["BLOCK_N"]
    HEAD_DIM = kwargs["HEAD_DIM"]
    
    kwargs["q_desc"] = TensorDescriptor.from_tensor(kwargs["Q"], [1, 1, BLOCK_M, HEAD_DIM])
    kwargs["k_desc"] = TensorDescriptor.from_tensor(kwargs["K"], [1, 1, BLOCK_N, HEAD_DIM])
    kwargs["v_desc"] = TensorDescriptor.from_tensor(kwargs["V"], [1, 1, BLOCK_N, HEAD_DIM])
    kwargs["do_desc"] = TensorDescriptor.from_tensor(kwargs["dO"], [1, 1, BLOCK_M, HEAD_DIM])
    kwargs["dk_desc"] = TensorDescriptor.from_tensor(kwargs["dK"], [1, 1, BLOCK_N, HEAD_DIM])
    kwargs["dv_desc"] = TensorDescriptor.from_tensor(kwargs["dV"], [1, 1, BLOCK_N, HEAD_DIM])
    kwargs["o_desc"] = TensorDescriptor.from_tensor(kwargs["O"], [1, 1, BLOCK_M, HEAD_DIM])

@triton.autotune(configs=get_dq_configs(), key=['seqlen'], pre_hook=bwd_dq_pre_hook)
@triton.jit
def bwd_dq_kernel(
    q_desc, k_desc, v_desc, do_desc, dq_desc, o_desc,
    Q, K, V, dO, dQ, O, L,
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
    
    # 4D TMA block loads safely abstract away standard memory addressing bounds padding
    q_4d = q_desc.load([pid_b, pid_h, q_start, 0])
    q = tl.reshape(q_4d, (BLOCK_M, HEAD_DIM))
    
    do_4d = do_desc.load([pid_b, pid_h, q_start, 0])
    do = tl.reshape(do_4d, (BLOCK_M, HEAD_DIM))
    
    o_4d = o_desc.load([pid_b, pid_h, q_start, 0])
    o = tl.reshape(o_4d, (BLOCK_M, HEAD_DIM))
    
    offs_m = q_start + tl.arange(0, BLOCK_M)
    mask_m = offs_m < seqlen
    offs_m_clamped = tl.minimum(offs_m, seqlen - 1)
    
    # Address bounding explicitly clamped to valid indices strictly passing tests
    l_ptrs = L + pid_b * stride_lb + pid_h * stride_lh + offs_m_clamped * stride_ls
    lse = tl.load(l_ptrs, mask=mask_m, other=0.0)
    
    delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
    dq = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)
    
    q_end = tl.minimum(q_start + BLOCK_M, seqlen)
    num_kv_tiles = tl.cdiv(q_end, BLOCK_N)
    
    num_unmasked_kv = q_start // BLOCK_N
    
    offs_n = tl.arange(0, BLOCK_N)
    RCP_LN2 = 1.4426950408889634
    
    # 1. Fully-unmasked Loop Stream
    for kv_tile in range(0, num_unmasked_kv):
        k_4d = k_desc.load([pid_b, pid_h, kv_tile * BLOCK_N, 0])
        k = tl.reshape(k_4d, (BLOCK_N, HEAD_DIM))
        
        v_4d = v_desc.load([pid_b, pid_h, kv_tile * BLOCK_N, 0])
        v = tl.reshape(v_4d, (BLOCK_N, HEAD_DIM))
        
        scores = tl.dot(q, k.T, out_dtype=tl.float32) * softmax_scale
        
        curr_n = kv_tile * BLOCK_N + offs_n
        mask_n = curr_n < seqlen
        valid_mask = mask_n[None, :] & mask_m[:, None]
        
        scores = tl.where(valid_mask, scores, float('-inf'))
        p = tl.math.exp2((scores - lse[:, None]) * RCP_LN2)
        
        dp = tl.dot(do, v.T, out_dtype=tl.float32)
        ds = p * (dp - delta[:, None]) * softmax_scale
        ds = tl.where(valid_mask, ds, 0.0)
        
        dq += tl.dot(ds.to(tl.bfloat16), k, out_dtype=tl.float32)
        
    # 2. Causally-masked Overlapped Loop Stream
    for kv_tile in range(num_unmasked_kv, num_kv_tiles):
        k_4d = k_desc.load([pid_b, pid_h, kv_tile * BLOCK_N, 0])
        k = tl.reshape(k_4d, (BLOCK_N, HEAD_DIM))
        
        v_4d = v_desc.load([pid_b, pid_h, kv_tile * BLOCK_N, 0])
        v = tl.reshape(v_4d, (BLOCK_N, HEAD_DIM))
        
        scores = tl.dot(q, k.T, out_dtype=tl.float32) * softmax_scale
        
        curr_n = kv_tile * BLOCK_N + offs_n
        mask_n = curr_n < seqlen
        causal_mask = offs_m[:, None] >= curr_n[None, :]
        valid_mask = causal_mask & mask_n[None, :] & mask_m[:, None]
        
        scores = tl.where(valid_mask, scores, float('-inf'))
        p = tl.math.exp2((scores - lse[:, None]) * RCP_LN2)
        
        dp = tl.dot(do, v.T, out_dtype=tl.float32)
        ds = p * (dp - delta[:, None]) * softmax_scale
        ds = tl.where(valid_mask, ds, 0.0)
        
        dq += tl.dot(ds.to(tl.bfloat16), k, out_dtype=tl.float32)
        
    dq_4d = tl.reshape(dq.to(tl.bfloat16), (1, 1, BLOCK_M, HEAD_DIM))
    dq_desc.store([pid_b, pid_h, q_start, 0], dq_4d)

@triton.autotune(configs=get_dkdv_configs(), key=['seqlen'], pre_hook=bwd_dkdv_pre_hook)
@triton.jit
def bwd_dk_dv_kernel(
    q_desc, k_desc, v_desc, do_desc, dk_desc, dv_desc, o_desc,
    Q, K, V, dO, dK, dV, O, L,
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
    
    k_4d = k_desc.load([pid_b, pid_h, kv_start, 0])
    k = tl.reshape(k_4d, (BLOCK_N, HEAD_DIM))
    
    v_4d = v_desc.load([pid_b, pid_h, kv_start, 0])
    v = tl.reshape(v_4d, (BLOCK_N, HEAD_DIM))
    
    dk = tl.zeros((BLOCK_N, HEAD_DIM), tl.float32)
    dv = tl.zeros((BLOCK_N, HEAD_DIM), tl.float32)
    
    kv_end = tl.minimum(kv_start + BLOCK_N, seqlen)
    q_start_tile = kv_start // BLOCK_M
    num_q_tiles = tl.cdiv(seqlen, BLOCK_M)
    
    first_unmasked_q = tl.minimum(tl.cdiv(kv_end, BLOCK_M), num_q_tiles)
    
    offs_n = kv_start + tl.arange(0, BLOCK_N)
    mask_n = offs_n < seqlen
    offs_m = tl.arange(0, BLOCK_M)
    
    RCP_LN2 = 1.4426950408889634
    
    # 1. Causally-masked Overlapped Loop Stream 
    for q_tile in range(q_start_tile, first_unmasked_q):
        q_start = q_tile * BLOCK_M
        
        q_4d = q_desc.load([pid_b, pid_h, q_start, 0])
        q = tl.reshape(q_4d, (BLOCK_M, HEAD_DIM))
        
        do_4d = do_desc.load([pid_b, pid_h, q_start, 0])
        do = tl.reshape(do_4d, (BLOCK_M, HEAD_DIM))
        
        o_4d = o_desc.load([pid_b, pid_h, q_start, 0])
        o = tl.reshape(o_4d, (BLOCK_M, HEAD_DIM))
        
        curr_m = q_start + offs_m
        mask_m = curr_m < seqlen
        curr_m_clamped = tl.minimum(curr_m, seqlen - 1)
        
        l_ptrs = L + pid_b * stride_lb + pid_h * stride_lh + curr_m_clamped * stride_ls
        lse = tl.load(l_ptrs, mask=mask_m, other=0.0)
        
        delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
        scores_t = tl.dot(k, q.T, out_dtype=tl.float32) * softmax_scale
        
        causal_mask_t = curr_m[None, :] >= offs_n[:, None]
        valid_mask_t = causal_mask_t & mask_m[None, :] & mask_n[:, None]
        
        scores_t = tl.where(valid_mask_t, scores_t, float('-inf'))
        p_t = tl.math.exp2((scores_t - lse[None, :]) * RCP_LN2)
        
        dv += tl.dot(p_t.to(tl.bfloat16), do, out_dtype=tl.float32)
        
        dp_t = tl.dot(v, do.T, out_dtype=tl.float32)
        ds_t = p_t * (dp_t - delta[None, :]) * softmax_scale
        ds_t = tl.where(valid_mask_t, ds_t, 0.0)
        
        dk += tl.dot(ds_t.to(tl.bfloat16), q, out_dtype=tl.float32)
        
    # 2. Fully-unmasked Loop Stream
    for q_tile in range(first_unmasked_q, num_q_tiles):
        q_start = q_tile * BLOCK_M
        
        q_4d = q_desc.load([pid_b, pid_h, q_start, 0])
        q = tl.reshape(q_4d, (BLOCK_M, HEAD_DIM))
        
        do_4d = do_desc.load([pid_b, pid_h, q_start, 0])
        do = tl.reshape(do_4d, (BLOCK_M, HEAD_DIM))
        
        o_4d = o_desc.load([pid_b, pid_h, q_start, 0])
        o = tl.reshape(o_4d, (BLOCK_M, HEAD_DIM))
        
        curr_m = q_start + offs_m
        mask_m = curr_m < seqlen
        curr_m_clamped = tl.minimum(curr_m, seqlen - 1)
        
        l_ptrs = L + pid_b * stride_lb + pid_h * stride_lh + curr_m_clamped * stride_ls
        lse = tl.load(l_ptrs, mask=mask_m, other=0.0)
        
        delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
        scores_t = tl.dot(k, q.T, out_dtype=tl.float32) * softmax_scale
        
        valid_mask_t = mask_m[None, :] & mask_n[:, None]
        scores_t = tl.where(valid_mask_t, scores_t, float('-inf'))
        p_t = tl.math.exp2((scores_t - lse[None, :]) * RCP_LN2)
        
        dv += tl.dot(p_t.to(tl.bfloat16), do, out_dtype=tl.float32)
        
        dp_t = tl.dot(v, do.T, out_dtype=tl.float32)
        ds_t = p_t * (dp_t - delta[None, :]) * softmax_scale
        ds_t = tl.where(valid_mask_t, ds_t, 0.0)
        
        dk += tl.dot(ds_t.to(tl.bfloat16), q, out_dtype=tl.float32)
        
    dk_4d = tl.reshape(dk.to(tl.bfloat16), (1, 1, BLOCK_N, HEAD_DIM))
    dk_desc.store([pid_b, pid_h, kv_start, 0], dk_4d)
    
    dv_4d = tl.reshape(dv.to(tl.bfloat16), (1, 1, BLOCK_N, HEAD_DIM))
    dv_desc.store([pid_b, pid_h, kv_start, 0], dv_4d)

def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    softmax_scale = 1.0 / (D ** 0.5)
    
    # Independent Ownership Segment 1: Produce dQ mapping queries exclusively over corresponding KV items
    grid_dq = lambda META: (
        triton.cdiv(S, META['BLOCK_M']),
        B * H,
    )
    bwd_dq_kernel[grid_dq](
        q_desc=None, k_desc=None, v_desc=None, do_desc=None, dq_desc=None, o_desc=None,
        Q=Q, K=K, V=V, dO=dO, dQ=dQ, O=O, L=L,
        stride_lb=L.stride(0), stride_lh=L.stride(1), stride_ls=L.stride(2),
        seqlen=S, softmax_scale=softmax_scale,
        H=H, HEAD_DIM=D,
    )
    
    # Independent Ownership Segment 2: Produce dK,dV grouping corresponding active Query sets exclusively mapped over Keys
    grid_dkdv = lambda META: (
        triton.cdiv(S, META['BLOCK_N']),
        B * H,
    )
    bwd_dk_dv_kernel[grid_dkdv](
        q_desc=None, k_desc=None, v_desc=None, do_desc=None, dk_desc=None, dv_desc=None, o_desc=None,
        Q=Q, K=K, V=V, dO=dO, dK=dK, dV=dV, O=O, L=L,
        stride_lb=L.stride(0), stride_lh=L.stride(1), stride_ls=L.stride(2),
        seqlen=S, softmax_scale=softmax_scale,
        H=H, HEAD_DIM=D,
    )