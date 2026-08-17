import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _bwd_dq_kernel(
    q_desc, k_desc, v_desc, do_desc, dq_desc,
    o_ptr, l_ptr,
    S, 
    stride_oz, stride_oh, stride_os, stride_od,
    stride_lz, stride_lh, stride_ls,
    scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, d: tl.constexpr, H: tl.constexpr
):
    q_tile = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    
    q_tile_offset = q_tile * BLOCK_M
    
    q = tl.reshape(q_desc.load([b, h, q_tile_offset, 0]), (BLOCK_M, d))
    do = tl.reshape(do_desc.load([b, h, q_tile_offset, 0]), (BLOCK_M, d))
    
    offs_m = q_tile_offset + tl.arange(0, BLOCK_M)
    mask_m = offs_m < S
    
    # Load O and L via pointers to avoid TMA shared-memory allocation 
    # and allow larger 128x128 matmul tiles within the SM limit.
    o_base = o_ptr + b * stride_oz + h * stride_oh
    offs_d = tl.arange(0, d)
    o_ptrs = o_base + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
    
    l_ptrs = l_ptr + b * stride_lz + h * stride_lh + offs_m * stride_ls
    l = tl.load(l_ptrs, mask=mask_m, other=0.0)
    
    delta = tl.sum(do.to(tl.float32) * o.to(tl.float32), axis=1)
    
    dq = tl.zeros((BLOCK_M, d), dtype=tl.float32)
    
    num_kv_tiles = tl.cdiv(S, BLOCK_N)
    for kv_tile in tl.range(0, num_kv_tiles, num_stages=2):
        kv_tile_offset = kv_tile * BLOCK_N
        
        k = tl.reshape(k_desc.load([b, h, kv_tile_offset, 0]), (BLOCK_N, d))
        v = tl.reshape(v_desc.load([b, h, kv_tile_offset, 0]), (BLOCK_N, d))
        
        offs_n = kv_tile_offset + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S
        mask = mask_m[:, None] & mask_n[None, :]
        
        scores = tl.dot(q, k.T) * scale
        scores = tl.where(mask, scores, -float('inf'))
        
        p = tl.math.exp(scores - l[:, None])
        
        dp = tl.dot(do, v.T)
        ds = tl.where(mask, p * (dp - delta[:, None]) * scale, 0.0)
        
        dq += tl.dot(ds.to(tl.bfloat16), k)
        
    dq_desc.store([b, h, q_tile_offset, 0], tl.reshape(dq.to(tl.bfloat16), (1, 1, BLOCK_M, d)))


@triton.jit
def _bwd_dk_dv_kernel(
    q_desc, k_desc, v_desc, do_desc, dk_desc, dv_desc,
    o_ptr, l_ptr,
    S,
    stride_oz, stride_oh, stride_os, stride_od,
    stride_lz, stride_lh, stride_ls,
    scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, d: tl.constexpr, H: tl.constexpr
):
    kv_tile = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    
    kv_tile_offset = kv_tile * BLOCK_N
    
    k = tl.reshape(k_desc.load([b, h, kv_tile_offset, 0]), (BLOCK_N, d))
    v = tl.reshape(v_desc.load([b, h, kv_tile_offset, 0]), (BLOCK_N, d))
    
    offs_n = kv_tile_offset + tl.arange(0, BLOCK_N)
    mask_n = offs_n < S
    
    dk = tl.zeros((BLOCK_N, d), dtype=tl.float32)
    dv = tl.zeros((BLOCK_N, d), dtype=tl.float32)
    
    # Initialize base pointers for O and L
    o_base = o_ptr + b * stride_oz + h * stride_oh
    offs_m_init = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, d)
    o_ptrs = o_base + offs_m_init[:, None] * stride_os + offs_d[None, :] * stride_od
    
    l_base = l_ptr + b * stride_lz + h * stride_lh
    l_ptrs = l_base + offs_m_init * stride_ls
    
    num_q_tiles = tl.cdiv(S, BLOCK_M)
    for q_tile in tl.range(0, num_q_tiles, num_stages=2):
        q_tile_offset = q_tile * BLOCK_M
        
        q = tl.reshape(q_desc.load([b, h, q_tile_offset, 0]), (BLOCK_M, d))
        do = tl.reshape(do_desc.load([b, h, q_tile_offset, 0]), (BLOCK_M, d))
        
        offs_m = q_tile_offset + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S
        mask = mask_n[:, None] & mask_m[None, :]
        
        o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
        l = tl.load(l_ptrs, mask=mask_m, other=0.0)
        
        # Advance pointers for the next stage
        o_ptrs += BLOCK_M * stride_os
        l_ptrs += BLOCK_M * stride_ls
        
        delta = tl.sum(do.to(tl.float32) * o.to(tl.float32), axis=1)
        
        scores_t = tl.dot(k, q.T) * scale
        scores_t = tl.where(mask, scores_t, -float('inf'))
        
        p_t = tl.math.exp(scores_t - l[None, :])
        
        dv += tl.dot(p_t.to(tl.bfloat16), do)
        
        dp_t = tl.dot(v, do.T)
        ds_t = tl.where(mask, p_t * (dp_t - delta[None, :]) * scale, 0.0)
        
        dk += tl.dot(ds_t.to(tl.bfloat16), q)
        
    dk_desc.store([b, h, kv_tile_offset, 0], tl.reshape(dk.to(tl.bfloat16), (1, 1, BLOCK_N, d)))
    dv_desc.store([b, h, kv_tile_offset, 0], tl.reshape(dv.to(tl.bfloat16), (1, 1, BLOCK_N, d)))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes backward gradients for multi-head attention using Blackwell-native standard Triton logic.
    Optimized with `TensorDescriptor` TMA loads/stores and split-ownership mapping.
    O and L are loaded via standard pointers to avoid TMA shared memory footprint limitations, 
    allowing 128x128 blocking for optimal bandwidth utilization on SM100.
    """
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    
    scale = 1.0 / (d ** 0.5)
    
    BLOCK = 128
    
    q_desc = TensorDescriptor.from_tensor(Q, [1, 1, BLOCK, d])
    k_desc = TensorDescriptor.from_tensor(K, [1, 1, BLOCK, d])
    v_desc = TensorDescriptor.from_tensor(V, [1, 1, BLOCK, d])
    do_desc = TensorDescriptor.from_tensor(dO, [1, 1, BLOCK, d])
    dq_desc = TensorDescriptor.from_tensor(dQ, [1, 1, BLOCK, d])
    dk_desc = TensorDescriptor.from_tensor(dK, [1, 1, BLOCK, d])
    dv_desc = TensorDescriptor.from_tensor(dV, [1, 1, BLOCK, d])
    
    # ------------------------------------------------------------------
    # 1. dQ Ownership Launch
    # ------------------------------------------------------------------
    num_q_tiles = triton.cdiv(S, BLOCK)
    
    _bwd_dq_kernel[(num_q_tiles, B * H)](
        q_desc, k_desc, v_desc, do_desc, dq_desc,
        O, L,
        S,
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        scale,
        BLOCK_M=BLOCK, BLOCK_N=BLOCK, d=d, H=H,
        num_warps=8, num_stages=2
    )
    
    # ------------------------------------------------------------------
    # 2. dK, dV Ownership Launch
    # ------------------------------------------------------------------
    num_kv_tiles = triton.cdiv(S, BLOCK)
    
    _bwd_dk_dv_kernel[(num_kv_tiles, B * H)](
        q_desc, k_desc, v_desc, do_desc, dk_desc, dv_desc,
        O, L,
        S,
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        scale,
        BLOCK_M=BLOCK, BLOCK_N=BLOCK, d=d, H=H,
        num_warps=8, num_stages=2
    )