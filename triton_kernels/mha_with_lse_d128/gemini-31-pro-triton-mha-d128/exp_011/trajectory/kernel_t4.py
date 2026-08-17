import math
import torch
import triton
import triton.language as tl

def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device=torch.cuda.current_device(), dtype=torch.int8)

triton.set_allocator(alloc_fn)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=5, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=5, num_warps=4),
    ],
    key=['S'],
)
@triton.jit
def mha_fwd_kernel(
    Q, K, V, O, LSE,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    B, H, S,
    sm_scale_log2, ln_2,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    # Map program instances to grid coordinates
    start_m = tl.program_id(0)
    off_h = tl.program_id(1)
    off_b = tl.program_id(2)

    # Calculate pointers to the current batch and head
    q_offset = off_b * stride_qb + off_h * stride_qh
    k_offset = off_b * stride_kb + off_h * stride_kh
    v_offset = off_b * stride_vb + off_h * stride_vh
    o_offset = off_b * stride_ob + off_h * stride_oh
    lse_offset = off_b * stride_lseb + off_h * stride_lseh

    # Setup device-side Hopper TMA descriptors
    q_desc = tl.make_tensor_descriptor(
        Q + q_offset,
        shape=[S, BLOCK_D],
        strides=[stride_qs, 1],
        block_shape=[BLOCK_M, BLOCK_D],
        padding_option="zero"
    )
    k_desc = tl.make_tensor_descriptor(
        K + k_offset,
        shape=[S, BLOCK_D],
        strides=[stride_ks, 1],
        block_shape=[BLOCK_N, BLOCK_D],
        padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        V + v_offset,
        shape=[S, BLOCK_D],
        strides=[stride_vs, 1],
        block_shape=[BLOCK_N, BLOCK_D],
        padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        O + o_offset,
        shape=[S, BLOCK_D],
        strides=[stride_os, 1],
        block_shape=[BLOCK_M, BLOCK_D]
    )

    # Load Q block via TMA. We pre-scale Q once in registers, eliminating an entire scaling operation
    # over the large QK result (saving ~16,384 multiplications per inner loop iteration).
    q = q_desc.load([start_m * BLOCK_M, 0])
    q = (q * sm_scale_log2).to(tl.bfloat16)

    # Initialize running softmax statistics and accumulator
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    num_n_blocks = tl.cdiv(S, BLOCK_N)
    
    # Software-pipelined loop mapped to Hopper hardware pipelines
    for n_idx in tl.range(num_n_blocks):
        start_n = n_idx * BLOCK_N
        
        # Prefetch K and V early via TMA
        k = k_desc.load([start_n, 0])
        v = v_desc.load([start_n, 0])
        
        # WGMMA RS GEMM: Q from registers (Register-Shared compatible path), K from Shared Memory
        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        
        # Mask out-of-bounds keys across the sequence dimension for the tail block only
        if start_n + BLOCK_N > S:
            offs_n_curr = start_n + tl.arange(0, BLOCK_N)
            qk = tl.where(offs_n_curr[None, :] < S, qk, float('-inf'))
        
        # Softmax base-2 online accumulation step
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp2(m_i - m_i_new)
        p = tl.exp2(qk - m_i_new[:, None])
        
        l_i_new = alpha * l_i + tl.sum(p, 1)
        
        # Drop precision to feed WGMMA
        p_cast = p.to(tl.bfloat16)
        
        # Rescale accumulator and perform second WGMMA RS GEMM (P from registers, V from SMEM)
        acc = acc * alpha[:, None]
        acc = tl.dot(p_cast, v, acc)
        
        # Advance statistics
        m_i = m_i_new
        l_i = l_i_new
        
    # Finalize output mathematical scaling
    l_i_inv = 1.0 / l_i
    acc = acc * l_i_inv[:, None]
    
    # LSE mathematically converted back from Base-2 to Natural Log Base-e
    lse = (m_i + tl.log2(l_i)) * ln_2
    
    # Store O via TMA gracefully avoiding out-of-bounds writes
    o_desc.store([start_m * BLOCK_M, 0], acc.to(tl.bfloat16))
    
    # Store LSE linearly
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    lse_ptrs = LSE + lse_offset + offs_m * stride_lses
    
    if start_m * BLOCK_M + BLOCK_M > S:
        tl.store(lse_ptrs, lse, mask=(offs_m < S))
    else:
        tl.store(lse_ptrs, lse)

def run(Q, K, V, O, LSE):
    """
    Computes a non-causal multi-head attention forward pass out-of-place leveraging 
    NVIDIA Hopper TMA, WGMMA operations, fast base-2 arithmetic for Softmax, and 
    mathematical pre-scaling optimizations.
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    sm_scale = 1.0 / math.sqrt(D)
    
    # Use base-2 mathematical equivalents for softmax to bypass slow `exp`/`log` HW pipelines
    log2_e = 1.4426950408889634
    ln_2 = 0.6931471805599453
    sm_scale_log2 = sm_scale * log2_e
    
    grid = lambda META: (
        triton.cdiv(S, META['BLOCK_M']),
        H,
        B
    )
    
    mha_fwd_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        B, H, S,
        sm_scale_log2, ln_2,
        BLOCK_D=D
    )