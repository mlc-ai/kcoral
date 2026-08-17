import math
import torch
import triton
import triton.language as tl

# Define the descriptor allocator for device-side descriptor bindings
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

@triton.autotune(
    configs=[
        # 128x128 Tiles - Optimized for Blackwell Shared Memory capabilities
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'WARP_SPECIALIZE': True, 'STAGE_COUNT': 3}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'WARP_SPECIALIZE': False, 'STAGE_COUNT': 3}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'WARP_SPECIALIZE': True, 'STAGE_COUNT': 2}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'WARP_SPECIALIZE': False, 'STAGE_COUNT': 2}, num_stages=2, num_warps=8),
        
        # 128x64 Tiles - Wider pipelining for memory bandwidth hiding 
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'WARP_SPECIALIZE': True, 'STAGE_COUNT': 3}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'WARP_SPECIALIZE': False, 'STAGE_COUNT': 3}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'WARP_SPECIALIZE': True, 'STAGE_COUNT': 4}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'WARP_SPECIALIZE': False, 'STAGE_COUNT': 4}, num_stages=4, num_warps=8),
        
        # 64x128 Tiles
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'WARP_SPECIALIZE': True, 'STAGE_COUNT': 3}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'WARP_SPECIALIZE': False, 'STAGE_COUNT': 3}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'WARP_SPECIALIZE': True, 'STAGE_COUNT': 4}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'WARP_SPECIALIZE': False, 'STAGE_COUNT': 4}, num_stages=4, num_warps=8),
    ],
    key=['S']
)
@triton.jit
def _attn_fwd_kernel(
    Q, K, V, O, LSE,
    sm_scale_log2,
    stride_qz, stride_qh, stride_qs,
    stride_kz, stride_kh, stride_ks,
    stride_vz, stride_vh, stride_vs,
    stride_oz, stride_oh, stride_os,
    stride_lsez, stride_lseh, stride_lsem,
    S, H,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr, STAGE_COUNT: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    batch = off_hz // H
    head = off_hz % H
    
    # Retrieve proper sequence offset pointers
    q_offset = batch * stride_qz + head * stride_qh
    k_offset = batch * stride_kz + head * stride_kh
    v_offset = batch * stride_vz + head * stride_vh
    o_offset = batch * stride_oz + head * stride_oh
    lse_offset = batch * stride_lsez + head * stride_lseh
    
    # Generate 2D Hardware-Bound Tensor Descriptors completely supplanting manual pointer addressing
    q_desc = tl.make_tensor_descriptor(
        Q + q_offset, shape=[S, BLOCK_DMODEL], strides=[stride_qs, 1],
        block_shape=[BLOCK_M, BLOCK_DMODEL], padding_option="zero"
    )
    k_desc = tl.make_tensor_descriptor(
        K + k_offset, shape=[S, BLOCK_DMODEL], strides=[stride_ks, 1],
        block_shape=[BLOCK_N, BLOCK_DMODEL], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        V + v_offset, shape=[S, BLOCK_DMODEL], strides=[stride_vs, 1],
        block_shape=[BLOCK_N, BLOCK_DMODEL], padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        O + o_offset, shape=[S, BLOCK_DMODEL], strides=[stride_os, 1],
        block_shape=[BLOCK_M, BLOCK_DMODEL], padding_option="zero"
    )
    
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    
    # Load primary Query sequence directly keeping it inside WGMMA-compliant raw formats
    q = q_desc.load([start_m * BLOCK_M, 0])
    
    offs_n_base = tl.arange(0, BLOCK_N)
    
    for start_n_idx in tl.range(0, tl.cdiv(S, BLOCK_N), num_stages=STAGE_COUNT, warp_specialize=WARP_SPECIALIZE):
        start_n = start_n_idx * BLOCK_N
        k = k_desc.load([start_n, 0])
        
        # Hardware dot matrix computation with FP32 Accumulators
        qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32)
        
        # Map onto a base-2 exponent space explicitly minimizing FLOPs matching cuDNN flash
        qk = qk * sm_scale_log2
        
        offs_n = start_n + offs_n_base
        qk = tl.where(offs_n[None, :] < S, qk, float("-inf"))
        
        m_ij = tl.max(qk, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        
        # Utilizing hardware intrinsic base-2 exponentiations mapping precisely matching LSE requirements
        alpha = tl.exp2(m_i - m_i_new)
        p = tl.exp2(qk - m_i_new[:, None])
        
        acc = acc * alpha[:, None]
        
        v = v_desc.load([start_n, 0])
        p_cast = p.to(tl.bfloat16)
        acc = tl.dot(p_cast, v, acc, out_dtype=tl.float32)
        
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new

    # Sequence Tail Epilogue formatting
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < S
    
    l_i_safe = tl.where(m_mask, l_i, 1.0)
    
    # Reconvert extracted scalars strictly back to math.e natural-log domain seamlessly for parity
    lse = (m_i * 0.6931471805599453) + tl.log(l_i_safe)
    
    lse_ptrs = LSE + lse_offset + offs_m * stride_lsem
    tl.store(lse_ptrs, lse, mask=m_mask)
    
    l_i_inv = 1.0 / l_i_safe
    acc = acc * l_i_inv[:, None]
    
    # TMA implicitly guards padded out-of-bounds row truncations naturally discarding them
    o_desc.store([start_m * BLOCK_M, 0], acc.to(tl.bfloat16))

def run(Q, K, V, O, LSE):
    """
    Computes Standard Attention forward pass and Log-Sum-Exp statistics (LSE).
    
    Args:
        Q: [B, H, S, D] dtype bfloat16
        K: [B, H, S, D] dtype bfloat16
        V: [B, H, S, D] dtype bfloat16
        O: [B, H, S, D] preallocated destination tensor dtype bfloat16
        LSE: [B, H, S] preallocated destination tensor dtype float32
    """
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    # Pre-compute Base 2 scalar conversions for extreme inner-loop FLOP eliminations (1.4426950408889634 == log2(e))
    sm_scale_log2 = (1.0 / math.sqrt(D)) * 1.4426950408889634
    
    if S > 0:
        grid = lambda META: (triton.cdiv(S, META['BLOCK_M']), B * H, 1)
        _attn_fwd_kernel[grid](
            Q, K, V, O, LSE,
            sm_scale_log2,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            LSE.stride(0), LSE.stride(1), LSE.stride(2),
            S, H,
            BLOCK_DMODEL=D,
        )