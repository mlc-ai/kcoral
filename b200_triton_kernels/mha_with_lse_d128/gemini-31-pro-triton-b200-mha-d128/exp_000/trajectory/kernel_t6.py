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
        # Safe Shared Memory Topologies (< 228 KiB Limit enforced mathematically)
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64},  num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64},  num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128}, num_stages=4, num_warps=8),
        
        # Clustered Execution paths (L2 Cache / DSMEM Read Sharing Amplification)
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=2, num_warps=8, num_ctas=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=2, num_warps=8, num_ctas=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64},  num_stages=4, num_warps=8, num_ctas=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128}, num_stages=4, num_warps=8, num_ctas=2),
    ],
    key=['S']
)
@triton.heuristics({
    'DIVISIBLE_M': lambda args: args['S'] % args['BLOCK_M'] == 0,
    'DIVISIBLE_N': lambda args: args['S'] % args['BLOCK_N'] == 0,
})
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
    DIVISIBLE_M: tl.constexpr, DIVISIBLE_N: tl.constexpr,
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
    
    # Generate 2D hardware-bound tensor descriptors completely supplanting pointer ops
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
    
    # Pre-scale Q heavily streamlining inner loop ops translating into natural base-2 natively
    q = q_desc.load([start_m * BLOCK_M, 0])
    q = (q * sm_scale_log2).to(tl.bfloat16)
    
    offs_n_base = tl.arange(0, BLOCK_N)
    
    for start_n in range(0, S, BLOCK_N):
        k = k_desc.load([start_n, 0])
        qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32)
        
        if not DIVISIBLE_N:
            offs_n = start_n + offs_n_base
            qk = tl.where(offs_n[None, :] < S, qk, float("-inf"))
        
        m_ij = tl.max(qk, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        
        # Intrinsic hardware exponentiations
        alpha = tl.exp2(m_i - m_i_new)
        p = tl.exp2(qk - m_i_new[:, None])
        
        acc = acc * alpha[:, None]
        
        v = v_desc.load([start_n, 0])
        acc = tl.dot(p.to(tl.bfloat16), v, acc, out_dtype=tl.float32)
        
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new

    # Sequence Tail Epilogue scaling
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    
    if DIVISIBLE_M:
        # Revert scalar domain back to math.e for cuDNN parity 
        lse = (m_i * 0.6931471805599453) + tl.log(l_i)
        lse_ptrs = LSE + lse_offset + offs_m * stride_lsem
        tl.store(lse_ptrs, lse)
        
        l_i_inv = 1.0 / l_i
        acc = acc * l_i_inv[:, None]
        
        # TMA inherently resolves bounds discarding execution drops beyond valid S matrices seamlessly
        o_desc.store([start_m * BLOCK_M, 0], acc.to(tl.bfloat16))
    else:
        m_mask = offs_m < S
        l_i_safe = tl.where(m_mask, l_i, 1.0)
        lse = (m_i * 0.6931471805599453) + tl.log(l_i_safe) 
        
        lse_ptrs = LSE + lse_offset + offs_m * stride_lsem
        tl.store(lse_ptrs, lse, mask=m_mask)
        
        l_i_inv = 1.0 / l_i_safe
        acc = acc * l_i_inv[:, None]
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
    
    # Hardware exponentiations mathematically translate QK scales tracking logic identically mapping Base E 
    # (1.4426950408889634 === math.log2(math.e))
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