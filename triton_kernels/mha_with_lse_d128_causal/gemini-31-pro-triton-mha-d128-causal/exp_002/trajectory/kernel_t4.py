import torch
import triton
import triton.language as tl
import math

def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

# Install the descriptor allocator for Hopper TMA
triton.set_allocator(alloc_fn)

def get_configs():
    return [
        # Large tiles for high arithmetic intensity
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        
        # Balanced tiles with deep software pipelining (optimal for causal masking balance)
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=5),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=6),
        
        # Memory-bound variants (loads more Q, fewer K/V)
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        
        # Fallback dense config
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
    ]

@triton.autotune(
    configs=get_configs(),
    key=['S']
)
@triton.jit
def _attn_fwd_kernel(
    Q, K, V, sm_scale_log2, O, LSE,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    B, H, S,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    batch_idx = off_hz // H
    head_idx = off_hz % H
    
    q_offset = batch_idx * stride_qb + head_idx * stride_qh
    k_offset = batch_idx * stride_kb + head_idx * stride_kh
    v_offset = batch_idx * stride_vb + head_idx * stride_vh
    o_offset = batch_idx * stride_ob + head_idx * stride_oh
    
    # Create Hopper TMA descriptors dynamically per head
    q_desc = tl.make_tensor_descriptor(
        Q + q_offset, shape=[S, BLOCK_D], strides=[stride_qs, stride_qd],
        block_shape=[BLOCK_M, BLOCK_D], padding_option="zero"
    )
    k_desc = tl.make_tensor_descriptor(
        K + k_offset, shape=[S, BLOCK_D], strides=[stride_ks, stride_kd],
        block_shape=[BLOCK_N, BLOCK_D], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        V + v_offset, shape=[S, BLOCK_D], strides=[stride_vs, stride_vd],
        block_shape=[BLOCK_N, BLOCK_D], padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        O + o_offset, shape=[S, BLOCK_D], strides=[stride_os, stride_od],
        block_shape=[BLOCK_M, BLOCK_D]
    )
    
    offset_m = start_m * BLOCK_M
    q = q_desc.load([offset_m, 0])
    
    # Pre-scale Q with combined (sm_scale * log2(e)) to entirely eliminate per-iteration scaling in the inner loop
    q = (q * sm_scale_log2).to(tl.bfloat16)
    
    # Initialize online softmax statistics
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    offs_m = offset_m + tl.arange(0, BLOCK_M)
    
    # Separate fully unmasked blocks from partially masked blocks (causal diagonal)
    num_full_blocks = offset_m // BLOCK_N
    max_n = tl.minimum(S, offset_m + BLOCK_M)
    num_total_blocks = tl.cdiv(max_n, BLOCK_N)
    
    num_full_blocks = tl.minimum(num_full_blocks, num_total_blocks)
    
    # 1. Core loop over fully valid blocks (bypasses masking entirely)
    for start_n_idx in tl.range(0, num_full_blocks):
        offset_n = start_n_idx * BLOCK_N
        k = k_desc.load([offset_n, 0])
        v = v_desc.load([offset_n, 0])
        
        # Hopper WGMMA natively accepts K^T
        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        
        # Log-sum-exp numerically stable statistics
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        
        alpha = tl.exp2(m_i - m_ij)
        
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        
        m_i = m_ij

    # 2. Epilogue loop for remaining blocks requiring causal constraints
    for start_n_idx in tl.range(num_full_blocks, num_total_blocks):
        offset_n = start_n_idx * BLOCK_N
        k = k_desc.load([offset_n, 0])
        v = v_desc.load([offset_n, 0])
        
        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        
        offs_n = offset_n + tl.arange(0, BLOCK_N)
        mask = offs_m[:, None] >= offs_n[None, :]
        qk = tl.where(mask, qk, float("-inf"))
        
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        
        alpha = tl.exp2(m_i - m_ij)
        
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        
        m_i = m_ij

    # Scale final accumulated values and map from log2(e) scale back to ln(e)
    acc = acc / l_i[:, None]
    lse = (m_i + tl.log2(l_i)) * 0.6931471805599453
    
    # Store back via TMA descriptor; natively handles sequence length boundaries padding limits
    o_desc.store([offset_m, 0], acc.to(tl.bfloat16))
    
    # Store LSE metadata back to global memory bounded by exact sequence length
    lse_offset = batch_idx * stride_lseb + head_idx * stride_lseh
    lse_ptrs = LSE + lse_offset + offs_m * stride_lses
    tl.store(lse_ptrs, lse, mask=offs_m < S)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    # Natural block-dispatch mapped linearly maximizes K/V locality on L2 cache natively
    grid = lambda META: (
        triton.cdiv(S, META['BLOCK_M']),
        B * H
    )
    
    # Combine scalar product constants: scaling by math.sqrt(D) and changing basis to log2
    sm_scale = 1.0 / math.sqrt(D)
    sm_scale_log2 = sm_scale * 1.4426950408889634
    
    _attn_fwd_kernel[grid](
        Q, K, V, sm_scale_log2, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        B, H, S,
        BLOCK_D=128
    )