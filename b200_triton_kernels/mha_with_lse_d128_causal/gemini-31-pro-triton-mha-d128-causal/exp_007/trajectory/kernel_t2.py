import torch
import triton
import triton.language as tl
import math

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=5, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
    ],
    key=['S']
)
@triton.jit
def _attn_fwd_kernel(
    Q, K, V, sm_scale,
    O, LSE,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    B, H, S, 
    BLOCK_D: tl.constexpr,
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    batch_head = tl.program_id(1)
    
    offset_m = start_m * BLOCK_M
    # Early exit if this CTA handles purely out-of-bounds queries
    if offset_m >= S:
        return
        
    batch_idx = batch_head // H
    head_idx = batch_head % H
    
    q_offset = batch_idx * stride_qb + head_idx * stride_qh
    k_offset = batch_idx * stride_kb + head_idx * stride_kh
    v_offset = batch_idx * stride_vb + head_idx * stride_vh
    o_offset = batch_idx * stride_ob + head_idx * stride_oh
    lse_offset = batch_idx * stride_lseb + head_idx * stride_lseh
    
    # Hopper TMA Tensor Descriptors for global memory accesses. 
    # Device-created descriptors are efficient and avoid complex host-side padding logic.
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
    
    # Load Query tile and dynamically scale once up-front to save internal loop ALU overhead
    q = q_desc.load([offset_m, 0])
    q = tl.cast(q, tl.float32) * sm_scale
    q = tl.cast(q, tl.bfloat16)
    
    # FlashAttention state initialization
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    n_start_max = offset_m + BLOCK_M
    if n_start_max > S:
        n_start_max = S
    
    num_n_blocks = tl.cdiv(n_start_max, BLOCK_N)
    
    offs_m_b = tl.arange(0, BLOCK_M)
    offs_n_b = tl.arange(0, BLOCK_N)
    m_idx = offset_m + offs_m_b
    
    # Main attention WGMMA dot-product pipeline loop
    for start_n in range(0, num_n_blocks):
        offset_n = start_n * BLOCK_N
        
        k = k_desc.load([offset_n, 0])
        v = v_desc.load([offset_n, 0])
        
        # WGMMA automatically supports logically transposed operands (K.T)
        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        
        n_idx = offset_n + offs_n_b
        # Standard strict causal attention masking (valid when query_pos >= key_pos)
        is_valid = m_idx[:, None] >= n_idx[None, :]
        
        qk = tl.where(is_valid, qk, float("-inf"))
        
        # Softmax numerically stable accumulation steps
        m_i_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        
        p_bf16 = tl.cast(p, tl.bfloat16)
        acc = acc * alpha[:, None] + tl.dot(p_bf16, v, out_dtype=tl.float32)
        
        m_i = m_i_new

    # Normalization step over the accumulating probability weightage
    acc = acc / l_i[:, None]
    lse = m_i + tl.log(l_i)
    
    # TMA auto-ignores stores hitting `offset_m + i >= S`
    o_desc.store([offset_m, 0], tl.cast(acc, tl.bfloat16))
    
    # Finalize Log-Sum-Exp outputs securely via conventional pointer mask operations
    m_mask = m_idx < S
    lse_ptrs = LSE + lse_offset + m_idx * stride_lses
    tl.store(lse_ptrs, lse, mask=m_mask)


def run(Q, K, V, O, LSE):
    """
    Computes causal multi-head attention forward and natural-log-sum-exp metrics securely on Hopper TMA grids.
    Destination tensors strictly filled in-place without reallocation footprint logic.
    """
    def alloc_fn(size: int, alignment: int, stream):
        return torch.empty(size, device="cuda", dtype=torch.int8)
    
    triton.set_allocator(alloc_fn)
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    
    # Grid limits program distribution mapped precisely corresponding to physical boundary limits across SMs.
    grid = lambda META: (triton.cdiv(S, META['BLOCK_M']), B * H)
    sm_scale = 1.0 / math.sqrt(D)
    
    _attn_fwd_kernel[grid](
        Q, K, V, sm_scale,
        O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        B, H, S, 
        BLOCK_D=D,
    )