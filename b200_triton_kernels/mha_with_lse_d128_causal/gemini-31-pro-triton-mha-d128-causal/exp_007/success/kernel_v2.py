import torch
import triton
import triton.language as tl
import math

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
    ],
    key=['S']
)
@triton.jit
def _attn_fwd_kernel(
    Q, K, V, sm_scale_log2,
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
    # Early exit for CTAs entirely out of sequence bounds
    if offset_m >= S:
        return
        
    batch_idx = batch_head // H
    head_idx = batch_head % H
    
    # Compute base offsets for this batch/head
    q_offset = batch_idx * stride_qb + head_idx * stride_qh
    k_offset = batch_idx * stride_kb + head_idx * stride_kh
    v_offset = batch_idx * stride_vb + head_idx * stride_vh
    o_offset = batch_idx * stride_ob + head_idx * stride_oh
    lse_offset = batch_idx * stride_lseb + head_idx * stride_lseh
    
    # Device-created Hopper TMA descriptors for efficient 2D tile asynchronous bulk transfers.
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
    
    # Load query matrix tile directly into registers via TMA
    q = q_desc.load([offset_m, 0])
    
    # Online Softmax states initialization
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    # Fully-computed non-causal blocks where query strictly proceeds keys (n < offset_m)
    num_full_blocks = offset_m // BLOCK_N
    
    # Stage 1: Process full keys/values blocks without any causal mask checks.
    # We use exp2 for much faster native hardware MUFU.EX2 operations.
    for start_n in range(num_full_blocks):
        offset_n = start_n * BLOCK_N
        
        k = k_desc.load([offset_n, 0])
        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        
        # Scaling integrated efficiently over fp32 dot accumulation
        qk = qk * sm_scale_log2
        
        m_i_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp2(m_i - m_i_new)
        p = tl.exp2(qk - m_i_new[:, None])
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        p_bf16 = tl.cast(p, tl.bfloat16)
        
        v = v_desc.load([offset_n, 0])
        acc = acc * alpha[:, None] + tl.dot(p_bf16, v, out_dtype=tl.float32)
        
        m_i = m_i_new

    # Stage 2: Process boundary/causal blocks with strictly enforced mask checks.
    n_start_max = offset_m + BLOCK_M
    if n_start_max > S:
        n_start_max = S
    
    num_n_blocks = tl.cdiv(n_start_max, BLOCK_N)
    offs_m = offset_m + tl.arange(0, BLOCK_M)
    
    for start_n in range(num_full_blocks, num_n_blocks):
        offset_n = start_n * BLOCK_N
        
        k = k_desc.load([offset_n, 0])
        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        qk = qk * sm_scale_log2
        
        offs_n = offset_n + tl.arange(0, BLOCK_N)
        
        # Simple logical causality coordinate validation
        is_valid = offs_m[:, None] >= offs_n[None, :]
        qk = tl.where(is_valid, qk, float("-inf"))
        
        m_i_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp2(m_i - m_i_new)
        p = tl.exp2(qk - m_i_new[:, None])
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        p_bf16 = tl.cast(p, tl.bfloat16)
        
        v = v_desc.load([offset_n, 0])
        acc = acc * alpha[:, None] + tl.dot(p_bf16, v, out_dtype=tl.float32)
        
        m_i = m_i_new

    # Normalize accumulator probabilities
    acc = acc / l_i[:, None]
    
    # Scale base-2 m_i to natural logarithms equivalent base-e matching output specification.
    # ln(2) = 0.6931471805599453
    lse = m_i * 0.6931471805599453 + tl.log(l_i)
    
    # Transparently saves only validated sequence extents through TMA shape bounds
    o_desc.store([offset_m, 0], tl.cast(acc, tl.bfloat16))
    
    # Store explicit masked LSE
    m_mask = offs_m < S
    lse_ptrs = LSE + lse_offset + offs_m * stride_lses
    tl.store(lse_ptrs, lse, mask=m_mask)


def run(Q, K, V, O, LSE):
    """
    Computes causal multi-head attention forward and natural-log-sum-exp metrics effectively mapped
    to TMA paths and WGMMA accelerated hardware dot tensor products on Hopper SM90 architectures.
    Results are securely persisted in preallocated destination tensors `O` and `LSE`.
    """
    def alloc_fn(size: int, alignment: int, stream):
        return torch.empty(size, device="cuda", dtype=torch.int8)
    
    # Install Triton TMA device descriptor allocator temporarily
    triton.set_allocator(alloc_fn)
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    
    # Map programs logically to coordinate bounds across blocks ensuring coherent batch-head cache utilization
    grid = lambda META: (triton.cdiv(S, META['BLOCK_M']), B * H)
    
    # We factor out ln(2) to directly perform base-2 MUFU.EX2 operations internally saving instructions
    sm_scale_log2 = (1.0 / math.sqrt(D)) * 1.4426950408889634
    
    _attn_fwd_kernel[grid](
        Q, K, V, sm_scale_log2,
        O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        B, H, S, 
        BLOCK_D=D,
    )