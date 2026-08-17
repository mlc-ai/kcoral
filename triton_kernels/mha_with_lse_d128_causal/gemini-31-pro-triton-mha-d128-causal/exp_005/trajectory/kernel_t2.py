import torch
import triton
import triton.language as tl


# Required allocator for Hopper device-side tensor descriptor creation
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=4),
    ],
    key=['S_len']
)
@triton.jit
def _causal_mha_fwd_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr,
    S_len, H,
    stride_qb, stride_qh, stride_qs,
    stride_kb, stride_kh, stride_ks,
    stride_vb, stride_vh, stride_vs,
    stride_ob, stride_oh, stride_os,
    stride_lseb, stride_lseh, stride_lses,
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    batch_idx = pid_bh // H
    head_idx = pid_bh % H
    
    start_m = pid_m * BLOCK_M
    
    # Early exit for entirely out-of-bounds queries
    if start_m >= S_len:
        return
        
    # Slicing bases per batch and head
    q_base = q_ptr + batch_idx * stride_qb + head_idx * stride_qh
    k_base = k_ptr + batch_idx * stride_kb + head_idx * stride_kh
    v_base = v_ptr + batch_idx * stride_vb + head_idx * stride_vh
    o_base = o_ptr + batch_idx * stride_ob + head_idx * stride_oh
    
    # Hopper TMA descriptors creation
    q_desc = tl.make_tensor_descriptor(
        q_base, shape=[S_len, BLOCK_D], strides=[stride_qs, 1],
        block_shape=[BLOCK_M, BLOCK_D], padding_option="zero"
    )
    k_desc = tl.make_tensor_descriptor(
        k_base, shape=[S_len, BLOCK_D], strides=[stride_ks, 1],
        block_shape=[BLOCK_N, BLOCK_D], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        v_base, shape=[S_len, BLOCK_D], strides=[stride_vs, 1],
        block_shape=[BLOCK_N, BLOCK_D], padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        o_base, shape=[S_len, BLOCK_D], strides=[stride_os, 1],
        block_shape=[BLOCK_M, BLOCK_D]
    )
    
    # Load Q block once
    q = q_desc.load([start_m, 0])
    
    # Trackers for numerically stable softmax
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    loop_end = start_m + BLOCK_M
    if S_len < loop_end:
        loop_end = S_len
        
    # Pre-compute block indices for causal and boundary masking
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_m_ext = offs_m[:, None]
    
    # TMA unrolled pipeline managed by compiler based on `num_stages`
    for start_n in range(0, loop_end, BLOCK_N):
        k = k_desc.load([start_n, 0])
        v = v_desc.load([start_n, 0])
        
        # Q @ K^T. The WGMMA layout handles k.T perfectly from descriptor
        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        qk = qk * sm_scale
        
        is_last_n = (start_n + BLOCK_N) > S_len
        is_causal_block = (start_n + BLOCK_N) > start_m
        
        # Minimal divergence causal and sequence bound masking
        if is_causal_block or is_last_n:
            offs_n = start_n + tl.arange(0, BLOCK_N)
            if is_causal_block and is_last_n:
                mask = (offs_m_ext >= offs_n[None, :]) & (offs_n[None, :] < S_len)
            elif is_causal_block:
                mask = (offs_m_ext >= offs_n[None, :])
            else:
                mask = offs_n[None, :] < S_len
            qk = tl.where(mask, qk, float("-inf"))
            
        # Standard Flash Attention math logic 
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        
        alpha = tl.exp(m_i - m_ij)
        acc = acc * alpha[:, None]
        
        p_bf16 = p.to(v.dtype)
        # Context accumulation. Row-major right operand `v` is native in Hopper.
        acc = tl.dot(p_bf16, v, acc)
        
        m_i = m_ij
        l_i = l_i * alpha + l_ij
        
    # Normalize Output
    l_i_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_i_safe[:, None]
    
    # TMA Store automatically drops out-of-bound `offs_m` rows padding
    o_desc.store([start_m, 0], acc.to(o_ptr.dtype.element_ty))
    
    # Non-TMA pointer path for LSE vectors
    m_mask = offs_m < S_len
    lse_base = lse_ptr + batch_idx * stride_lseb + head_idx * stride_lseh
    lse_ptrs = lse_base + offs_m * stride_lses
    lse_val = m_i + tl.log(l_i_safe)
    
    tl.store(lse_ptrs, lse_val, mask=m_mask)


def run(Q, K, V, O, LSE):
    """
    Computes causal Multi-Head Attention forward pass and returns Output and Log-Sum-Exp.
    Writes outputs directly into preallocated `O` and `LSE` tensors using destination-passing convention.
    """
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    sm_scale = 1.0 / (D ** 0.5)
    
    if S > 0:
        grid = lambda META: (triton.cdiv(S, META['BLOCK_M']), B * H)
        
        _causal_mha_fwd_kernel[grid](
            Q, K, V, O, LSE,
            S, H,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            LSE.stride(0), LSE.stride(1), LSE.stride(2),
            sm_scale,
            BLOCK_D=D
        )