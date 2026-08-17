import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=5),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=5),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
    ],
    key=['S_len']
)
@triton.jit
def _causal_mha_fwd_kernel(
    q, k, v, o, lse,
    S_len, H,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
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
    
    # Early exit if the entire block is strictly out-of-bounds
    if start_m >= S_len:
        return
        
    # Q block layout (Standard)
    q_base = q + batch_idx * stride_qb + head_idx * stride_qh
    q_block = tl.make_block_ptr(
        base=q_base,
        shape=(S_len, BLOCK_D),
        strides=(stride_qs, stride_qd),
        offsets=(start_m, 0),
        block_shape=(BLOCK_M, BLOCK_D),
        order=(1, 0)
    )
    
    # K block layout (Logically Transposed to enable native WGMMA column-major right operands)
    k_base = k + batch_idx * stride_kb + head_idx * stride_kh
    k_block = tl.make_block_ptr(
        base=k_base,
        shape=(BLOCK_D, S_len),
        strides=(stride_kd, stride_ks),
        offsets=(0, 0),
        block_shape=(BLOCK_D, BLOCK_N),
        order=(0, 1)
    )
    
    # V block layout (Standard)
    v_base = v + batch_idx * stride_vb + head_idx * stride_vh
    v_block = tl.make_block_ptr(
        base=v_base,
        shape=(S_len, BLOCK_D),
        strides=(stride_vs, stride_vd),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_D),
        order=(1, 0)
    )
    
    # Q loaded directly (bounds checking gracefully handles tail block padding to zeroes)
    q_val = tl.load(q_block, boundary_check=(0, 1), padding_option="zero")
    
    # Numerically stable trackers initialization
    offs_m = start_m + tl.arange(0, BLOCK_M)
    m_mask = offs_m < S_len
    
    m_i = tl.where(m_mask, float("-inf"), 0.0).to(tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    # Determine limits separating unmasked safe iterations vs masked boundary iterations
    unmasked_end = (start_m // BLOCK_N) * BLOCK_N
    
    # -----------------------------------------------------------
    # Phase 1: Fully Unmasked Loop
    # Processes Key blocks strictly before the Query block starts.
    # No causal masks or bounds checks are necessary internally.
    # -----------------------------------------------------------
    for start_n in range(0, unmasked_end, BLOCK_N):
        k_val = tl.load(k_block, boundary_check=(0, 1), padding_option="zero")
        v_val = tl.load(v_block, boundary_check=(0, 1), padding_option="zero")
        
        # Scaling happens post-dot in FP32 precision to prevent compounding BF16 rounding errors
        qk = tl.dot(q_val, k_val, out_dtype=tl.float32)
        qk = qk * sm_scale
        
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        
        alpha = tl.exp(m_i - m_ij)
        acc = acc * alpha[:, None]
        
        p_bf16 = p.to(v_val.dtype)
        acc = tl.dot(p_bf16, v_val, acc)
        
        m_i = m_ij
        l_i = l_i * alpha + l_ij
        
        k_block = tl.advance(k_block, (0, BLOCK_N))
        v_block = tl.advance(v_block, (BLOCK_N, 0))
        
    # -----------------------------------------------------------
    # Phase 2: Causal and Bounds-Masked Loop
    # Processes the remaining segment intersecting the diagonal block.
    # -----------------------------------------------------------
    masked_end = start_m + BLOCK_M
    if S_len < masked_end:
        masked_end = S_len
    masked_end_aligned = ((masked_end + BLOCK_N - 1) // BLOCK_N) * BLOCK_N
    
    for start_n in range(unmasked_end, masked_end_aligned, BLOCK_N):
        k_val = tl.load(k_block, boundary_check=(0, 1), padding_option="zero")
        v_val = tl.load(v_block, boundary_check=(0, 1), padding_option="zero")
        
        qk = tl.dot(q_val, k_val, out_dtype=tl.float32)
        qk = qk * sm_scale
        
        offs_n = start_n + tl.arange(0, BLOCK_N)
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        n_mask = offs_n[None, :] < S_len
        mask = causal_mask & n_mask
        
        qk = tl.where(mask, qk, float("-inf"))
        
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        
        alpha = tl.exp(m_i - m_ij)
        acc = acc * alpha[:, None]
        
        p_bf16 = p.to(v_val.dtype)
        acc = tl.dot(p_bf16, v_val, acc)
        
        m_i = m_ij
        l_i = l_i * alpha + l_ij
        
        k_block = tl.advance(k_block, (0, BLOCK_N))
        v_block = tl.advance(v_block, (BLOCK_N, 0))
        
    # Normalize final context embeddings
    l_i_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_i_safe[:, None]
    
    # Store Output using block pointer bounds handling
    o_base = o + batch_idx * stride_ob + head_idx * stride_oh
    o_block = tl.make_block_ptr(
        base=o_base,
        shape=(S_len, BLOCK_D),
        strides=(stride_os, stride_od),
        offsets=(start_m, 0),
        block_shape=(BLOCK_M, BLOCK_D),
        order=(1, 0)
    )
    tl.store(o_block, acc.to(o.dtype.element_ty), boundary_check=(0, 1))
    
    # Store LSE scalars properly bounded
    lse_base = lse + batch_idx * stride_lseb + head_idx * stride_lseh
    lse_ptrs = lse_base + offs_m * stride_lses
    lse_val = m_i + tl.log(l_i_safe)
    tl.store(lse_ptrs, lse_val, mask=m_mask)


def run(Q, K, V, O, LSE):
    """
    Computes causal Multi-Head Attention forward pass and returns Output and Log-Sum-Exp.
    Writes outputs directly into preallocated `O` and `LSE` tensors following the destination-passing contract.
    """
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    sm_scale = 1.0 / (D ** 0.5)
    
    if S > 0:
        grid = lambda META: (triton.cdiv(S, META['BLOCK_M']), B * H)
        
        _causal_mha_fwd_kernel[grid](
            Q, K, V, O, LSE,
            S, H,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            LSE.stride(0), LSE.stride(1), LSE.stride(2),
            sm_scale,
            BLOCK_D=D
        )