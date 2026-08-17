import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
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
    
    # Early exit if the entire block is out of sequence bounds
    if start_m >= S_len:
        return
        
    is_m_full = (start_m + BLOCK_M) <= S_len
    offs_m = start_m + tl.arange(0, BLOCK_M)
    m_mask = offs_m < S_len
    m_mask_dim1 = m_mask[:, None]
    
    # Base pointers for the current batch and head
    q_base = q + batch_idx * stride_qb + head_idx * stride_qh
    k_base = k + batch_idx * stride_kb + head_idx * stride_kh
    v_base = v + batch_idx * stride_vb + head_idx * stride_vh
    
    offs_d = tl.arange(0, BLOCK_D)
    q_ptrs = q_base + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    
    # Load Q block once
    if is_m_full:
        q_val = tl.load(q_ptrs)
    else:
        q_val = tl.load(q_ptrs, mask=m_mask_dim1, other=0.0)
        
    # Numerically stable softmax trackers. 
    # Masked-out rows get m_i = 0.0 to prevent propagating NaNs to the accumulator.
    m_i = tl.where(m_mask, float("-inf"), 0.0).to(tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    # End index for keys due to causal masking
    loop_end = start_m + BLOCK_M
    if S_len < loop_end:
        loop_end = S_len
        
    num_steps = tl.cdiv(loop_end, BLOCK_N)
    offs_n = tl.arange(0, BLOCK_N)
    
    k_ptrs = k_base + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = v_base + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
    
    # Pipelined loop over keys and values
    for step in range(0, num_steps):
        start_n = step * BLOCK_N
        current_n = start_n + offs_n
        
        n_mask = current_n < S_len
        n_mask_dim1 = n_mask[:, None]
        
        # Load K and V
        k_val = tl.load(k_ptrs, mask=n_mask_dim1, other=0.0)
        v_val = tl.load(v_ptrs, mask=n_mask_dim1, other=0.0)
        
        # Q @ K^T
        qk = tl.dot(q_val, tl.trans(k_val), out_dtype=tl.float32)
        qk = qk * sm_scale
        
        # Determine valid mask for the dot products
        mask = n_mask[None, :]
        if not is_m_full:
            mask = mask & m_mask_dim1
            
        is_causal = (start_n + BLOCK_N) > start_m
        if is_causal:
            causal_mask = offs_m[:, None] >= current_n[None, :]
            mask = mask & causal_mask
            
        qk = tl.where(mask, qk, float("-inf"))
            
        # Online softmax logic
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        
        alpha = tl.exp(m_i - m_ij)
        acc = acc * alpha[:, None]
        
        # Context accumulation (P @ V)
        p_bf16 = p.to(v_val.dtype)
        acc = tl.dot(p_bf16, v_val, acc)
        
        m_i = m_ij
        l_i = l_i * alpha + l_ij
        
        # Advance pointers explicitly for hardware unrolling
        k_ptrs += BLOCK_N * stride_ks
        v_ptrs += BLOCK_N * stride_vs
        
    # Final normalization
    l_i_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_i_safe[:, None]
    
    # Output offsets
    o_base = o + batch_idx * stride_ob + head_idx * stride_oh
    o_ptrs = o_base + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    
    lse_base = lse + batch_idx * stride_lseb + head_idx * stride_lseh
    lse_ptrs = lse_base + offs_m * stride_lses
    lse_val = m_i + tl.log(l_i_safe)
    
    # Store outputs safely
    if is_m_full:
        tl.store(o_ptrs, acc.to(q_val.dtype))
        tl.store(lse_ptrs, lse_val)
    else:
        tl.store(o_ptrs, acc.to(q_val.dtype), mask=m_mask_dim1)
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
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            LSE.stride(0), LSE.stride(1), LSE.stride(2),
            sm_scale,
            BLOCK_D=D
        )