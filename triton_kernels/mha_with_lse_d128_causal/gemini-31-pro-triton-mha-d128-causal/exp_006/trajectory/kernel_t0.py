import math
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
    ],
    key=['S']
)
@triton.jit
def _causal_fwd_kernel(
    Q, K, V, O, LSE,
    sm_scale, S, H,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr
):
    start_m = tl.program_id(0)
    batch_head = tl.program_id(1)
    
    batch_idx = batch_head // H
    head_idx = batch_head % H
    
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    q_ptrs = Q + batch_idx * stride_qb + head_idx * stride_qh + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    k_ptrs = K + batch_idx * stride_kb + head_idx * stride_kh + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = V + batch_idx * stride_vb + head_idx * stride_vh + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
    
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    
    q_mask = offs_m[:, None] < S
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)
    
    # Restrict K/V iteration based on causal causality
    max_k = tl.minimum(S, (start_m + 1) * BLOCK_M)
    num_k_blocks = tl.cdiv(max_k, BLOCK_N)
    
    for start_n in range(0, num_k_blocks):
        start_n_idx = start_n * BLOCK_N
        offs_n_curr = start_n_idx + offs_n
        
        k_mask = offs_n_curr[:, None] < S
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)
        
        # Calculate QK^T
        qk = tl.dot(q, k.trans(1, 0), out_dtype=tl.float32)
        qk = qk * sm_scale
        
        # Apply causal and sequence bounds masking
        causal_mask = offs_m[:, None] >= offs_n_curr[None, :]
        seq_mask = offs_n_curr[None, :] < S
        mask = causal_mask & seq_mask
        qk = tl.where(mask, qk, float("-inf"))
        
        # Numerically stable softmax incrementally 
        m_ij = tl.max(qk, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])
        
        l_i_new = alpha * l_i + tl.sum(p, 1)
        
        # Scale the accumulator
        acc = acc * alpha[:, None]
        
        # Load V and aggregate
        v = tl.load(v_ptrs, mask=k_mask, other=0.0)
        p_cast = p.to(v.dtype)
        acc = tl.dot(p_cast, v, acc, out_dtype=tl.float32)
        
        # Update running states and pointers
        m_i = m_i_new
        l_i = l_i_new
        
        k_ptrs += BLOCK_N * stride_ks
        v_ptrs += BLOCK_N * stride_vs

    # Normalize attention output by L
    acc = acc / l_i[:, None]

    # Store finalized O tensor
    o_ptrs = O + batch_idx * stride_ob + head_idx * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    o_mask = offs_m[:, None] < S
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=o_mask)

    # Compute and store Log-Sum-Exp
    lse = m_i + tl.log(l_i)
    lse_ptrs = LSE + batch_idx * stride_lseb + head_idx * stride_lseh + offs_m * stride_lses
    lse_mask = offs_m < S
    tl.store(lse_ptrs, lse, mask=lse_mask)


def run(Q, K, V, O, LSE):
    """
    Computes causal Multi-Head Attention via FlashAttention forward pass directly.
    Inputs and outputs strictly match definition shapes.
    """
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    sm_scale = 1.0 / math.sqrt(D)

    grid = lambda META: (
        triton.cdiv(S, META['BLOCK_M']),
        B * H
    )

    _causal_fwd_kernel[grid](
        Q, K, V, O, LSE,
        sm_scale, S, H,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        BLOCK_DMODEL=128
    )