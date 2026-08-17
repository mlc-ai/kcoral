import torch
import triton
import triton.language as tl
import math

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=3, num_warps=8),
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
    
    batch_idx = batch_head // H
    head_idx = batch_head % H
    
    q_offset = batch_idx * stride_qb + head_idx * stride_qh
    k_offset = batch_idx * stride_kb + head_idx * stride_kh
    v_offset = batch_idx * stride_vb + head_idx * stride_vh
    o_offset = batch_idx * stride_ob + head_idx * stride_oh
    lse_offset = batch_idx * stride_lseb + head_idx * stride_lseh
    
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    
    m_mask = offs_m < S
    
    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    o_ptrs = O + o_offset + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    lse_ptrs = LSE + lse_offset + offs_m * stride_lses
    
    q = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0)
    
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    # Calculate the max range of sequence to support causal constraint
    n_start_max = (start_m + 1) * BLOCK_M
    if n_start_max > S:
        n_start_max = S
    
    num_n_blocks = tl.cdiv(n_start_max, BLOCK_N)
    
    for start_n in range(0, num_n_blocks):
        offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < S
        
        k_ptrs = K + k_offset + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        v_ptrs = V + v_offset + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
        
        k = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)
        
        qk = tl.dot(q, k.T, out_dtype=tl.float32) * sm_scale
        
        # Apply bounds and causal mask where query_pos < key_pos limits access
        is_valid_elem = n_mask[None, :] & (offs_m[:, None] >= offs_n[None, :])
        qk = tl.where(is_valid_elem, qk, float("-inf"))
        # To avoid NaN propagations during softmax scaling for padded/masked outer sequence elements
        qk = tl.where(m_mask[:, None], qk, 0.0)
        
        m_i_new = tl.maximum(m_i, tl.max(qk, axis=1))
        
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        
        p_bf16 = tl.cast(p, tl.bfloat16)
        
        acc = acc * alpha[:, None] + tl.dot(p_bf16, v, out_dtype=tl.float32)
        
        m_i = m_i_new

    # Finalize probabilities and LSE mapping
    acc = acc / l_i[:, None]
    lse = m_i + tl.log(l_i)
    
    tl.store(o_ptrs, tl.cast(acc, tl.bfloat16), mask=m_mask[:, None])
    tl.store(lse_ptrs, lse, mask=m_mask)

def run(Q, K, V, O, LSE):
    """
    Computes causal multi-head attention forward and natural-log-sum-exp metrics.
    Results are mapped onto the preallocated destination tensors `O` and `LSE`.
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    
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