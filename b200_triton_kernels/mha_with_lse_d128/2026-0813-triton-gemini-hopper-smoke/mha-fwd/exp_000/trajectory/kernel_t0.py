import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
    ],
    key=['S'],
)
@triton.jit
def mha_fwd_kernel(
    Q, K, V, O, LSE,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    B, H, S,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    b_id = pid_bh // H
    h_id = pid_bh % H
    
    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    
    # Offsets and Pointers
    q_ptrs = Q + b_id * stride_qb + h_id * stride_qh + (offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd)
    k_ptrs = K + b_id * stride_kb + h_id * stride_kh + (offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd)
    v_ptrs = V + b_id * stride_vb + h_id * stride_vh + (offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd)
    
    mask_m = offs_m < S
    mask_q = mask_m[:, None] & (offs_d[None, :] < BLOCK_D)
    
    # Load Query
    q = tl.load(q_ptrs, mask=mask_q, other=0.0)
    
    # Initialize Running Stats and Accumulator
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    num_n_blocks = tl.cdiv(S, BLOCK_N)
    
    for n_idx in range(num_n_blocks):
        start_n = n_idx * BLOCK_N
        curr_offs_n = start_n + offs_n
        mask_n = curr_offs_n < S
        mask_kv = mask_n[:, None] & (offs_d[None, :] < BLOCK_D)
        
        # Load Key and Value
        k = tl.load(k_ptrs + start_n * stride_ks, mask=mask_kv, other=0.0)
        v = tl.load(v_ptrs + start_n * stride_vs, mask=mask_kv, other=0.0)
        
        # Compute Dot Product Q @ K^T
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, k.T, acc=qk)
        qk = qk * scale
        
        # Apply Causal/Sequence Mask (Non-causal padding masking)
        qk = tl.where(mask_n[None, :], qk, -float('inf'))
        
        # Compute Online Softmax
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        
        # Update running max and scaling factor
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        
        # Update Accumulator
        acc = acc * alpha[:, None]
        p_bf16 = p.to(tl.bfloat16)
        acc = tl.dot(p_bf16, v, acc=acc)
        
        m_i = m_ij

    # Final Softmax Normalization
    acc = acc / l_i[:, None]
    
    # LogSumExp
    lse = m_i + tl.log(l_i)
    
    # Output Pointers
    o_ptrs = O + b_id * stride_ob + h_id * stride_oh + (offs_m[:, None] * stride_os + offs_d[None, :] * stride_od)
    lse_ptrs = LSE + b_id * stride_lseb + h_id * stride_lseh + offs_m * stride_lses
    
    # Store Output and LSE
    tl.store(o_ptrs, acc.to(tl.bfloat16), mask=mask_q)
    tl.store(lse_ptrs, lse, mask=mask_m)


def run(Q, K, V, O, LSE):
    """
    Computes Scaled Dot-Product Attention in a FlashAttention-like non-causal way.
    Writes outputs purely destination-passing.
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    scale = 1.0 / (D ** 0.5)
    
    grid = lambda META: (triton.cdiv(S, META['BLOCK_M']), B * H, 1)
    
    mha_fwd_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        B, H, S,
        scale,
        BLOCK_D=128,
    )