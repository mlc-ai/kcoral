import torch
import triton
import triton.language as tl

def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
    ],
    key=['seq_len'],
)
@triton.jit
def _attn_fwd_kernel(
    Q, K, V, sm_scale,
    O, LSE,
    stride_qb, stride_qh, stride_qs,
    stride_kb, stride_kh, stride_ks,
    stride_vb, stride_vh, stride_vs,
    stride_ob, stride_oh, stride_os,
    stride_lseb, stride_lseh, stride_lses,
    seq_len, H, D,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    # Causal sequence length upper-bound optimization
    if pid_m * BLOCK_M >= seq_len:
        return
        
    b = pid_bh // H
    h = pid_bh % H
    
    # 1. Base pointers for this batch/head
    q_base = Q + b * stride_qb + h * stride_qh
    k_base = K + b * stride_kb + h * stride_kh
    v_base = V + b * stride_vb + h * stride_vh
    o_base = O + b * stride_ob + h * stride_oh
    
    # 2. Setup TMA Descriptors
    q_desc = tl.make_tensor_descriptor(
        q_base, shape=[seq_len, D], strides=[stride_qs, 1],
        block_shape=[BLOCK_M, BLOCK_D], padding_option="zero"
    )
    k_desc = tl.make_tensor_descriptor(
        k_base, shape=[seq_len, D], strides=[stride_ks, 1],
        block_shape=[BLOCK_N, BLOCK_D], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        v_base, shape=[seq_len, D], strides=[stride_vs, 1],
        block_shape=[BLOCK_N, BLOCK_D], padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        o_base, shape=[seq_len, D], strides=[stride_os, 1],
        block_shape=[BLOCK_M, BLOCK_D], padding_option="zero"
    )
    
    # Load Q block via TMA
    q = q_desc.load([pid_m * BLOCK_M, 0])
    
    # Initialize FlashAttention accumulator statistics
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    # Fast path: blocks fully prior to the causal boundary
    limit_n = (pid_m * BLOCK_M) // BLOCK_N * BLOCK_N
    
    for start_n in range(0, limit_n, BLOCK_N):
        k = k_desc.load([start_n, 0])
        v = v_desc.load([start_n, 0])
        
        qk = tl.dot(q, k.T, out_dtype=tl.float32) * sm_scale
        
        # No causal mask needed for fully unmasked prior blocks
        m_ij = tl.max(qk, 1)
        m_new = tl.maximum(m_i, m_ij)
        
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(qk - m_new[:, None])
        
        l_i = l_i * alpha + tl.sum(beta, 1)
        
        acc = acc * alpha[:, None]
        acc = tl.dot(beta.to(tl.bfloat16), v, acc)
        
        m_i = m_new

    # Causal path: blocks intersecting the causal diagonal
    causal_start = limit_n
    causal_end = (pid_m + 1) * BLOCK_M
    if seq_len < causal_end:
        causal_end = seq_len
        
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    
    for start_n in range(causal_start, causal_end, BLOCK_N):
        k = k_desc.load([start_n, 0])
        v = v_desc.load([start_n, 0])
        
        qk = tl.dot(q, k.T, out_dtype=tl.float32) * sm_scale
        
        # Apply strict causal masking
        offs_n_curr = start_n + offs_n
        causal_mask = offs_m[:, None] >= offs_n_curr[None, :]
        qk = tl.where(causal_mask, qk, float("-inf"))
        
        m_ij = tl.max(qk, 1)
        m_new = tl.maximum(m_i, m_ij)
        
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(qk - m_new[:, None])
        
        l_i = l_i * alpha + tl.sum(beta, 1)
        
        acc = acc * alpha[:, None]
        acc = tl.dot(beta.to(tl.bfloat16), v, acc)
        
        m_i = m_new

    # Normalize output and compute standard Log-Sum-Exp
    acc = acc / l_i[:, None]
    lse = m_i + tl.log(l_i)
    
    # Store matrix output via TMA descriptor
    o_desc.store([pid_m * BLOCK_M, 0], acc.to(tl.bfloat16))
    
    # Store LSE vector via standard pointers (1D sequence)
    lse_offset = b * stride_lseb + h * stride_lseh
    lse_ptrs = LSE + lse_offset + offs_m * stride_lses
    
    q_mask = offs_m < seq_len
    tl.store(lse_ptrs, lse, mask=q_mask)

def run(Q, K, V, O, LSE):
    """
    Computes causal Multi-Head Attention forward pass returning Output and Log-Sum-Exp.
    Hardware execution is accelerated using Tensor Memory Descriptors (TMA).
    """
    torch.cuda.set_device(Q.device)
    triton.set_allocator(alloc_fn)
    
    B, H, S, D = Q.shape
    if S == 0:
        return
    
    sm_scale = 1.0 / (D ** 0.5)
    
    grid = lambda META: (
        triton.cdiv(S, META["BLOCK_M"]),
        B * H,
        1
    )
    
    _attn_fwd_kernel[grid](
        Q, K, V, sm_scale,
        O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        S, H, D,
        BLOCK_D=128
    )