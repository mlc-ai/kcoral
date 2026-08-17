import torch
import triton
import triton.language as tl

# Allocator strictly for Triton infrastructure (device-side descriptors)
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
    ],
    key=['seq_len'],
)
@triton.jit
def _attn_fwd_kernel(
    Q, K, V, sm_scale_log2,
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
    
    # Early exit for programs entirely outside the valid sequence
    if pid_m * BLOCK_M >= seq_len:
        return
        
    b = pid_bh // H
    h = pid_bh % H
    
    # 1. Base pointers for this batch/head
    q_base = Q + b * stride_qb + h * stride_qh
    k_base = K + b * stride_kb + h * stride_kh
    v_base = V + b * stride_vb + h * stride_vh
    o_base = O + b * stride_ob + h * stride_oh
    
    # 2. Setup Device TMA Descriptors
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
    
    # 3. Load Q and deeply optimize scaled dot product
    # By pushing sm_scale and log2(e) into Q pre-scaling, 
    # we completely bypass 16k FMULs in the inner loop.
    q = q_desc.load([pid_m * BLOCK_M, 0])
    q = (q.to(tl.float32) * sm_scale_log2).to(tl.bfloat16)
    
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    # Fast path bounds: full blocks that sit strictly left of the causal diagonal
    limit_n = (pid_m * BLOCK_M) // BLOCK_N * BLOCK_N
    
    # 4. Phase 1 - Unmasked fully-valid keys (Highly software pipelined)
    for start_n in range(0, limit_n, BLOCK_N):
        k = k_desc.load([start_n, 0])
        v = v_desc.load([start_n, 0])
        
        # Native math mapping straight to Hopper/Blackwell MMA instruction layouts
        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        
        m_ij = tl.max(qk, 1)
        m_new = tl.maximum(m_i, m_ij)
        
        # Leveraging `exp2` explicitly avoiding multi-instruction `exp` routines
        alpha = tl.exp2(m_i - m_new)
        beta = tl.exp2(qk - m_new[:, None])
        
        l_i = l_i * alpha + tl.sum(beta, 1)
        
        # TMA scales cleanly using Tensor Memory scale-and-accumulate logic
        acc = acc * alpha[:, None]
        acc = tl.dot(beta.to(tl.bfloat16), v, acc)
        
        m_i = m_new

    # 5. Phase 2 - Causal path: blocks physically intersecting the boundary
    causal_start = limit_n
    causal_end = tl.minimum((pid_m + 1) * BLOCK_M, seq_len)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    
    for start_n in range(causal_start, causal_end, BLOCK_N):
        k = k_desc.load([start_n, 0])
        v = v_desc.load([start_n, 0])
        
        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        
        # Apply lower triangular causal masking bounds unconditionally
        offs_n_curr = start_n + offs_n
        causal_mask = offs_m[:, None] >= offs_n_curr[None, :]
        qk = tl.where(causal_mask, qk, float("-inf"))
        
        m_ij = tl.max(qk, 1)
        m_new = tl.maximum(m_i, m_ij)
        
        alpha = tl.exp2(m_i - m_new)
        beta = tl.exp2(qk - m_new[:, None])
        
        l_i = l_i * alpha + tl.sum(beta, 1)
        
        acc = acc * alpha[:, None]
        acc = tl.dot(beta.to(tl.bfloat16), v, acc)
        
        m_i = m_new

    # 6. Accumulator normalization & LSE calculation mapping bounds back to base-e
    inv_l_i = 1.0 / l_i
    acc = acc * inv_l_i[:, None]
    
    ln_2 = 0.6931471805599453
    lse = (m_i + tl.log2(l_i)) * ln_2
    
    # Store matrix output natively via TMA bypassing boundary math masks
    o_desc.store([pid_m * BLOCK_M, 0], acc.to(tl.bfloat16))
    
    # Store LSE vector dynamically checking bounds to suppress output garbage
    lse_offset = b * stride_lseb + h * stride_lseh
    lse_ptrs = LSE + lse_offset + offs_m * stride_lses
    
    q_mask = offs_m < seq_len
    tl.store(lse_ptrs, lse, mask=q_mask)


def run(Q, K, V, O, LSE):
    """
    Dest-passing causal Multi-Head Attention forward with Output & Log-Sum-Exp.
    Strictly relies on Triton 3.7 descriptor arrays natively bridging TMA hardware 
    across Blackwell and efficiently caching overlapping Head sequences in L2.
    """
    torch.cuda.set_device(Q.device)
    triton.set_allocator(alloc_fn)
    
    B, H, S, D = Q.shape
    if S == 0:
        return
    
    # Optimize scale factor to natively embed `log2(e)` to circumvent internal
    # FP32 bounds conversion routines enabling 1 instruction EX2 loop processing
    sm_scale = 1.0 / (D ** 0.5)
    log2_e = 1.4426950408889634
    scale_log2 = sm_scale * log2_e
    
    grid = lambda META: (
        triton.cdiv(S, META["BLOCK_M"]),
        B * H,
        1
    )
    
    _attn_fwd_kernel[grid](
        Q, K, V, scale_log2,
        O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        S, H, D,
        BLOCK_D=128
    )