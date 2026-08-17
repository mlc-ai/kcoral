import torch
import triton
import triton.language as tl


# Configures the global allocator required for rapid device-side tensor descriptor compilation
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64},  num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64},  num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64},  num_warps=8, num_stages=4),
    ],
    key=['S'],
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
    B, H, S,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)
    
    q_start = pid_m * BLOCK_M
    # Exit precisely if sequence mapping limits completely miss the target boundary
    if q_start >= S:
        return
        
    q_base = Q + pid_b * stride_qb + pid_h * stride_qh
    k_base = K + pid_b * stride_kb + pid_h * stride_kh
    v_base = V + pid_b * stride_vb + pid_h * stride_vh
    o_base = O + pid_b * stride_ob + pid_h * stride_oh
    lse_ptrs = LSE + pid_b * stride_lseb + pid_h * stride_lseh
    
    # Establish unified 2D descriptors configured natively for Blackwell bounds checking  
    q_desc = tl.make_tensor_descriptor(
        q_base, shape=[S, D], strides=[stride_qs, 1],
        block_shape=[BLOCK_M, D], padding_option="zero"
    )
    k_desc = tl.make_tensor_descriptor(
        k_base, shape=[S, D], strides=[stride_ks, 1],
        block_shape=[BLOCK_N, D], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        v_base, shape=[S, D], strides=[stride_vs, 1],
        block_shape=[BLOCK_N, D], padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        o_base, shape=[S, D], strides=[stride_os, 1],
        block_shape=[BLOCK_M, D], padding_option="zero"
    )
    
    # Fetch queries using the TMA layout rules mapping 
    q = q_desc.load([q_start, 0])
    
    # Initialized purely in math bounds escaping `float("-inf")` to eliminate conditional runtime ops
    m_i = tl.full((BLOCK_M,), -50000.0, tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, D), tl.float32)
    
    # Calculate unified scaling variables explicitly retaining fp32 formats inside inner loops 
    RCP_LN2 = 1.4426950408889634
    sm_scale_log2 = sm_scale * RCP_LN2
    
    offs_m = q_start + tl.arange(0, BLOCK_M)
    q_idx = offs_m[:, None]
    
    # Dense chunks bounded tightly prior to causal diagonals
    n_full_steps = q_start // BLOCK_N
    
    # Phase 1: Pure Math Intensive Fast-Path Stream Loop 
    for k_step in range(0, n_full_steps):
        k_start = k_step * BLOCK_N
        
        k = k_desc.load([k_start, 0])
        v = v_desc.load([k_start, 0])
        
        # Accumulate matrix ops in pure FP32 preventing precision degradation
        scores = tl.dot(q, k.T, out_dtype=tl.float32) * sm_scale_log2
        
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        
        alpha = tl.math.exp2(m_i - m_ij)
        p = tl.math.exp2(scores - m_ij[:, None])
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(q.dtype), v, acc=acc, out_dtype=tl.float32)
        m_i = m_ij
        
    # Process limited edge-cases requiring valid constraints bounds
    k_max = tl.minimum(S, q_start + BLOCK_M)
    n_total_steps = tl.cdiv(k_max, BLOCK_N)
    offs_n_base = tl.arange(0, BLOCK_N)
    
    # Phase 2: Causal Masking Stream Loop 
    for k_step in range(n_full_steps, n_total_steps):
        k_start = k_step * BLOCK_N
        
        k = k_desc.load([k_start, 0])
        v = v_desc.load([k_start, 0])
        
        scores = tl.dot(q, k.T, out_dtype=tl.float32) * sm_scale_log2
        
        # Invalidate purely using bound limit variables matching mathematical FP32 domains mapping
        k_idx = k_start + offs_n_base[None, :]
        valid_score = (q_idx >= k_idx)
        scores = tl.where(valid_score, scores, -50000.0)
        
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        
        alpha = tl.math.exp2(m_i - m_ij)
        p = tl.math.exp2(scores - m_ij[:, None])
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(q.dtype), v, acc=acc, out_dtype=tl.float32)
        m_i = m_ij
        
    # Efficient normalization division using inverse reciprocals avoiding `BLOCK_M` overhead bottlenecks
    inv_l_i = 1.0 / l_i
    out = acc * inv_l_i[:, None]
    
    # Scale result rigorously back to `e` limits
    LN2 = 0.6931471805599453
    lse = (m_i + tl.math.log2(l_i)) * LN2
    
    # Write components effectively relying on zero padding rules seamlessly managing limit sizes
    o_desc.store([q_start, 0], out.to(q.dtype))
    
    lse_offs = offs_m * stride_lses
    mask_lse = offs_m < S
    tl.store(lse_ptrs + lse_offs, lse, mask=mask_lse)


def run(Q, K, V, O, LSE):
    """
    Computes optimal native causal Multi-Head Attention forward.
    Q, K, V, O have shape (B, H, S, D) and type bfloat16.
    LSE has shape (B, H, S) and type float32.
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    sm_scale = 1.0 / (D ** 0.5)
    
    # Map cleanly to Blackwell hardware distribution L2 characteristics naturally mapped
    grid = lambda META: (triton.cdiv(S, META["BLOCK_M"]), B, H)
    
    _attn_fwd_kernel[grid](
        Q, K, V, sm_scale,
        O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        B, H, S,
        D=D
    )