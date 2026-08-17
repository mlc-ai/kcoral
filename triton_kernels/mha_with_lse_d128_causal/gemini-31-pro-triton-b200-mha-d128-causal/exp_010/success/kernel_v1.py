import torch
import triton
import triton.language as tl


def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'STAGE_LOOP': 2}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'STAGE_LOOP': 3}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'STAGE_LOOP': 3}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'STAGE_LOOP': 2}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'STAGE_LOOP': 3}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'STAGE_LOOP': 3}, num_warps=4, num_stages=4),
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
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    STAGE_LOOP: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)
    
    q_start = pid_m * BLOCK_M
    # Early exit if the query block starts strictly beyond Sequence length
    if q_start >= S:
        return
        
    # Apply memory offset for isolated (batch, head) contiguous subspace
    q_base = Q + pid_b * stride_qb + pid_h * stride_qh
    k_base = K + pid_b * stride_kb + pid_h * stride_kh
    v_base = V + pid_b * stride_vb + pid_h * stride_vh
    o_base = O + pid_b * stride_ob + pid_h * stride_oh
    lse_ptrs = LSE + pid_b * stride_lseb + pid_h * stride_lseh
    
    # Establish device TMA descriptors assuming standard layout
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
    
    # TMA Load queries
    q = q_desc.load([q_start, 0])
    
    # Running online softmax state initialized in fp32
    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, D), tl.float32)
    
    # Precompute scale mapped natively to base-2
    RCP_LN2 = 1.4426950408889634
    sm_scale_log2 = sm_scale * RCP_LN2
    
    # Restrict K iteration space directly onto the Causal Mask boundary 
    k_max = tl.minimum(S, q_start + BLOCK_M)
    n_steps = tl.cdiv(k_max, BLOCK_N)
    
    offs_m = q_start + tl.arange(0, BLOCK_M)
    offs_n_base = tl.arange(0, BLOCK_N)
    
    q_idx = offs_m[:, None]
    q_mask = q_idx < S
    
    # Main Streaming Loop utilizing async TMA load overlap via software pipeline staging
    for k_step in tl.range(0, n_steps, num_stages=STAGE_LOOP):
        k_start = k_step * BLOCK_N
        
        k = k_desc.load([k_start, 0])
        v = v_desc.load([k_start, 0])
        
        scores = tl.dot(q, k.T, out_dtype=tl.float32) * sm_scale_log2
        
        # Valid causal mask application
        k_idx = k_start + offs_n_base[None, :]
        valid_score = (q_idx >= k_idx) & q_mask
        scores = tl.where(valid_score, scores, -float("inf"))
        
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        safe_m_ij = tl.where(m_ij == -float("inf"), 0.0, m_ij)
        
        alpha = tl.math.exp2(m_i - safe_m_ij)
        p = tl.math.exp2(scores - safe_m_ij[:, None])
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(q.dtype), v, acc=acc, out_dtype=tl.float32)
        m_i = m_ij
        
    # Scale result by norm denom
    safe_l_i = tl.where(l_i == 0.0, 1.0, l_i)
    out = acc / safe_l_i[:, None]
    
    # Convert LogSumExp state back rigorously to the designated Natural Log domain
    LN2 = 0.6931471805599453
    lse = tl.where(l_i == 0.0, -float("inf"), (m_i + tl.math.log2(safe_l_i)) * LN2)
    
    # TMA Store outputs (Auto-masks completely correctly using padded metadata constraints!)
    o_desc.store([q_start, 0], out.to(q.dtype))
    
    # Directly map store for the 1D LSE components
    lse_offs = offs_m * stride_lses
    mask_lse = offs_m < S
    tl.store(lse_ptrs + lse_offs, lse, mask=mask_lse)


def run(Q, K, V, O, LSE):
    """
    Computes causal Multi-Head Attention forward via destination-passing.
    Q, K, V, O have shape (B, H, S, D) and type bfloat16.
    LSE has shape (B, H, S) and type float32.
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    sm_scale = 1.0 / (D ** 0.5)
    
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