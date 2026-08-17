import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
    ],
    key=['S'],
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
    D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)
    
    q_start = pid_m * BLOCK_M
    # Early exit for fully out-of-bounds query tiles
    if q_start >= S:
        return
        
    # Offset base pointers for the specific batch and head
    q_ptrs = Q + pid_b * stride_qb + pid_h * stride_qh
    k_ptrs = K + pid_b * stride_kb + pid_h * stride_kh
    v_ptrs = V + pid_b * stride_vb + pid_h * stride_vh
    o_ptrs = O + pid_b * stride_ob + pid_h * stride_oh
    lse_ptrs = LSE + pid_b * stride_lseb + pid_h * stride_lseh
    
    offs_m = q_start + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    
    # Load Q tile
    q_offs = offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    mask_q = offs_m[:, None] < S
    q = tl.load(q_ptrs + q_offs, mask=mask_q, other=0.0)
    
    # Running online softmax state
    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, D), tl.float32)
    
    # Precompute fast scale for log2 reductions
    RCP_LN2 = 1.4426950408889634
    sm_scale_log2 = sm_scale * RCP_LN2
    
    # Causal sequence length cap for this block of queries
    k_max = tl.minimum(S, q_start + BLOCK_M)
    n_steps = tl.cdiv(k_max, BLOCK_N)
    
    for k_step in range(0, n_steps):
        k_start = k_step * BLOCK_N
        
        # Load K transposed (D, BLOCK_N) directly to avoid transposing in SMEM
        k_offs = offs_d[:, None] * stride_kd + (k_start + offs_n[None, :]) * stride_ks
        mask_k = (k_start + offs_n[None, :]) < k_max
        k = tl.load(k_ptrs + k_offs, mask=mask_k, other=0.0)
        
        # Load V (BLOCK_N, D)
        v_offs = (k_start + offs_n[:, None]) * stride_vs + offs_d[None, :] * stride_vd
        mask_v = (k_start + offs_n[:, None]) < k_max
        v = tl.load(v_ptrs + v_offs, mask=mask_v, other=0.0)
        
        scores = tl.dot(q, k) * sm_scale_log2
        
        # Causal mask processing
        k_idx = k_start + offs_n[None, :]
        q_idx = offs_m[:, None]
        valid_score = (q_idx >= k_idx)
        scores = tl.where(valid_score, scores, -float("inf"))
        
        # Running max and normalization
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        safe_m_ij = tl.where(m_ij == -float("inf"), 0.0, m_ij)
        
        alpha = tl.math.exp2(m_i - safe_m_ij)
        p = tl.math.exp2(scores - safe_m_ij[:, None])
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(q.dtype), v, acc)
        m_i = m_ij
        
    safe_l_i = tl.where(l_i == 0.0, 1.0, l_i)
    out = acc / safe_l_i[:, None]
    
    # Convert LogSumExp state back to natural log domain (base e)
    LN2 = 0.6931471805599453
    lse = tl.where(l_i == 0.0, -float("inf"), (m_i + tl.math.log2(safe_l_i)) * LN2)
    
    # Write Final Output
    o_offs = offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(o_ptrs + o_offs, out.to(O.dtype.element_ty), mask=mask_q)
    
    # Write LSE
    lse_offs = offs_m * stride_lses
    mask_lse = offs_m < S
    tl.store(lse_ptrs + lse_offs, lse, mask=mask_lse)


def run(Q, K, V, O, LSE):
    """
    Computes causal Multi-Head Attention forward.
    Q, K, V, O have shape (B, H, S, D) and type bfloat16.
    LSE has shape (B, H, S) and type float32.
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    sm_scale = 1.0 / (D ** 0.5)
    
    # Map a 1D grid of query block indices alongside batch and head dimensions
    grid = lambda META: (triton.cdiv(S, META["BLOCK_M"]), B, H)
    
    _attn_fwd_kernel[grid](
        Q, K, V, sm_scale,
        O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        B, H, S,
        D=D
    )