import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=5),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=3),
    ],
    key=["S"],
)
@triton.jit
def _attn_fwd_kernel(
    Q, K, V, O, LSE,
    sm_scale_log2,
    S,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    H,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Determine the query sequence tile and batch/head index
    start_m = tl.program_id(0)
    batch_head = tl.program_id(1)
    
    batch_idx = batch_head // H
    head_idx = batch_head % H
    
    # Early exit if the entire sequence tile is out of bounds
    if start_m * BLOCK_M >= S:
        return

    # Base pointers for this specific batch and head
    q_base = Q + batch_idx * stride_qb + head_idx * stride_qh
    k_base = K + batch_idx * stride_kb + head_idx * stride_kh
    v_base = V + batch_idx * stride_vb + head_idx * stride_vh
    o_base = O + batch_idx * stride_ob + head_idx * stride_oh
    lse_base = LSE + batch_idx * stride_lseb + head_idx * stride_lseh
    
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    
    q_ptrs = q_base + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    q_mask = offs_m < S
    
    # Load Q and scale it outside the inner loop to save math operations.
    # Q remains safe from NaNs due to out-of-bounds because padded values will be 0.
    q = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)
    q = (q * sm_scale_log2).to(tl.bfloat16)
    
    # Initialize accumulators
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    
    # Setup K and V pointers for the first block
    # By advancing pointers directly inside the loop, we avoid heavy address calculations
    k_ptrs = k_base + tl.arange(0, BLOCK_N)[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = v_base + tl.arange(0, BLOCK_N)[:, None] * stride_vs + offs_d[None, :] * stride_vd
    
    # Pre-calculate sequence boundaries for the different phases
    max_n = tl.minimum((start_m + 1) * BLOCK_M, S)
    num_full_blocks = tl.minimum(S // BLOCK_N, start_m * BLOCK_M // BLOCK_N)
    end_n_blocks = (max_n + BLOCK_N - 1) // BLOCK_N
    
    # Phase 1: Fully unmasked sequence blocks
    # K and V sequence blocks are guaranteed entirely in-bounds and without causal constraints
    for _ in range(0, num_full_blocks):
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)
        
        qk = tl.dot(q, k.T)
        
        m_i_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp2(m_i - m_i_new)
        p = tl.exp2(qk - m_i_new[:, None])
        
        l_i_new = alpha * l_i + tl.sum(p, axis=1)
        
        acc = acc * alpha[:, None]
        p_bf16 = p.to(tl.bfloat16)
        acc = tl.dot(p_bf16, v, acc=acc)
        
        m_i = m_i_new
        l_i = l_i_new
        
        # Advance pointers to next K/V sequence block safely
        k_ptrs += BLOCK_N * stride_ks
        v_ptrs += BLOCK_N * stride_vs

    # Phase 2: Causal boundary blocks
    # Require masking for causality, but dynamically skip sequence bounds masking when fully within bounds
    for start_n in range(num_full_blocks, end_n_blocks):
        offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
        
        # Optimize: skip sequence boundary masks if this block is fully within the overall sequence S
        if start_n * BLOCK_N + BLOCK_N <= S:
            k = tl.load(k_ptrs)
            v = tl.load(v_ptrs)
            qk = tl.dot(q, k.T)
            valid_mask = offs_m[:, None] >= offs_n[None, :]
            qk = tl.where(valid_mask, qk, float("-inf"))
        else:
            k_mask = offs_n < S
            k = tl.load(k_ptrs, mask=k_mask[:, None], other=0.0)
            v = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0)
            qk = tl.dot(q, k.T)
            valid_mask = (offs_m[:, None] >= offs_n[None, :]) & k_mask[None, :]
            qk = tl.where(valid_mask, qk, float("-inf"))
        
        m_i_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp2(m_i - m_i_new)
        p = tl.exp2(qk - m_i_new[:, None])
        
        l_i_new = alpha * l_i + tl.sum(p, axis=1)
        
        acc = acc * alpha[:, None]
        p_bf16 = p.to(tl.bfloat16)
        acc = tl.dot(p_bf16, v, acc=acc)
        
        m_i = m_i_new
        l_i = l_i_new
        
        k_ptrs += BLOCK_N * stride_ks
        v_ptrs += BLOCK_N * stride_vs

    # Final Softmax Normalization
    acc = acc / l_i[:, None]
    
    # Store O to preallocated tensor cleanly respecting query out-of-bounds masks
    o_ptrs = o_base + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(tl.bfloat16), mask=q_mask[:, None])
    
    # Convert LSE from optimized base-2 mapping back to standard natural log scale
    ln2 = 0.6931471805599453
    lse = (m_i + tl.log2(l_i)) * ln2
    
    lse_ptrs = lse_base + offs_m * stride_lses
    tl.store(lse_ptrs, lse, mask=q_mask)

def run(Q, K, V, O, LSE):
    """
    Computes Standard Causal Multi-Head Attention using highly optimized Triton FlashAttention-v2 logic
    tailored specifically for Hopper H100 SM90 features avoiding CuDNN overhead.
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    
    # Precompute scaling alongside exp2 mapping transformation coefficient
    sm_scale = 1.0 / (D ** 0.5)
    log2_e = 1.4426950408889634
    sm_scale_log2 = sm_scale * log2_e
    
    # Standard grid implicitly handles L2 locality grouping automatically
    grid = lambda META: (triton.cdiv(S, META["BLOCK_M"]), B * H)
    
    _attn_fwd_kernel[grid](
        Q, K, V, O, LSE,
        sm_scale_log2,
        S,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        H,
        D=D,
    )