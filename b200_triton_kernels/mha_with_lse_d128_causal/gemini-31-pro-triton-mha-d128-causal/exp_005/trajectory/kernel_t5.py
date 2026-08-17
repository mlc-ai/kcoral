import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=5),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=5),
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
    
    # Early exit if the entire block is strictly out-of-bounds
    if start_m >= S_len:
        return
        
    is_m_full = (start_m + BLOCK_M) <= S_len
    offs_m = start_m + tl.arange(0, BLOCK_M)
    m_mask = offs_m < S_len
    
    # Base block pointers for current batch and head
    q_base = q + batch_idx * stride_qb + head_idx * stride_qh
    k_base = k + batch_idx * stride_kb + head_idx * stride_kh
    v_base = v + batch_idx * stride_vb + head_idx * stride_vh
    
    offs_d = tl.arange(0, BLOCK_D)
    q_ptrs = q_base + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    
    # Load Q explicitly outside the loop. Unpredicated if fully within bounds.
    if is_m_full:
        q_val = tl.load(q_ptrs)
    else:
        q_val = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0)
        
    # Scale Q upfront to save a multiply instruction per iteration in the inner loops
    q_val = (q_val * sm_scale).to(q_val.dtype)
        
    # Numerically stable softmax initializations.
    # We initialize masked-out rows with m_i = 0.0 to prevent computing exp(-inf - (-inf)) -> NaN.
    # Since these invalid rows also compute qk = -inf later, p = 0.0 and they won't corrupt acc.
    m_i = tl.where(m_mask, float("-inf"), 0.0).to(tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    # Configure unrolled loop starting pointers
    k_ptrs = k_base + tl.arange(0, BLOCK_N)[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = v_base + tl.arange(0, BLOCK_N)[:, None] * stride_vs + offs_d[None, :] * stride_vd
    
    # Phase 1: Fully unmasked WGMMA loop.
    # Processes Key blocks strictly before the Query block starts.
    # No causal masks or bounds checks needed inside this loop.
    unmasked_end = (start_m // BLOCK_N) * BLOCK_N
    for start_n in range(0, unmasked_end, BLOCK_N):
        k_val = tl.load(k_ptrs)
        v_val = tl.load(v_ptrs)
        
        qk = tl.dot(q_val, k_val.T, out_dtype=tl.float32)
        
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        
        alpha = tl.exp(m_i - m_ij)
        acc = acc * alpha[:, None]
        
        p_bf16 = p.to(v_val.dtype)
        acc = tl.dot(p_bf16, v_val, acc)
        
        m_i = m_ij
        l_i = l_i * alpha + l_ij
        
        k_ptrs += BLOCK_N * stride_ks
        v_ptrs += BLOCK_N * stride_vs
        
    # Phase 2: Causal and bounds-masked loop.
    # Process blocks covering the query segment up to the sequence limit.
    masked_end = start_m + BLOCK_M
    if S_len < masked_end:
        masked_end = S_len
        
    for start_n in range(unmasked_end, masked_end, BLOCK_N):
        current_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = current_n < S_len
        n_mask_dim1 = n_mask[:, None]
        
        k_val = tl.load(k_ptrs, mask=n_mask_dim1, other=0.0)
        v_val = tl.load(v_ptrs, mask=n_mask_dim1, other=0.0)
        
        qk = tl.dot(q_val, k_val.T, out_dtype=tl.float32)
        
        # Build causal mask securely applying bounds dynamically
        causal_mask = offs_m[:, None] >= current_n[None, :]
        mask = causal_mask & n_mask[None, :]
        if not is_m_full:
            mask = mask & m_mask[:, None]
            
        qk = tl.where(mask, qk, float("-inf"))
        
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        
        alpha = tl.exp(m_i - m_ij)
        acc = acc * alpha[:, None]
        
        p_bf16 = p.to(v_val.dtype)
        acc = tl.dot(p_bf16, v_val, acc)
        
        m_i = m_ij
        l_i = l_i * alpha + l_ij
        
        k_ptrs += BLOCK_N * stride_ks
        v_ptrs += BLOCK_N * stride_vs
        
    # Prevent 0/0 NaN for strictly invalid rows
    l_i_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_i_safe[:, None]
    
    # Store Output and LSE tracking arrays mapped safely to bounds
    o_base = o + batch_idx * stride_ob + head_idx * stride_oh
    o_ptrs = o_base + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    
    lse_base = lse + batch_idx * stride_lseb + head_idx * stride_lseh
    lse_ptrs = lse_base + offs_m * stride_lses
    lse_val = m_i + tl.log(l_i_safe)
    
    if is_m_full:
        tl.store(o_ptrs, acc.to(q.dtype.element_ty))
        tl.store(lse_ptrs, lse_val)
    else:
        tl.store(o_ptrs, acc.to(q.dtype.element_ty), mask=m_mask[:, None])
        tl.store(lse_ptrs, lse_val, mask=m_mask)


def run(Q, K, V, O, LSE):
    """
    Computes causal Multi-Head Attention forward pass and returns Output and Log-Sum-Exp.
    Writes outputs directly into preallocated `O` and `LSE` tensors following the destination-passing contract.
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