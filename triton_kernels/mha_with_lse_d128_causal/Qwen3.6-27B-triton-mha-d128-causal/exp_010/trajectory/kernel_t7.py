import torch
import triton
import triton.language as tl


@triton.jit
def _mha_causal_kernel(
    Q, K, V, O, LSE,
    B, H, S,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_lse_b, stride_lse_h,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D_HEAD: tl.constexpr,
):
    """
    Causal multi-head attention kernel with online softmax.
    
    Each program instance computes attention for one (batch, head, query_tile).
    Iterates over key tiles, applying causal mask and computing online softmax.
    """
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    pid_b = pid_bh // H
    pid_h = pid_bh % H

    # Index generators
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D_HEAD)

    # Starting position for this query tile
    m_start = pid_m * BLOCK_M

    # Base pointer offsets for this (batch, head)
    q_base = Q + pid_b * stride_qb + pid_h * stride_qh
    kv_base = K + pid_b * stride_qb + pid_h * stride_qh
    v_base = V + pid_b * stride_qb + pid_h * stride_qh
    o_base = O + pid_b * stride_qb + pid_h * stride_qh
    lse_base = LSE + pid_b * stride_lse_b + pid_h * stride_lse_h

    # Load Q tile [BLOCK_M, D_HEAD]
    q_ptrs = q_base + (m_start + offs_m[:, None]) * stride_qs + offs_d[None, :] * stride_qd
    q_mask = (m_start + offs_m[:, None]) < S
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # Attention scale
    inv_scale = tl.sqrt(tl.cast(D_HEAD, tl.float32))

    # Initialize online softmax accumulators (all fp32)
    acc = tl.zeros((BLOCK_M, D_HEAD), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), value=float('-inf'), dtype=tl.float32)
    l_i = tl.full((BLOCK_M,), value=1.0, dtype=tl.float32)

    num_k_tiles = tl.cdiv(S, BLOCK_N)

    for start_n in range(num_k_tiles):
        n_start = start_n * BLOCK_N

        # Load K tile as [D_HEAD, BLOCK_N] so tl.dot(q, k) gives [BLOCK_M, BLOCK_N]
        k_ptrs = kv_base + offs_d[:, None] * stride_qd + (n_start + offs_n[None, :]) * stride_qs
        k_mask = (n_start + offs_n[None, :]) < S
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        # Q @ K^T: q is [BLOCK_M, D_HEAD], k is [D_HEAD, BLOCK_N] => result [BLOCK_M, BLOCK_N]
        scores = tl.dot(q, k)
        scores = scores / inv_scale

        # Causal mask: query_pos >= key_pos AND valid positions
        q_pos = m_start + offs_m[:, None]
        k_pos = n_start + offs_n[None, :]
        
        causal = q_pos >= k_pos
        valid = (q_pos < S) & (k_pos < S)
        mask = causal & valid
        
        scores = tl.where(mask, scores, float('-inf'))

        # Online softmax step
        m_prev = m_i
        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        
        # Stable exponentials
        p_alpha = tl.exp(m_prev - m_new)
        p_scores = tl.exp(scores - m_new[:, None])

        # Rescale previous accumulator
        acc = acc * p_alpha[:, None]

        # Load V tile [BLOCK_N, D_HEAD]
        v_ptrs = v_base + (n_start + offs_n[:, None]) * stride_qs + offs_d[None, :] * stride_qd
        v_mask = (n_start + offs_n[:, None]) < S
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)

        # P @ V: p_scores is [BLOCK_M, BLOCK_N], v is [BLOCK_N, D_HEAD] => result [BLOCK_M, D_HEAD]
        acc = acc + tl.dot(p_scores.to(tl.bfloat16), v)

        # Update normalization denominator
        p_rowsum = tl.sum(p_scores, axis=1)
        l_i = p_alpha * l_i + p_rowsum

        # Update running max
        m_i = m_new

    # Final normalization
    denom = tl.where(l_i != 0, l_i, 1e-12)
    o_out = acc / denom[:, None]

    # Store output O
    o_ptrs = o_base + (m_start + offs_m[:, None]) * stride_qs + offs_d[None, :] * stride_qd
    tl.store(o_ptrs, o_out.to(tl.bfloat16), mask=q_mask)

    # Store LSE = m_i + log(l_i)
    lse_vals = tl.where(l_i != 0, m_i + tl.log(l_i), float('-inf'))
    lse_ptrs = lse_base + (m_start + offs_m)
    tl.store(lse_ptrs, lse_vals, mask=(m_start + offs_m) < S)


def run(Q, K, V, O, LSE):
    """
    Compute causal multi-head attention forward pass.
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    
    # Tile sizes optimized for Hopper with D=128
    BLOCK_M = 128
    BLOCK_N = 64
    
    # Strides for Q/K/V/O tensors [B, H, S, D]
    stride_qb = H * S * D
    stride_qh = S * D
    stride_qs = D
    stride_qd = 1
    
    # Strides for LSE [B, H, S]
    stride_lse_b = H * S
    stride_lse_h = S
    
    grid = ((B * H, triton.cdiv(S, BLOCK_M)))
    
    _mha_causal_kernel[grid](
        Q, K, V, O, LSE,
        B, H, S,
        stride_qb, stride_qh, stride_qs, stride_qd,
        stride_lse_b, stride_lse_h,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        D_HEAD=D,
        num_warps=8,
        num_stages=3,
    )