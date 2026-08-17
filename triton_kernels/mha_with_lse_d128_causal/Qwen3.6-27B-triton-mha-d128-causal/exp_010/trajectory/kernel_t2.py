import torch
import triton
import triton.language as tl


@triton.jit
def _mha_causal_kernel(
    Q, K, V, O, LSE,
    B, H, S,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_lse_b, stride_lse_h, stride_lse_s,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D_HEAD: tl.constexpr,
):
    """
    Causal multi-head attention kernel.
    
    Each program instance computes attention for one (batch, head, query_tile).
    Iterates over key tiles, applying causal mask and computing online softmax.
    """
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    pid_b = pid_bh // H
    pid_h = pid_bh % H

    # Row and column index generators
    row_idx = tl.arange(0, BLOCK_M)
    col_idx_base = tl.arange(0, BLOCK_N)
    d_idx = tl.arange(0, D_HEAD)

    # Query offset start position for this (batch, head)
    q_offset = pid_b * stride_qb + pid_h * stride_qh
    kv_offset = pid_b * stride_qb + pid_h * stride_qh
    o_offset = pid_b * stride_qb + pid_h * stride_qh
    lse_offset = pid_b * stride_lse_b + pid_h * stride_lse_h

    # Load Q tile [BLOCK_M, D_HEAD] with proper masking
    q_row_start = pid_m * BLOCK_M
    q_ptrs = Q + q_offset + (q_row_start + row_idx[:, None]) * stride_qs + d_idx[None, :] * stride_qd
    m_mask = (q_row_start + row_idx[:, None]) < S
    q = tl.load(q_ptrs, mask=m_mask, other=0.0)

    # Scale factor for attention scores: 1/sqrt(D)
    scale = 1.0 / tl.sqrt(tl.full((1,), D_HEAD, dtype=tl.float32))
    scale = scale.to(tl.float32)

    # Initialize online softmax accumulators
    acc_o = tl.zeros((BLOCK_M, D_HEAD), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), value=float('-inf'), dtype=tl.float32)
    l_i = tl.full((BLOCK_M,), value=1.0, dtype=tl.float32)

    num_k_tiles = tl.cdiv(S, BLOCK_N)

    # Main loop: iterate over key tiles
    for start_n in range(num_k_tiles):
        # Column offsets for this iteration
        k_col_start = start_n * BLOCK_N
        col_idx = k_col_start + col_idx_base
        
        # Load K tile [BLOCK_N, D_HEAD]
        k_ptrs = K + kv_offset + (k_col_start + col_idx[:, None]) * stride_qs + d_idx[None, :] * stride_qd
        kn_mask = (k_col_start + col_idx[:, None]) < S
        k = tl.load(k_ptrs, mask=kn_mask, other=0.0)

        # Compute QK^T scores [BLOCK_M, BLOCK_N]
        scores = tl.dot(q.to(tl.float32), k.to(tl.float32).T)
        scores = scores * scale

        # Apply causal mask: keep only where query_pos >= key_pos (lower triangle)
        query_pos = q_row_start + row_idx[:, None]  # [BLOCK_M, 1]
        key_pos = k_col_start + col_idx[None, :]    # [1, BLOCK_N]
        causal_mask = query_pos >= key_pos
            
        # Also handle boundary: if key position exceeds S, mask it out
        valid_keys = key_pos < S
        full_mask = causal_mask & valid_keys
            
        scores = tl.where(full_mask, scores, float('-inf'))

        # Online softmax update
        old_m = m_i
        row_max = tl.max(scores, axis=1)
        new_m = tl.maximum(old_m, row_max)
        
        alpha = tl.exp(old_m - new_m)
        p = tl.exp(scores - new_m[:, None])

        # Rescale accumulator before adding
        acc_o_scaled = acc_o * alpha[:, None]
        
        # Load V tile [BLOCK_N, D_HEAD]
        v_ptrs = V + kv_offset + (k_col_start + col_idx[:, None]) * stride_qs + d_idx[None, :] * stride_qd
        vn_mask = (k_col_start + col_idx[:, None]) < S
        v = tl.load(v_ptrs, mask=vn_mask, other=0.0)
        
        # Accumulate output: P @ V
        acc_o_scaled = acc_o_scaled + tl.dot(p.to(tl.bfloat16), v.to(tl.bfloat16))
        
        # Update normalization factor
        p_sum = tl.sum(p, axis=1)
        new_l = alpha * l_i + p_sum
        
        m_i = new_m
        l_i = new_l
        acc_o = acc_o_scaled

    # Final normalization: divide by sum
    denom = l_i + 1e-6  # Add small epsilon for numerical stability
    l_inv = 1.0 / denom
    o_final = acc_o * l_inv[:, None]

    # Store output O [BLOCK_M, D_HEAD]
    o_ptrs = O + o_offset + (q_row_start + row_idx[:, None]) * stride_qs + d_idx[None, :] * stride_qd
    tl.store(o_ptrs, o_final.to(tl.bfloat16), mask=m_mask)

    # Store LSE [BLOCK_M] = m_i + log(l_i)
    lse_vals = m_i + tl.log(l_i)
    lse_ptrs = LSE + lse_offset + (q_row_start + row_idx)
    tl.store(lse_ptrs, lse_vals, mask=(q_row_start + row_idx) < S)


def run(Q, K, V, O, LSE):
    """
    Compute causal multi-head attention forward pass.
    
    Args:
        Q: Query tensor [B, H, S, D], bfloat16
        K: Key tensor [B, H, S, D], bfloat16  
        V: Value tensor [B, H, S, D], bfloat16
        O: Preallocated output tensor [B, H, S, D], bfloat16
        LSE: Preallocated LSE tensor [B, H, S], float32
    
    Computes:
        O = softmax(Q @ K^T / sqrt(D)) @ V with causal mask
        LSE = max(P) + log(sum(exp(P - max(P)))) where P = Q @ K^T / sqrt(D)
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    
    # Tile configuration optimized for Hopper with D=128
    BLOCK_M = 64
    BLOCK_N = 64
    
    # Strides for Q/K/V/O tensors [B, H, S, D]
    stride_qb = H * S * D
    stride_qh = S * D
    stride_qs = D
    stride_qd = 1
    
    # Strides for LSE [B, H, S]
    stride_lse_b = H * S
    stride_lse_h = S
    stride_lse_s = 1
    
    # Grid: one program per (batch*head, query_tile)
    grid = ((B * H, triton.cdiv(S, BLOCK_M)))
    
    _mha_causal_kernel[grid](
        Q, K, V, O, LSE,
        B, H, S,
        stride_qb, stride_qh, stride_qs, stride_qd,
        stride_lse_b, stride_lse_h, stride_lse_s,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        D_HEAD=D,
        num_warps=4,
        num_stages=2,
    )