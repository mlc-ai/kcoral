import torch
import triton
import triton.language as tl


@triton.jit
def _mha_causal_kernel(
    Q, K, V, O, LSE,
    S, D,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    D_HEAD: tl.constexpr,
):
    """
    Causal multi-head attention kernel.
    
    Each program instance computes attention for one (batch, head, query_tile).
    Iterates over key tiles, applying causal mask and computing online softmax.
    """
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    H = D  # Will be inferred from context if needed, but we compute stride from args

    pid_b = pid_bh // 48  # B=4, H=48 => divide by 48 heads
    pid_h = pid_bh % 48

    # Row and column index generators
    row_idx = tl.arange(0, BLOCK_M)
    col_idx_base = tl.arange(0, BLOCK_N)
    head_idx = tl.arange(0, D_HEAD)

    # Compute strides for tensor layout [B, H, S, D]
    stride_d = 1
    stride_s = D_HEAD
    stride_h = S * D_HEAD
    stride_b = 48 * S * D_HEAD

    # Base offsets into Q, K, V for this (batch, head)
    base_offset_q = pid_b * stride_b + pid_h * stride_h
    base_offset_kv = pid_b * stride_b + pid_h * stride_h
    base_offset_o = pid_b * stride_b + pid_h * stride_h
    base_offset_lse = pid_b * (S * 48) + pid_h * S

    # Load Q tile [BLOCK_M, D_HEAD] with mask
    q_ptrs = Q + base_offset_q + row_idx[:, None] * stride_s + head_idx[None, :] * stride_d
    m_mask = row_idx[:, None] < S
    q = tl.load(q_ptrs, mask=m_mask, other=0.0)  # Loads as bf16

    # Scale factor for attention scores
    scale = 1.0 / tl.sqrt(tl.full([], D_HEAD, dtype=tl.float32))

    # Initialize online softmax accumulators
    acc_o = tl.zeros((BLOCK_M, D_HEAD), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), value=-float('inf'), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # Main loop: iterate over key tiles
    for start_n in range(0, tl.cdiv(S, BLOCK_N)):
        # Column offsets for this iteration
        col_idx = start_n * BLOCK_N + col_idx_base
        n_mask = col_idx_base < tl.cdiv(S, BLOCK_N)  # Not used directly

        # Load K tile [BLOCK_N, D_HEAD]
        k_ptrs = K + base_offset_kv + col_idx[:, None] * stride_s + head_idx[None, :] * stride_d
        kn_mask = col_idx[:, None] < S
        k = tl.load(k_ptrs, mask=kn_mask, other=0.0)

        # Compute QK^T scores [BLOCK_M, BLOCK_N]
        scores = tl.dot(q.to(tl.float32), k.to(tl.float32).T) * scale

        # Apply causal mask: mask out positions where col >= row
        causal_cond = row_idx[:, None] < col_idx[None, :]
        full_mask = causal_cond | ~(row_idx[:, None] < S) | ~(col_idx[None, :] < S)
        scores = tl.where(full_mask, -float('inf'), scores)

        # Online softmax update
        old_m = m_i
        row_max = tl.max(scores, axis=1)
        new_m = tl.maximum(old_m, row_max)
        
        alpha = tl.exp(old_m - new_m)
        p = tl.exp(scores - new_m[:, None])

        # Rescale and accumulate output
        acc_o = acc_o.to(tl.bfloat16) * alpha[:, None].to(tl.bfloat16)
        
        # Load V tile [BLOCK_N, D_HEAD]
        v_ptrs = V + base_offset_kv + col_idx[:, None] * stride_s + head_idx[None, :] * stride_d
        vn_mask = col_idx[:, None] < S
        v = tl.load(v_ptrs, mask=vn_mask, other=0.0)
        
        acc_o = tl.dot(p.to(tl.bfloat16), v.to(tl.bfloat16), acc=acc_o)
        
        # Update normalization factor
        p_sum = tl.sum(p, axis=1)
        new_l = alpha * l_i + p_sum
        
        m_i = new_m
        l_i = new_l

    # Final normalization
    l_inv = tl.reciprocal(l_i)
    o_final = acc_o * l_inv[:, None]

    # Store output O [BLOCK_M, D_HEAD]
    o_ptrs = O + base_offset_o + row_idx[:, None] * stride_s + head_idx[None, :] * stride_d
    tl.store(o_ptrs, o_final.to(tl.bfloat16), mask=m_mask)

    # Store LSE [BLOCK_M] = m_i + log(l_i)
    lse_vals = m_i + tl.log(l_i)
    lse_ptrs = LSE + base_offset_lse + row_idx
    tl.store(lse_ptrs, lse_vals, mask=row_idx < S)


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
    
    # Grid: one program per (batch, head, query_tile)
    grid = (B * H, triton.cdiv(S, BLOCK_M))
    
    _mha_causal_kernel[grid](
        Q, K, V, O, LSE,
        S, D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        D_HEAD=D,
        num_warps=4,
        num_stages=2,
    )