import torch
import triton
import triton.language as tl


@triton.jit
def _causal_mha_kernel(
    Q, K, V, O, LSE,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    S, H, D,
    NEG_INF: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    
    num_pid_m = tl.cdiv(S, BLOCK_M)
    num_bh = B * H
    
    bh = pid // num_pid_m
    bm = pid % num_pid_m
    
    bid = bh // H
    hid = bh % H
    
    # Base pointers for this (batch, head)
    q_base = Q + bid * stride_qb + hid * stride_qh
    k_base = K + bid * stride_kb + hid * stride_kh
    v_base = V + bid * stride_vb + hid * stride_vh
    o_base = O + bid * stride_ob + hid * stride_oh
    lse_base = LSE + bid * stride_lseb + hid * stride_lseh
    
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    
    # Query position (absolute within sequence)
    q_abs = bm * BLOCK_M + offs_m  # [BLOCK_M]
    
    # Q pointers: [BLOCK_M, BLOCK_D]
    q_ptrs = q_base + q_abs[:, None] * stride_qs + offs_d[None, :] * stride_qd
    q_mask = (q_abs[:, None] < S) & (offs_d[None, :] < D)
    Q_TILE = tl.load(q_ptrs, mask=q_mask, other=0.0)
    
    # Initialize online softmax state
    m_i = tl.full((BLOCK_M,), NEG_INF, dtype=tl.float32)   # running max
    l_i = tl.full((BLOCK_M,), 1.0, dtype=tl.float32)        # running sum of exp
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)     # accumulated output
    
    inv_sqrt_d = 1.0 / tl.sqrt(D, dtype=tl.float32)
    
    # Sweep over key-value blocks
    for start_n in range(0, S, BLOCK_N):
        n_abs = start_n + offs_n  # [BLOCK_N]
        
        # Load K tile [BLOCK_N, BLOCK_D]
        k_ptrs = k_base + n_abs[:, None] * stride_ks + offs_d[None, :] * stride_kd
        k_mask = (n_abs[:, None] < S) & (offs_d[None, :] < D)
        K_TILE = tl.load(k_ptrs, mask=k_mask, other=0.0)
        
        # Load V tile [BLOCK_N, BLOCK_D]
        v_ptrs = v_base + n_abs[:, None] * stride_vs + offs_d[None, :] * stride_vd
        V_TILE = tl.load(v_ptrs, mask=k_mask, other=0.0)
        
        # Compute scores: Q[K,D] @ K^T[D,N] -> [BLOCK_M, BLOCK_N]
        scores = tl.dot(Q_TILE, K_TILE.T) * inv_sqrt_d
        
        # Apply causal mask: query_pos >= key_pos
        causal_mask = q_abs[:, None] >= n_abs[None, :]
        scores = tl.where(causal_mask, scores, NEG_INF)
        
        # Online softmax: find new row-wise max
        row_max = tl.max(scores, axis=1)  # [BLOCK_M]
        m_new = tl.maximum(m_i, row_max)
        
        # Scale factor for old accumulation
        alpha = tl.exp(m_i - m_new)  # [BLOCK_M]
        
        # Attention probs
        P = tl.exp(scores - m_new[:, None])  # [BLOCK_M, BLOCK_N]
        
        # Update denominator (running sum of exp)
        l_i = alpha * l_i + tl.sum(P, axis=1)  # [BLOCK_M]
        
        # Update numerator (weighted accumulation of V)
        # old_acc * alpha + P @ V_TILE
        acc = tl.dot(P, V_TILE, acc=alpha * acc)
        
        # Update running max
        m_i = m_new
    
    # Final normalization
    l_i_safe = tl.where(l_i > 0.0, l_i, 1.0)
    O_TILE = acc / l_i_safe[:, None]
    
    # LSE = m_i + log(l_i); if no valid keys, LSE = -inf
    lse_valid = l_i > 0.0
    lse = tl.where(lse_valid, m_i + tl.log(l_i), NEG_INF)
    
    # Store outputs
    o_ptrs = o_base + q_abs[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(o_ptrs, O_TILE.to(torch.bfloat16), mask=q_mask)
    
    lse_ptrs = lse_base + q_abs * stride_lses
    tl.store(lse_ptrs, lse, mask=q_abs < S, other=NEG_INF)


def run(Q, K, V, O, LSE):
    """Compute causal multi-head attention: O = softmax(Q@K^T/sqrt(D))@V, LSE = logsumexp(Q@K^T/sqrt(D)).
    
    Args:
        Q:  (B, H, S, D) bfloat16
        K:  (B, H, S, D) bfloat16
        V:  (B, H, S, D) bfloat16
        O:  (B, H, S, D) bfloat16 - preallocated output
        LSE:(B, H, S) float32 - preallocated LSE output
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    
    if S == 0 or B == 0 or H == 0:
        # Nothing to compute; outputs already allocated
        return
    
    # Grid: each program handles one (batch, head) x one query tile
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = 128
    
    num_tiles_m = triton.cdiv(S, BLOCK_M)
    grid = (B * H * num_tiles_m,)
    
    _causal_mha_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        S, H, D,
        NEG_INF=float("-inf"),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=3,
    )