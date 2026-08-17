import torch
import triton
import triton.language as tl


@triton.jit
def _attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    lse_ptr,
    stride_seq,
    S,
    D,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """FlashAttention-style MHA kernel operating on a single (batch, head) pair.
    
    Each program instance handles one query tile for one (batch, head) pair.
    Loops over key sequence tiles with online softmax accumulation.
    """

    pid_bh = tl.program_id(0)
    pid_qtile = tl.program_id(1)

    # Absolute row indices for this query tile
    m_abs = pid_qtile * BLOCK_M
    m_idx = m_abs + tl.arange(0, BLOCK_M)
    m_mask = m_idx < S

    # Head dimension indices
    d_idx = tl.arange(0, BLOCK_D)
    d_mask = d_idx < D

    # Base offset in flattened [BH, S, D] layout
    bh_offset = pid_bh * stride_seq * S

    # Scale for attention
    scale = 1.0 / tl.sqrt(D).to(tl.float32)

    # Online softmax accumulators
    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.full((BLOCK_M,), 1.0, dtype=tl.float32)
    acc_o = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    # Iterate over key sequence tiles
    for start_n in range(0, S, BLOCK_N):
        n_idx = start_n + tl.arange(0, BLOCK_N)
        n_mask = n_idx < S

        # Accumulate Q @ K^T in blocks of BLOCK_D
        s = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        
        for dk in range(0, D, BLOCK_D):
            d_inner = tl.arange(0, BLOCK_D)
            
            q_tile = tl.load(
                q_ptr + bh_offset + m_idx[:, None] * stride_seq + (dk + d_inner)[None, :],
                mask=m_mask[:, None] & d_mask[None, :],
                other=0.0,
            )
            k_tile = tl.load(
                k_ptr + bh_offset + n_idx[:, None] * stride_seq + (dk + d_inner)[None, :],
                mask=n_mask[:, None] & d_mask[None, :],
                other=0.0,
            )
            s = tl.dot(q_tile, k_tile.T, s)

        s = s * scale

        # Update running softmax max
        m_ij = tl.max(s, axis=1)
        m_new = tl.maximum(m_i, m_ij)

        # Scale factors for numerical stability
        alpha = tl.exp(m_i - m_new)
        
        # Recompute stabilized scores and attention probs
        p = tl.exp(s - m_new[:, None])

        # Update output accumulator: acc = alpha * acc + p @ V
        v_tile = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
        for dk in range(0, D, BLOCK_D):
            d_inner = tl.arange(0, BLOCK_D)
            v_tile = tl.load(
                v_ptr + bh_offset + n_idx[:, None] * stride_seq + (dk + d_inner)[None, :],
                mask=n_mask[:, None] & d_mask[None, :],
                other=0.0,
            )
        
        acc_o = alpha[:, None] * acc_o + tl.dot(p, v_tile)

        # Update l accumulator
        beta = tl.sum(p, axis=1)
        l_i = alpha * l_i + beta

        # Update m
        m_i = m_new

    # Final normalization and output
    l_safe = tl.where(l_i > 0, l_i, 1.0)
    acc_o = acc_o / l_safe[:, None]

    # Write O
    o_ptrs = o_ptr + bh_offset + m_idx[:, None] * stride_seq + d_idx[None, :]
    tl.store(o_ptrs, acc_o.to(tl.bfloat16), mask=m_mask[:, None] & d_mask[None, :])

    # Write LSE = m + log(l)
    lse = m_i + tl.log(l_safe)
    lse_row_offset = pid_bh * S + m_abs
    tl.store(lse_ptr + lse_row_offset + tl.arange(0, BLOCK_M), lse, mask=m_mask)


def run(Q, K, V, O, LSE):
    """Multi-head attention forward: O = softmax(Q@K^T/sqrt(D))@V with LSE.
    
    Inputs Q, K, V: [B, H, S, D] bf16
    Outputs O: [B, H, S, D] bf16, LSE: [B, H, S] f32
    
    Destination-passing: writes into preallocated O and LSE.
    """
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape

    # Contiguous views as [BH, S, D] for simple indexing
    Q_c = Q.reshape(B * H, S, D).contiguous()
    K_c = K.reshape(B * H, S, D).contiguous()
    V_c = V.reshape(B * H, S, D).contiguous()
    O_c = O.reshape(B * H, S, D).contiguous()

    num_bh = B * H
    stride_seq = D  # stride along S in [BH, S, D] row-major layout

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = 64

    # Grid: first dim = batch*head pairs (capped at NUM_SMS for persistent scheduling),
    #        second dim = query sequence tiles
    NUM_SMS = 132
    grid = (min(NUM_SMS, num_bh), triton.cdiv(S, BLOCK_M))

    _attention_kernel[grid](
        Q_c, K_c, V_c,
        O_c, LSE,
        stride_seq,
        S, D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=2,
    )