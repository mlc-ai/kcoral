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
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """FlashAttention-style MHA kernel for a single (batch, head) pair."""

    pid_bh = tl.program_id(0)
    pid_qtile = tl.program_id(1)

    # Absolute row indices for this query tile
    m_abs = pid_qtile * BLOCK_M
    m_idx = m_abs + tl.arange(0, BLOCK_M)
    m_mask = m_idx < S

    # Head dimension indices
    d_idx = tl.arange(0, BLOCK_D)
    d_mask = d_idx < BLOCK_D  # BLOCK_D == D here, always fully covered

    # Base offset in flattened [BH, S, D] layout
    bh_offset = pid_bh * stride_seq * S

    # Online softmax accumulators
    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.full((BLOCK_M,), 1.0, dtype=tl.float32)
    acc_o = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    # Iterate over key sequence tiles
    for start_n in range(0, S, BLOCK_N):
        n_idx = start_n + tl.arange(0, BLOCK_N)
        n_mask = n_idx < S

        # Build Q pointers and load tile [BLOCK_M, BLOCK_D]
        q_ptrs = q_ptr + bh_offset + m_idx[:, None] * stride_seq + d_idx[None, :]
        q = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0)

        # Build K pointers and load tile [BLOCK_N, BLOCK_D]
        k_ptrs = k_ptr + bh_offset + n_idx[:, None] * stride_seq + d_idx[None, :]
        k = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)

        # Build V pointers and load tile [BLOCK_N, BLOCK_D]
        v_ptrs = v_ptr + bh_offset + n_idx[:, None] * stride_seq + d_idx[None, :]
        v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)

        # Compute attention scores: Q @ K^T * scale  -> [BLOCK_M, BLOCK_N]
        # Cast to fp32 for accumulation
        q_f32 = q.to(tl.float32)
        k_f32 = k.to(tl.float32)
        v_f32 = v.to(tl.float32)

        s = tl.dot(q_f32, k_f32.T) * scale

        # Online softmax: update running max
        m_ij = tl.max(s, axis=1)
        m_new = tl.maximum(m_i, m_ij)

        # Stable scaling factors
        alpha = tl.exp(m_i - m_new)

        # Stabilized attention probabilities
        p = tl.exp(s - m_new[:, None])

        # Update output accumulator: acc = alpha * acc + p @ V
        acc_o = alpha[:, None] * acc_o + tl.dot(p, v_f32)

        # Update l accumulator (sum of probs scaled by exp(old_max - new_max))
        beta = tl.sum(p, axis=1)
        l_i = alpha * l_i + beta

        # Advance running max
        m_i = m_new

    # Final normalization
    l_safe = tl.where(l_i > 0.0, l_i, 1.0)
    acc_o = acc_o / l_safe[:, None]

    # Write output O [BLOCK_M, BLOCK_D] -> bf16
    o_ptrs = o_ptr + bh_offset + m_idx[:, None] * stride_seq + d_idx[None, :]
    tl.store(o_ptrs, acc_o.to(tl.bfloat16), mask=m_mask[:, None])

    # Write LSE = m + log(l) [BLOCK_M] -> float32
    lse_val = m_i + tl.log(l_safe)
    lse_offsets = pid_bh * S + m_idx
    tl.store(lse_ptr + lse_offsets, lse_val, mask=m_mask)


def run(Q, K, V, O, LSE):
    """Multi-head attention forward: O = softmax(Q@K^T/sqrt(D))@V with LSE.

    Inputs Q, K, V: [B, H, S, D] bf16
    Outputs O: [B, H, S, D] bf16, LSE: [B, H, S] f32
    Destination-passing: writes into preallocated O and LSE tensors.
    """
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape

    # Reshape to [BH, S, D] contiguous for simple linear indexing
    Q_c = Q.reshape(B * H, S, D).contiguous()
    K_c = K.reshape(B * H, S, D).contiguous()
    V_c = V.reshape(B * H, S, D).contiguous()
    O_c = O.reshape(B * H, S, D).contiguous()

    num_bh = B * H
    stride_seq = D  # stride along S in row-major [BH, S, D] layout

    # Tile sizes
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = D  # Full head dimension in one tile (D=128)

    # Precompute attention scale on host
    scale = 1.0 / (D ** 0.5)

    # Grid: first dim = batch*head pairs, second dim = query sequence tiles
    grid = (num_bh, triton.cdiv(S, BLOCK_M))

    _attention_kernel[grid](
        Q_c, K_c, V_c,
        O_c, LSE,
        stride_seq,
        S,
        scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=2,
    )