import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=3),
    ],
    key=['S', 'D'],
)
@triton.jit
def _mha_fwd_kernel(
    Q, K, V, O, LSE,
    stride_bq, stride_hq, stride_sq, stride_dq,
    stride_bk, stride_hk, stride_sk, stride_dk,
    stride_bv, stride_hv, stride_sv, stride_dv,
    stride_bo, stride_ho, stride_so, stride_do,
    stride_blse, stride_hlse, stride_slse,
    B, H, S, D,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """Flash-Attention style causal MHA kernel with online softmax."""
    batch_head = tl.program_id(0)
    start_m = tl.program_id(1) * BLOCK_M

    # Offsets for rows (query seq pos) and columns (head dim)
    off_m = start_m + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    off_d = tl.arange(0, D)                    # [D]

    # Precompute batch+head offset for all tensors
    bh_offset_q = batch_head * stride_hq       # combined B,H offset using stride_hq
    bh_offset_k = batch_head * stride_hk
    bh_offset_v = batch_head * stride_hv
    bh_offset_o = batch_head * stride_ho
    bh_offset_lse = batch_head * stride_hlse

    # Load Q tile [BLOCK_M, D]
    q_ptrs = Q + bh_offset_q + off_m[:, None] * stride_sq + off_d[None, :] * stride_dq
    q = tl.load(q_ptrs, mask=(off_m[:, None] < S) & (off_d[None, :] < D), other=0.0)

    # Initialize online softmax state
    acc_o = tl.zeros((BLOCK_M, D), dtype=tl.float32)       # accumulated output
    m_i = tl.full((BLOCK_M,), value=float('-inf'), dtype=tl.float32)  # running max
    d_i = tl.full((BLOCK_M,), value=1.0, dtype=tl.float32)  # running sum of exp

    inv_sqrt_d = 1.0 / tl.sqrt(float(D))

    # Iterate over K/V blocks
    for start_n in range(0, S, BLOCK_N):
        off_n = start_n + tl.arange(0, BLOCK_N)  # [BLOCK_N]

        # Load K tile [BLOCK_N, D]
        k_ptrs = K + bh_offset_k + (start_n + tl.arange(0, BLOCK_N))[:, None] * stride_sk + off_d[None, :] * stride_dk
        k = tl.load(k_ptrs,
                     mask=((start_n + tl.arange(0, BLOCK_N))[:, None] < S) & (off_d[None, :] < D),
                     other=0.0)

        # Load V tile [BLOCK_N, D]
        v_ptrs = V + bh_offset_v + (start_n + tl.arange(0, BLOCK_N))[:, None] * stride_sv + off_d[None, :] * stride_dv
        v = tl.load(v_ptrs,
                     mask=((start_n + tl.arange(0, BLOCK_N))[:, None] < S) & (off_d[None, :] < D),
                     other=0.0)

        # Compute scores: Q@K^T / sqrt(D) -> [BLOCK_M, BLOCK_N]
        scores = tl.dot(q, k.T) * inv_sqrt_d

        # Causal mask: query_pos <= key_pos
        causal_mask = (start_m + off_m[:, None]) <= (start_n + off_n[None, :])
        scores = tl.where(causal_mask, scores, float('-inf'))

        # Online softmax update (online algorithm for numerical stability)
        m_ij = tl.max(scores, axis=1)          # [BLOCK_M]
        m_i_old = m_i
        m_i = tl.maximum(m_i, m_ij)            # new running max
        p = tl.exp(scores - m_i[:, None])      # stabilized probabilities [BLOCK_M, BLOCK_N]
        alpha = tl.exp(m_i_old - m_i)          # rescaling factor [BLOCK_M]
        d_i = d_i * alpha + tl.sum(p, axis=1)  # update running denominator
        acc_o = acc_o * alpha[:, None] + tl.dot(p, v)  # update accumulated output

    # Normalize and store output O
    acc_o = acc_o / d_i[:, None]
    o_ptrs = O + bh_offset_o + off_m[:, None] * stride_so + off_d[None, :] * stride_do
    tl.store(o_ptrs, acc_o.to(dtype=tl.bfloat16),
             mask=(off_m[:, None] < S) & (off_d[None, :] < D))

    # Store LSE = m_i + log(d_i)
    lse_val = m_i + tl.log(d_i)
    lse_ptrs = LSE + bh_offset_lse + off_m * stride_slse
    tl.store(lse_ptrs, lse_val, mask=off_m < S)


def run(Q, K, V, O, LSE):
    """Compute causal multi-head attention with LSE output.

    Args:
        Q: Input query tensor [B, H, S, D], bf16
        K: Input key tensor [B, H, S, D], bf16
        V: Input value tensor [B, H, S, D], bf16
        O: Output tensor [B, H, S, D], bf16 (preallocated)
        LSE: Log-sum-exp output [B, H, S], fp32 (preallocated)
    """
    torch.cuda.set_device(Q.device)

    B, H, S, D = Q.shape

    n_batch_heads = B * H
    grid = (n_batch_heads, triton.cdiv(S, 128))

    _mha_fwd_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        B, H, S, D,
    )