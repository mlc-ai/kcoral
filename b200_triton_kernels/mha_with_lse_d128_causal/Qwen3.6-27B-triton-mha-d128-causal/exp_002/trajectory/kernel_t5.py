import torch
import triton
import triton.language as tl


@triton.jit
def _causal_attn_fwd(
    Q, K, V, O, LSE,
    S,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lsb, stride_lsh, stride_ls,
    softmax_scale,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    b = pid_bh // NUM_HEADS
    h = pid_bh % NUM_HEADS

    # Compile-time indices
    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_d = tl.arange(0, HEAD_DIM)

    q_valid = off_m < S          # [BLOCK_M]
    q_mask = q_valid[:, None]    # [BLOCK_M, 1]

    # Load Q tile [BLOCK_M, HEAD_DIM] -> fp32
    q = tl.load(
        Q + b * stride_qb + h * stride_qh
          + off_m[:, None] * stride_qs + off_d[None, :] * stride_qd,
        mask=q_mask, other=0.0,
    ).to(tl.float32)

    # Online softmax accumulators (fp32 throughout)
    acc_o = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)

    num_kv_blocks = tl.cdiv(S, BLOCK_N)

    for blk in range(num_kv_blocks):
        off_n = blk * BLOCK_N + tl.arange(0, BLOCK_N)

        k_valid = off_n < S            # [BLOCK_N]
        k_mask = k_valid[None, :]      # [1, BLOCK_N]

        # Load K tile [BLOCK_N, HEAD_DIM] -> fp32
        k = tl.load(
            K + b * stride_kb + h * stride_kh
              + off_n[:, None] * stride_ks + off_d[None, :] * stride_kd,
            mask=k_mask.T, other=0.0,
        ).to(tl.float32)

        # Attention scores [BLOCK_M, BLOCK_N] = Q @ K^T * scale
        scores = tl.dot(q, k.T) * softmax_scale

        # Causal mask: key pos <= query pos
        causal = off_m[:, None] >= off_n[None, :]
        valid = causal & q_mask & k_mask

        # Mask invalid scores to -inf
        scores = tl.where(valid, scores, float("-inf"))

        # --- Online softmax update ---
        raw_max = tl.max(scores, axis=1, keep_dims=False)
        m_ij = tl.maximum(raw_max, -1e10)  # avoid NaN from all-inf blocks

        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)

        p = tl.exp(scores - m_new[:, None])

        # Scale down old accumulation
        acc_o = acc_o * alpha[:, None]

        # Load V tile [BLOCK_N, HEAD_DIM] -> fp32
        v = tl.load(
            V + b * stride_vb + h * stride_vh
              + off_n[:, None] * stride_vs + off_d[None, :] * stride_vd,
            mask=k_mask.T, other=0.0,
        ).to(tl.float32)

        # Accumulate p @ V
        pv = tl.dot(p, v)
        acc_o = acc_o + pv

        # Update normalization denominator
        p_sum = tl.sum(p, axis=1, keep_dims=False)
        l_i = alpha * l_i + p_sum

        # Advance running max
        m_i = m_new

    # Write output O as bf16
    tl.store(
        O + b * stride_ob + h * stride_oh
          + off_m[:, None] * stride_os + off_d[None, :] * stride_od,
        acc_o.to(tl.bfloat16),
        mask=q_mask,
    )

    # Write LSE as fp32
    lse_val = m_i + tl.log(l_i)
    tl.store(
        LSE + b * stride_lsb + h * stride_lsh + off_m * stride_ls,
        lse_val,
        mask=q_valid,
    )


def run(Q, K, V, O, LSE):
    """Compute causal MHA forward (O, LSE) into pre-allocated output tensors."""
    torch.cuda.set_device(Q.device)

    B, H, S, D = Q.shape
    softmax_scale = 1.0 / float(D ** 0.5)

    BLOCK_M = 64
    BLOCK_N = 64

    grid = (B * H, triton.cdiv(S, BLOCK_M))

    _causal_attn_fwd[grid](
        Q, K, V, O, LSE,
        S,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        softmax_scale,
        NUM_HEADS=H,
        HEAD_DIM=D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=3,
    )