import torch
import triton
import triton.language as tl


@triton.jit
def _mha_causal_fwd_kernel(
    Q, K, V, O, LSE,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lsb, stride_lsh, stride_ls,
    B, H, S,
    softmax_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh % H

    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_d = tl.arange(0, D)

    # Load Q tile [BLOCK_M, D], immediately cast to fp32
    q = tl.load(
        Q + b * stride_qb + h * stride_qh +
        off_m[:, None] * stride_qs + off_d[None, :] * stride_qd,
        mask=(off_m[:, None] < S), other=0.0,
    ).to(tl.float32)

    # Online softmax accumulators (fp32 throughout)
    acc_out = tl.zeros((BLOCK_M, D), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)

    num_kv_tiles = tl.cdiv(S, BLOCK_N)

    for blk in range(num_kv_tiles):
        off_n = blk * BLOCK_N + tl.arange(0, BLOCK_N)

        # Load K tile [BLOCK_N, D] -> fp32
        k = tl.load(
            K + b * stride_kb + h * stride_kh +
            off_n[:, None] * stride_ks + off_d[None, :] * stride_kd,
            mask=(off_n[:, None] < S), other=0.0,
        ).to(tl.float32)

        # Attention scores [BLOCK_M, BLOCK_N] = Q @ K^T * scale
        scores = tl.dot(q, k.T) * softmax_scale

        # Build combined validity mask
        # Causal: key pos j must satisfy j <= query pos i
        q_pos = off_m[:, None]      # [BLOCK_M, 1]
        k_pos = off_n[None, :]      # [1, BLOCK_N]
        causal_mask = q_pos >= k_pos       # [BLOCK_M, BLOCK_N]
        q_valid = q_pos < S              # [BLOCK_M, 1]
        k_valid = k_pos < S              # [1, BLOCK_N]
        valid = causal_mask & q_valid & k_valid  # [BLOCK_M, BLOCK_N]

        # Mask invalid scores to -inf
        scores = tl.where(valid, scores, float("-inf"))

        # --- Online softmax update ---
        m_ij = tl.max(scores, axis=1, keep_dims=False)  # [BLOCK_M]
        m_new = tl.maximum(m_i, m_ij)                    # [BLOCK_M]

        # Scale factor for stale accumulators
        alpha = tl.exp(m_i - m_new)                      # [BLOCK_M]

        # Probabilities centered around new max
        p = tl.exp(scores - m_new[:, None])              # [BLOCK_M, BLOCK_N]

        # Decay old output accumulator
        acc_out = acc_out * alpha[:, None]

        # Load V tile [BLOCK_N, D] -> fp32
        v = tl.load(
            V + b * stride_vb + h * stride_vh +
            off_n[:, None] * stride_vs + off_d[None, :] * stride_vd,
            mask=(off_n[:, None] < S), other=0.0,
        ).to(tl.float32)

        # Accumulate p @ v into output
        acc_out = tl.dot(p, v, acc=acc_out)

        # Update normalization denominator
        l_i = alpha * l_i + tl.sum(p, axis=1, keep_dims=False)  # [BLOCK_M]

        # Advance running max
        m_i = m_new

    # Store output O [BLOCK_M, D] as bf16
    tl.store(
        O + b * stride_ob + h * stride_oh +
        off_m[:, None] * stride_os + off_d[None, :] * stride_od,
        acc_out.to(tl.bfloat16),
        mask=(off_m[:, None] < S),
    )

    # Store LSE [BLOCK_M] as fp32: LSE = m_final + log(l_final)
    lse_val = m_i + tl.log(l_i)
    tl.store(
        LSE + b * stride_lsb + h * stride_lsh + off_m * stride_ls,
        lse_val,
        mask=(off_m < S),
    )


def run(Q, K, V, O, LSE):
    """Compute causal MHA forward (O, LSE) into pre-allocated output tensors."""
    torch.cuda.set_device(Q.device)

    B, H, S, D = Q.shape
    softmax_scale = 1.0 / float(D ** 0.5)

    BLOCK_M = 64
    BLOCK_N = 64

    grid = (B * H, triton.cdiv(S, BLOCK_M))

    _mha_causal_fwd_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        B, H, S,
        softmax_scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        D=D,
        num_warps=4,
        num_stages=3,
    )