import torch
import triton
import triton.language as tl


@triton.jit
def _mha_forward(
    Q, K, V, O, LSE,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lb, stride_lh, stride_ls,
    B, H, S, D,
    SCALE,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Flash-Attention style non-causal MHA forward kernel with LSE output."""

    start_m = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H

    off_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_d = tl.arange(0, BLOCK_D)
    off_n = tl.arange(0, BLOCK_N)

    # ------------------------------------------------------------------
    # Load Q tile (once, reused across K-loop)
    # Shape: (BLOCK_M, BLOCK_D)  dtype: bf16
    # ------------------------------------------------------------------
    q_offs = b * stride_qb + h * stride_qh + off_m[:, None] * stride_qs + off_d[None, :] * stride_qd
    q_mask = (off_m[:, None] < S) & (off_d[None, :] < D)
    Q_tile = tl.load(Q + q_offs, mask=q_mask, other=0.0)

    # ------------------------------------------------------------------
    # Online-softmax accumulators  (fp32)
    #   acc : running softmax-weighted sum of V  [BLOCK_M, BLOCK_D]
    #   m_i : running max of attention scores      [BLOCK_M]
    #   l_i : running sum of exp(scores - max)     [BLOCK_M]
    # ------------------------------------------------------------------
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.full((BLOCK_M,), 0.0, dtype=tl.float32)

    # ------------------------------------------------------------------
    # Sweep K/V sequence in BLOCK_N-sized tiles
    # ------------------------------------------------------------------
    for start_n in range(0, tl.cdiv(S, BLOCK_N)):
        cur_n = start_n * BLOCK_N + off_n

        # Load K tile  [BLOCK_N, BLOCK_D]
        k_offs = b * stride_kb + h * stride_kh + cur_n[:, None] * stride_ks + off_d[None, :] * stride_kd
        k_mask = (cur_n[:, None] < S) & (off_d[None, :] < D)
        K_tile = tl.load(K + k_offs, mask=k_mask, other=0.0)

        # Attention scores  [BLOCK_M, BLOCK_N]  accumulated in fp32
        p = tl.dot(Q_tile, K_tile.T) * SCALE

        # Mask out-of-bound (q, k) pairs so they do not influence softmax
        mask_p = (off_m[:, None] < S) & (cur_n[None, :] < S)
        p = tl.where(mask_p, p, float("-inf"))

        # --- online softmax bookkeeping ---
        m_i_new = tl.maximum(m_i, tl.max(p, axis=1))       # [BLOCK_M]
        alpha = tl.exp(m_i - m_i_new)                       # decay for old acc
        beta  = tl.exp(p - m_i_new[:, None])                # new attention weights

        # Scale previous accumulator & add new contribution
        acc = acc * alpha[:, None]

        v_offs = b * stride_vb + h * stride_vh + cur_n[:, None] * stride_vs + off_d[None, :] * stride_vd
        v_mask = (cur_n[:, None] < S) & (off_d[None, :] < D)
        V_tile = tl.load(V + v_offs, mask=v_mask, other=0.0)

        # acc += beta @ V   [BLOCK_M, BLOCK_N] x [BLOCK_N, BLOCK_D] -> [BLOCK_M, BLOCK_D]
        acc = acc + tl.dot(beta, V_tile.to(tl.float32))

        # Update sum-of-exponentials
        l_i = l_i * alpha + tl.sum(beta, axis=1)
        m_i = m_i_new

    # ------------------------------------------------------------------
    # Normalise and write-back
    # ------------------------------------------------------------------
    acc = acc / l_i[:, None]

    # Store O  (bf16)
    o_offs = b * stride_ob + h * stride_oh + off_m[:, None] * stride_os + off_d[None, :] * stride_od
    tl.store(O + o_offs, acc.to(tl.bfloat16), mask=q_mask)

    # Store LSE = max(P) + log(sum(exp(P - max(P))))  (fp32)
    lse_offs = b * stride_lb + h * stride_lh + off_m * stride_ls
    lse = m_i + tl.log(l_i)
    tl.store(LSE + lse_offs, lse, mask=(off_m < S))


def run(Q, K, V, O, LSE):
    """Non-causal MHA forward: O = softmax(Q@K^T / sqrt(D)) @ V, plus LSE."""
    torch.cuda.set_device(Q.device)

    B, H, S, D = Q.shape
    scale = 1.0 / float(D ** 0.5)

    BLOCK_M = 64
    BLOCK_N = 32
    BLOCK_D = 128          # matches const D=128

    grid = (triton.cdiv(S, BLOCK_M), B * H)

    _mha_forward[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        B, H, S, D,
        scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=2,
    )