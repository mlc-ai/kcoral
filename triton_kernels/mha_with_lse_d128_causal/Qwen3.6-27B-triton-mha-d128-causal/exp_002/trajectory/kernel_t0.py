import torch
import triton
import triton.language as tl


@triton.jit
def _mha_causal_fwd(
    Q, K, V,
    O, LSE,
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
    """
    Causal multi-head attention forward kernel with log-sum-exp output.

    Each program instance handles one (batch, head) combination and a block of
    BLOCK_M query rows.  Inside, we iterate over BLOCK_N-sized key blocks,
    maintaining an online softmax accumulator so the output is numerically
    equivalent to a full softmax-at-once computation.
    """
    pid_bh = tl.program_id(0)
    pid_m  = tl.program_id(1)

    pid_b = pid_bh // H
    pid_h = pid_bh % H

    # ---- index helpers --------------------------------------------------
    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    off_d = tl.arange(0, D)                            # [D]

    # ---- load Q tile  [BLOCK_M, D] --------------------------------------
    q_off = (pid_b * stride_qb + pid_h * stride_qh +
             off_m[:, None] * stride_qs + off_d[None, :] * stride_qd)
    q = tl.load(Q + q_off, mask=(off_m[:, None] < S), other=0.0)

    # ---- online softmax state -------------------------------------------
    acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)

    num_k_tiles = tl.cdiv(S, BLOCK_N)

    # ---- iterate over key / value tiles ---------------------------------
    for start_n in range(num_k_tiles):
        off_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)

        # Load K tile  [BLOCK_N, D]
        k_off = (pid_b * stride_kb + pid_h * stride_kh +
                 off_n[:, None] * stride_ks + off_d[None, :] * stride_kd)
        k = tl.load(K + k_off, mask=(off_n[:, None] < S), other=0.0)

        # Q @ K^T  ->  [BLOCK_M, BLOCK_N]
        qk = tl.dot(q, k.T)
        qk = qk * softmax_scale

        # Causal + out-of-bounds mask
        q_pos = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        k_pos = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
        causal = q_pos[:, None] >= k_pos[None, :]
        valid  = (off_m[:, None] < S) & (off_n[None, :] < S)
        qk = tl.where(causal & valid, qk, float("-inf"))

        # --- online softmax update ---------------------------------------
        m_ij      = tl.max(qk, axis=1)
        m_i_new   = tl.maximum(m_i, m_ij)
        alpha     = tl.exp(m_i - m_i_new)          # [BLOCK_M]
        acc       = acc * alpha[:, None]            # scale old partial sums
        p         = tl.exp(qk - m_i_new[:, None])   # [BLOCK_M, BLOCK_N]

        # Load V tile  [BLOCK_N, D]
        v_off = (pid_b * stride_vb + pid_h * stride_vh +
                 off_n[:, None] * stride_vs + off_d[None, :] * stride_vd)
        v = tl.load(V + v_off, mask=(off_n[:, None] < S), other=0.0)
        v = v.to(tl.float32)

        # acc += P @ V
        acc = tl.dot(p, v, acc)

        # normalisation running sum
        l_i = alpha * l_i + tl.sum(p, axis=1)
        m_i = m_i_new

    # ---- store output O  [BLOCK_M, D] -----------------------------------
    o_off = (pid_b * stride_ob + pid_h * stride_oh +
             off_m[:, None] * stride_os + off_d[None, :] * stride_od)
    tl.store(O + o_off, acc.to(tl.bfloat16), mask=(off_m[:, None] < S))

    # ---- store LSE  [BLOCK_M] -------------------------------------------
    lse_off = pid_b * stride_lsb + pid_h * stride_lsh + off_m * stride_ls
    lse_val = m_i + tl.log(l_i)
    tl.store(LSE + lse_off, lse_val, mask=(off_m < S))


def run(Q, K, V, O, LSE):
    """Compute causal MHA forward (O, LSE) into pre-allocated output tensors."""
    torch.cuda.set_device(Q.device)

    B, H, S, D = Q.shape
    softmax_scale = 1.0 / (D ** 0.5)

    BLOCK_M = 64
    BLOCK_N = 64

    grid = (B * H, triton.cdiv(S, BLOCK_M))

    _mha_causal_fwd[grid](
        Q, K, V,
        O, LSE,
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
        num_stages=2,
    )