import torch
import triton
import triton.language as tl


@triton.jit
def _mha_forward_kernel(
    Q, K, V, O, LSE,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    B, H, S, D,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Forward multi-head attention kernel with LSE output.

    Each program handles one (batch, head) pair and one tile of query positions.
    Iterates over key/value sequence tiles with online softmax accumulation.
    """
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    bid_b = pid_bh // H
    bid_h = pid_bh % H

    # Advance base pointers to this (b, h) slice
    Q += bid_b * stride_qb + bid_h * stride_qh
    K += bid_b * stride_kb + bid_h * stride_kh
    V += bid_b * stride_vb + bid_h * stride_vh
    O += bid_b * stride_ob + bid_h * stride_oh
    LSE += bid_b * stride_lseb + bid_h * stride_lseh

    # Index arrays (compile-time shapes)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = tl.arange(0, BLOCK_N)                     # [BLOCK_N]
    offs_d = tl.arange(0, BLOCK_D)                     # [BLOCK_D]

    # ----- Load Q tile once (outside N loop) -----
    q_ptrs = Q + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    q_mask = (offs_m[:, None] < S) & (offs_d[None, :] < D)
    Q_tile = tl.load(q_ptrs, mask=q_mask, other=0.0)   # [BLOCK_M, BLOCK_D] bf16

    # ----- Initialize online softmax state -----
    acc_o = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)  # running max
    l_i = tl.full((BLOCK_M,), 1.0, dtype=tl.float32)            # running sum

    # ----- Loop over N (key sequence) tiles -----
    for start_n in range(0, S, BLOCK_N):
        n_offs = start_n + offs_n               # [BLOCK_N]
        n_mask = n_offs < S                      # [BLOCK_N]

        # Load K tile [BLOCK_N, BLOCK_D], keep bf16 for dot with Q
        k_ptrs = K + n_offs[:, None] * stride_ks + offs_d[None, :] * stride_kd
        k_mask = n_mask[:, None] & (offs_d[None, :] < D)
        K_tile = tl.load(k_ptrs, mask=k_mask, other=0.0)

        # Scores = Q @ K^T / sqrt(D), both bf16 -> fp32 accumulator
        scores = tl.dot(Q_tile, K_tile.T) * scale           # [BLOCK_M, BLOCK_N] fp32
        # Mask out-of-bound N positions to -inf
        scores = tl.where(n_mask[None, :], scores, float("-inf"))

        # --- Online softmax update ---
        m_ij = tl.max(scores, axis=1)                       # [BLOCK_M]
        m_new = tl.maximum(m_i, m_ij)                        # [BLOCK_M]

        alpha = tl.exp(m_i - m_new)                          # [BLOCK_M]
        p = tl.exp(scores - m_new[:, None])                  # [BLOCK_M, BLOCK_N] fp32
        l_new = alpha * l_i + tl.sum(p, axis=1)              # [BLOCK_M]

        # Scale previous accumulator
        acc_o = acc_o * alpha[:, None]

        # Load V tile [BLOCK_N, BLOCK_D], cast to fp32 for dot with fp32 probs
        v_ptrs = V + n_offs[:, None] * stride_vs + offs_d[None, :] * stride_vd
        v_mask = n_mask[:, None] & (offs_d[None, :] < D)
        V_tile = tl.load(v_ptrs, mask=v_mask, other=0.0).to(tl.float32)

        # Accumulate: acc += softmax(scores) @ V, both fp32
        acc_o = acc_o + tl.dot(p, V_tile)                    # [BLOCK_M, BLOCK_D] fp32

        m_i = m_new
        l_i = l_new

    # ----- Normalize and store O -----
    acc_o = acc_o / l_i[:, None]                             # [BLOCK_M, BLOCK_D] fp32
    o_ptrs = O + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    o_mask = (offs_m[:, None] < S) & (offs_d[None, :] < D)
    tl.store(o_ptrs, acc_o.to(tl.bfloat16), mask=o_mask)

    # ----- Store LSE = max(P) + log(sum(exp(P - max(P)))) -----
    lse_vals = m_i + tl.log(l_i)                             # [BLOCK_M] fp32
    lse_ptrs = LSE + offs_m * stride_lses
    tl.store(lse_ptrs, lse_vals, mask=offs_m < S)


def run(Q, K, V, O, LSE):
    """Compute non-causal multi-head attention forward pass with LSE.

    O = softmax(Q @ K^T / sqrt(D)) @ V      (bf16, [B, H, S, D])
    LSE = logsumexp(Q @ K^T / sqrt(D))       (fp32, [B, H, S])

    Arguments
    ---------
    Q, K, V : bf16 [B, H, S, D] -- preallocated inputs
    O       : bf16 [B, H, S, D] -- preallocated output for attention result
    LSE     : fp32 [B, H, S]    -- preallocated output for log-sum-exp
    """
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = 128  # D is constant 128

    scale = 1.0 / (D ** 0.5)

    # Grid: [B*H] x [ceil(S / BLOCK_M)]
    grid = (B * H, triton.cdiv(S, BLOCK_M))

    _mha_forward_kernel[grid](
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
        num_stages=3,
    )