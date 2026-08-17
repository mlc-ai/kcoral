import torch
import triton
import triton.language as tl


@triton.jit
def _mha_kernel(
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
    """Optimized MHA forward kernel targeting Hopper WGMMA."""
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh % H

    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_d = tl.arange(0, BLOCK_D)
    off_n = tl.arange(0, BLOCK_N)

    # Precompute batch-head offsets
    q_base = Q + b * stride_qb + h * stride_qh
    k_base = K + b * stride_kb + h * stride_kh
    v_base = V + b * stride_vb + h * stride_vh
    o_base = O + b * stride_ob + h * stride_oh
    lse_base = LSE + b * stride_lb + h * stride_lh

    # Load Q tile [BLOCK_M, BLOCK_D] - loaded once per program
    q_ptrs = off_m[:, None] * stride_qs + off_d[None, :] * stride_qd
    q_mask = (off_m[:, None] < S) & (off_d[None, :] < D)
    Q_tile = tl.load(q_base + q_ptrs, mask=q_mask, other=0.0, eviction_policy="evict_last")

    # Online softmax accumulators
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.full((BLOCK_M,), 1.0, dtype=tl.float32)

    # Number of K/V tiles to iterate
    num_k_steps = tl.cdiv(S, BLOCK_N)

    for sn in range(num_k_steps):
        cur_n = sn * BLOCK_N + off_n

        # Load K tile [BLOCK_N, BLOCK_D]
        k_ptrs = cur_n[:, None] * stride_ks + off_d[None, :] * stride_kd
        k_mask = (cur_n[:, None] < S) & (off_d[None, :] < D)
        K_tile = tl.load(k_base + k_ptrs, mask=k_mask, other=0.0)

        # Compute attention scores [BLOCK_M, BLOCK_N]
        scores = tl.dot(Q_tile, K_tile.T) * SCALE

        # Apply causal-like masking for out-of-bounds positions
        seq_mask = (off_m[:, None] < S) & (cur_n[None, :] < S)
        scores = tl.where(seq_mask, scores, float("-inf"))

        # Online softmax update
        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])

        acc = acc * alpha[:, None]

        # Load V tile [BLOCK_N, BLOCK_D]
        v_ptrs = cur_n[:, None] * stride_vs + off_d[None, :] * stride_vd
        v_mask = (cur_n[:, None] < S) & (off_d[None, :] < D)
        V_tile = tl.load(v_base + v_ptrs, mask=v_mask, other=0.0)

        acc += tl.dot(p, V_tile.to(tl.float32))

        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    # Normalize and store output
    acc = acc / l_i[:, None]

    o_ptrs = off_m[:, None] * stride_os + off_d[None, :] * stride_od
    tl.store(o_base + o_ptrs, acc.to(tl.bfloat16), mask=q_mask)

    # Store LSE
    lse_ptrs = off_m * stride_ls
    tl.store(lse_base + lse_ptrs, m_i + tl.log(l_i), mask=(off_m < S))


def run(Q, K, V, O, LSE):
    """Non-causal MHA forward: O = softmax(Q@K^T / sqrt(D)) @ V, plus LSE."""
    torch.cuda.set_device(Q.device)

    B, H, S, D = Q.shape
    scale = 1.0 / float(D ** 0.5)

    # Optimized tile configuration for Hopper
    BLOCK_M = 128
    BLOCK_N = 32
    BLOCK_D = 128  # Full D dimension loaded at once

    grid = (triton.cdiv(S, BLOCK_M), B * H)

    _mha_kernel[grid](
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