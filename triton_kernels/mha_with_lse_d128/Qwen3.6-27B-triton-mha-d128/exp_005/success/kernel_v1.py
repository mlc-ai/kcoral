import torch
import triton
import triton.language as tl


@triton.jit
def _mha_fwd_kernel(
    Q, K, V,
    Out, Lse,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lb, stride_lh, stride_ls,
    num_heads,
    seq_len,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
):
    """FlashAttention-style multi-head attention forward kernel.

    Computes O = softmax(QK^T / sqrt(D)) V and LSE using online softmax,
    avoiding materialization of the full S×S attention matrix.
    Optimized for NVIDIA Hopper with WGMMA-friendly tile sizes.
    """
    pid_zh = tl.program_id(0)
    start_m = tl.program_id(1)

    off_b = pid_zh // num_heads
    off_h = pid_zh % num_heads

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_SIZE)

    q_valid = offs_m < seq_len

    # Base pointers for the current (batch, head) slice
    Q_base = Q + off_b * stride_qb + off_h * stride_qh
    K_base = K + off_b * stride_kb + off_h * stride_kh
    V_base = V + off_b * stride_vb + off_h * stride_vh
    O_base = Out + off_b * stride_ob + off_h * stride_oh
    LSE_base = Lse + off_b * stride_lb + off_h * stride_lh

    # Load Q tile once – used across all KV-sequence blocks
    q_ptrs = Q_base + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    Q_tile = tl.load(q_ptrs, mask=q_valid[:, None], other=0.0)
    dtype = Q_tile.dtype

    # Online-softmax running state (FP32 for numerical quality)
    m_i = tl.full([BLOCK_M], value=float('-inf'), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], value=1.0, dtype=tl.float32)
    acc_o = tl.zeros([BLOCK_M, HEAD_SIZE], dtype=tl.float32)

    num_steps = tl.cdiv(seq_len, BLOCK_N)

    for step in range(num_steps):
        n_idx = step * BLOCK_N + offs_n
        n_valid = n_idx < seq_len

        # Load K and V tiles
        k_ptrs = K_base + n_idx[:, None] * stride_ks + offs_d[None, :] * stride_kd
        K_tile = tl.load(k_ptrs, mask=n_valid[:, None], other=0.0)

        v_ptrs = V_base + n_idx[:, None] * stride_vs + offs_d[None, :] * stride_vd
        V_tile = tl.load(v_ptrs, mask=n_valid[:, None], other=0.0)

        # Attention scores [BLOCK_M, BLOCK_N] in FP32
        scores = tl.dot(Q_tile, K_tile.T) * scale

        # Pad-mask: zero-out scores where KV index exceeds sequence length
        scores = tl.where(n_valid[None, :], scores, float('-inf'))

        # --- Online softmax recurrence ---
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        p = tl.exp(scores - m_ij[:, None])

        alpha = tl.exp(m_i - m_ij)
        acc_o = acc_o * alpha[:, None] + tl.dot(p.to(dtype), V_tile)

        l_ij = tl.sum(p, axis=1)
        l_i = l_i * alpha + l_ij
        m_i = m_ij

    # Epilogue: normalize and store results
    o_final = acc_o / l_i[:, None]

    out_ptrs = O_base + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(out_ptrs, o_final.to(dtype), mask=q_valid[:, None])

    lse_ptrs = LSE_base + offs_m * stride_ls
    tl.store(lse_ptrs, m_i + tl.log(l_i), mask=q_valid)


def run(Q, K, V, O, LSE):
    """Destination-passing entry point.

    Args:
        Q, K, V : (B, H, S, D) bfloat16 inputs.
        O       : (B, H, S, D) bfloat16 preallocated output tensor.
        LSE     : (B, H, S)   float32   preallocated output tensor.
    """
    torch.cuda.set_device(Q.device)

    B, H, S, D = Q.shape
    scale = 1.0 / float(D ** 0.5)

    # Hopper-optimized tile configuration for BF16 D=128:
    #   BLOCK_M=128 → better WGMMA throughput (M≥64 multiple);
    #   BLOCK_N=128 → fewer dot iterations, amortized scheduling cost;
    #   num_warps=8 → 2 warp-groups for Hopper WGMMA requirement;
    #   num_stages=2 → balances shared-memory (128×128×2×2×2 ≈ 128 KiB) vs latency hiding.
    BLOCK_M = 128
    BLOCK_N = 128
    num_warps = 8
    num_stages = 2

    # Head-major grid improves L2 locality: each SM processes a whole
    # (batch, head) before migrating, keeping that head's Q/K/V cached.
    grid = (B * H, triton.cdiv(S, BLOCK_M))

    _mha_fwd_kernel[grid](
        Q, K, V,
        O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        H,
        S,
        scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        HEAD_SIZE=D,
        num_warps=num_warps,
        num_stages=num_stages,
    )