import torch
import triton
import triton.language as tl


@triton.jit
def _causal_mha_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    lse_ptr,
    stride_s,
    stride_lse_s,
    stride_bh,
    stride_lse_bh,
    S,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    bh_id = tl.program_id(0)
    pid_m = tl.program_id(1)

    # Base pointer offsets for this (batch, head) group
    q_off = bh_id * stride_bh
    k_off = bh_id * stride_bh
    v_off = bh_id * stride_bh
    o_off = bh_id * stride_bh
    lse_off = bh_id * stride_lse_bh

    # Row indices for this query tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < S

    # Head-dimension offsets
    offs_d = tl.arange(0, BLOCK_D)

    # Scale factor for attention scores
    scale = 1.0 / tl.sqrt(tl.cast(BLOCK_D, tl.float32))

    # Load Q tile: [BLOCK_M, BLOCK_D]
    q_ptrs = q_ptr + q_off + offs_m[:, None] * stride_s + offs_d[None, :]
    q_tile = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    # Online softmax accumulators (fp32 for stability)
    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    # Iterate over key/value tiles along the sequence dimension
    num_steps = tl.cdiv(S, BLOCK_N)
    for step in range(num_steps):
        start_n = step * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S

        # Load K tile: [BLOCK_D, BLOCK_N]  (logical K^T)
        k_ptrs = k_ptr + k_off + offs_n[None, :] * stride_s + offs_d[:, None]
        k_tile = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0)

        # Load V tile: [BLOCK_N, BLOCK_D]
        v_ptrs = v_ptr + v_off + offs_n[:, None] * stride_s + offs_d[None, :]
        v_tile = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

        # Attention scores: [BLOCK_M, BLOCK_N]
        scores = tl.dot(q_tile, k_tile) * scale

        # Causal mask: query_pos >= key_pos (lower-triangular including diagonal)
        causal = offs_m[:, None] >= offs_n[None, :]
        valid = causal & mask_m[:, None] & mask_n[None, :]

        # Mask invalid positions to -inf before softmax
        masked_scores = tl.where(valid, scores, -float("inf"))

        # Online softmax state update
        new_m = tl.maximum(m_i, tl.max(masked_scores, axis=1))

        alpha = tl.exp(m_i - new_m)

        p = tl.exp(masked_scores - new_m[:, None])

        new_l = alpha * l_i + tl.sum(p, axis=1)

        # Weighted accumulation: acc = alpha * acc + P @ V
        acc = acc * alpha[:, None] + tl.dot(p, v_tile)

        m_i = new_m
        l_i = new_l

    # Final normalization: O = acc / l_i
    l_safe = tl.where(l_i > 0.0, l_i, 1.0)
    acc = acc / l_safe[:, None]

    # Store output O: [BLOCK_M, BLOCK_D] -> bfloat16
    o_ptrs = o_ptr + o_off + offs_m[:, None] * stride_s + offs_d[None, :]
    tl.store(o_ptrs, acc.to(tl.bfloat16), mask=mask_m[:, None])

    # Store LSE: [BLOCK_M] -> float32
    lse_val = m_i + tl.log(l_safe)
    lse_ptrs = lse_ptr + lse_off + offs_m
    tl.store(lse_ptrs, lse_val, mask=mask_m)


def run(Q, K, V, O, LSE):
    """Causal multi-head attention forward returning O and LSE.

    Computes O = softmax(Q @ K^T / sqrt(D), causal) @ V and
    LSE = logsumexp(Q @ K^T / sqrt(D), causal).

    Q, K, V: (B, H, S, D) bfloat16
    O: (B, H, S, D) bfloat16 (preallocated)
    LSE: (B, H, S) float32 (preallocated)
    """
    torch.cuda.set_device(Q.device)

    B, H, S, D = Q.shape

    # Block dimensions tuned for Hopper with D=128
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = D  # Always 128 for this task; passed as constexpr

    # Strides for correct (B, H, S, D) layout navigation
    stride_s = D            # stride along sequence axis (axis 2) = D
    stride_bh = S * D       # stride to advance one (B,H) group
    stride_lse_s = 1        # LSE last axis is contiguous
    stride_lse_bh = S       # stride per (B,H) group in LSE [B,H,S]

    BH = B * H
    num_pid_m = triton.cdiv(S, BLOCK_M)
    grid = (BH, num_pid_m)

    _causal_mha_kernel[grid](
        Q, K, V, O, LSE,
        stride_s, stride_lse_s, stride_bh, stride_lse_bh,
        S,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=3,
    )