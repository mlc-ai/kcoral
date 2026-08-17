import torch
import triton
import triton.language as tl


@triton.jit
def _mha_causal_kernel(
    Q, K, V, O, LSE,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lse_b, stride_lse_h, stride_lse_s,
    B, H, S, D,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Flash-attention-style causal softmax kernel per (batch, head) pair."""
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    bid = pid_bh // H
    hid = pid_bh % H

    scale = 1.0 / tl.sqrt(D.to(tl.float32))

    # Block offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    offs_d = tl.arange(0, BLOCK_D)                      # [BLOCK_D]

    # Load Q tile once
    q_base = Q + bid * stride_qb + hid * stride_qh
    q_ptrs = q_base + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=offs_m[:, None] < S, other=0.0)

    # Initial softmax state
    # Using large finite sentinel instead of -inf to avoid NaN from (-inf)-(-inf)
    NEG_INF_SENTINEL = -1e10
    row_m = tl.full([BLOCK_M], NEG_INF_SENTINEL, dtype=tl.float32)
    row_logsumexp = tl.full([BLOCK_M], 0.0, dtype=tl.float32)
    o_accum = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    k_base = K + bid * stride_kb + hid * stride_kh
    v_base = V + bid * stride_vb + hid * stride_vh

    num_kv_tiles = tl.cdiv(S, BLOCK_N)
    for tile_idx in range(num_kv_tiles):
        start_n = tile_idx * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)           # [BLOCK_N]

        # Causal + sequence mask
        n_in_bounds = (offs_n < S)                          # [BLOCK_N]
        causal = (offs_m[:, None] >= offs_n[None, :])       # [BLOCK_M, BLOCK_N]
        attn_mask = causal & n_in_bounds[None, :]           # [BLOCK_M, BLOCK_N]

        # Load K
        k_ptrs = k_base + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=n_in_bounds[None, :], other=0.0)  # [BLOCK_N, BLOCK_D]

        # QK^T dot product, scaled
        scores = tl.dot(q, k.T) * scale                     # [BLOCK_M, BLOCK_N], fp32

        # Apply mask: unmasked positions get NEG_INF_SENTINEL
        scores = tl.where(attn_mask, scores, NEG_INF_SENTINEL)

        # New max per query row
        new_max = tl.max(scores, axis=1)                    # [BLOCK_M]

        # Determine whether this block contributes any valid probability
        # (i.e., at least one key position is causally valid for this query row)
        row_has_valid_keys = new_max > NEG_INF_SENTINEL     # [BLOCK_M]

        # Compute softmax probabilities safely:
        # Subtract new_max (always defined, no inf issues since we use finite sentinel)
        scores_centered = scores - new_max[:, None]         # [BLOCK_M, BLOCK_N]
        # Clamp to prevent overflow from extremely large centered scores
        scores_clamped = tl.minimum(scores_centered, 0.0)
        probs = tl.exp(scores_clamped)                       # [BLOCK_M, BLOCK_N], exp(0)=1 max
        # Zero out invalid (masked) positions after exponentiation
        probs = tl.where(attn_mask, probs, 0.0)

        block_sum = tl.sum(probs, axis=1)                   # [BLOCK_M]

        # Rescaling factor: exp(old_max - new_max)
        # If old_max was the sentinel (-1e10) and new_max is real, exp yields ~0.0
        # If both are sentinel, exp(0)=1 but block_sum=0 so no net change
        # We clamp negative args to avoid extreme underflow
        max_diff = row_m - new_max                           # [BLOCK_M]
        max_diff_clamped = tl.maximum(max_diff, -50.0)      # clamp exp arg
        old_scale = tl.exp(max_diff_clamped)                 # [BLOCK_M]

        # Update running stats
        row_logsumexp = old_scale * row_logsumexp + block_sum  # [BLOCK_M]
        row_m = tl.maximum(row_m, new_max)                    # [BLOCK_M]

        # Scale down old accumulator and add new contribution
        o_accum = o_accum * old_scale[:, None]               # [BLOCK_M, BLOCK_D]

        # Accumulate weighted V
        v_ptrs = v_base + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=n_in_bounds[None, :], other=0.0)  # [BLOCK_N, BLOCK_D]

        p_bf16 = probs.to(tl.bfloat16)                       # [BLOCK_M, BLOCK_N]
        o_accum = o_accum + tl.dot(p_bf16, v)               # [BLOCK_M, BLOCK_D]

    # Normalize output
    inv_denom = 1.0 / row_logsumexp                          # [BLOCK_M]
    o_final = o_accum * inv_denom[:, None]                   # [BLOCK_M, BLOCK_D]

    # Store output O
    o_base = O + bid * stride_ob + hid * stride_oh
    o_ptrs = o_base + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(o_ptrs, o_final.to(tl.bfloat16), mask=offs_m[:, None] < S)

    # Store LSE = row_m + log(row_logsumexp)
    lse_val = row_m + tl.log(row_logsumexp)                  # [BLOCK_M]
    lse_base = LSE + bid * stride_lse_b + hid * stride_lse_h
    lse_ptrs = lse_base + offs_m * stride_lse_s
    tl.store(lse_ptrs, lse_val, mask=offs_m < S)


def run(Q, K, V, O, LSE):
    """Causal multi-head attention forward with LSE output."""
    torch.cuda.set_device(Q.device)

    B, H, S, D = Q.shape

    BLOCK_M = 64
    BLOCK_N = 64
    num_warps = 4
    num_stages = 3

    grid = (triton.cdiv(S, BLOCK_M), B * H)

    _mha_causal_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        B, H, S, D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=D,
        num_warps=num_warps,
        num_stages=num_stages,
    )