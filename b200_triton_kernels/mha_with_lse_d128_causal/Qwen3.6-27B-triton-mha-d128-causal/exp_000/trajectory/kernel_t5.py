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
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    bid = pid_bh // H
    hid = pid_bh % H

    scale = 1.0 / tl.sqrt(D.to(tl.float32))

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    # Load Q tile once
    q_base = Q + bid * stride_qb + hid * stride_qh
    q_offs_m = offs_m[:, None]                      # [BLOCK_M, 1]
    q_offs_d = offs_d[None, :]                       # [1, BLOCK_D]
    q_ptrs = q_base + q_offs_m * stride_qs + q_offs_d * stride_qd
    q_mask = q_offs_m < S                            # [BLOCK_M, 1]
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # Initial softmax state using finite sentinel
    NEG_INF_SENTINEL = -1e20
    row_m = tl.full([BLOCK_M], NEG_INF_SENTINEL, dtype=tl.float32)
    row_sum = tl.full([BLOCK_M], 0.0, dtype=tl.float32)
    o_accum = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    k_base = K + bid * stride_kb + hid * stride_kh
    v_base = V + bid * stride_vb + hid * stride_vh

    num_kv_tiles = tl.cdiv(S, BLOCK_N)
    for tile_idx in range(num_kv_tiles):
        start_n = tile_idx * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)

        # Mask shapes: causal [BLOCK_M,BLOCK_N], n_valid_broadcasted [BLOCK_N,BLOCK_D]
        n_valid_scalar = offs_n < S                          # [BLOCK_N]
        causal_mask = (offs_m[:, None] >= offs_n[None, :])   # [BLOCK_M, BLOCK_N]
        kv_load_mask = n_valid_scalar[:, None]               # [BLOCK_N, 1]

        # Load K: shape [BLOCK_N, BLOCK_D]
        k_ptrs = k_base + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=kv_load_mask, other=0.0)

        # QK^T scores: [BLOCK_M, BLOCK_N], fp32
        scores = tl.dot(q, k.T) * scale

        # Create attention mask [BLOCK_M, BLOCK_N]
        attn_mask = causal_mask & n_valid_scalar[None, :]

        # Apply mask
        scores = tl.where(attn_mask, scores, NEG_INF_SENTINEL)

        # New max per query row
        new_max = tl.max(scores, axis=1)                     # [BLOCK_M]

        # Center and compute probabilities
        scores_centered = scores - new_max[:, None]           # [BLOCK_M, BLOCK_N]
        probs = tl.exp(scores_centered)
        probs = tl.where(attn_mask, probs, 0.0)              # [BLOCK_M, BLOCK_N]

        block_sum = tl.sum(probs, axis=1)                    # [BLOCK_M]

        # Compute old_scale = exp(row_m - new_max), clamp to avoid underflow
        diff = row_m - new_max                               # [BLOCK_M]
        diff_clamped = tl.maximum(diff, -50.0)
        old_scale = tl.exp(diff_clamped)                     # [BLOCK_M]

        # Update running stats
        row_sum = old_scale * row_sum + block_sum            # [BLOCK_M]
        row_m = tl.maximum(row_m, new_max)                   # [BLOCK_M]

        # Scale down old accumulator
        o_accum = o_accum * old_scale[:, None]               # [BLOCK_M, BLOCK_D]

        # Load V and accumulate
        v_ptrs = v_base + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=kv_load_mask, other=0.0)   # [BLOCK_N, BLOCK_D]

        p_f16 = probs.to(tl.bfloat16)                        # [BLOCK_M, BLOCK_N]
        o_new = tl.dot(p_f16, v)                             # [BLOCK_M, BLOCK_D]
        o_accum = o_accum + o_new

    # Normalize
    inv_denom = 1.0 / row_sum                                # [BLOCK_M]
    o_final = o_accum * inv_denom[:, None]                   # [BLOCK_M, BLOCK_D]

    # Store O
    o_base = O + bid * stride_ob + hid * stride_oh
    o_ptrs = o_base + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(o_ptrs, o_final.to(tl.bfloat16), mask=offs_m[:, None] < S)

    # Store LSE
    lse_val = row_m + tl.log(row_sum)                        # [BLOCK_M]
    lse_base = LSE + bid * stride_lse_b + hid * stride_lse_h
    lse_ptrs = lse_base + offs_m * stride_lse_s
    tl.store(lse_ptrs, lse_val, mask=offs_m < S)


def run(Q, K, V, O, LSE):
    """Causal multi-head attention forward with LSE output."""
    torch.cuda.set_device(Q.device)

    B, H, S, D = Q.shape

    BLOCK_M = 64
    BLOCK_N = 64

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
        num_warps=4,
        num_stages=3,
    )