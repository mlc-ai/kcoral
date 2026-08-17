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

    scale = 1.0 / tl.sqrt(tl.cast(D, tl.float32))

    # Lane offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    offs_d = tl.arange(0, BLOCK_D)                       # [BLOCK_D]

    # --- Load Q once [BLOCK_M, BLOCK_D] ---
    q_base = Q + bid * stride_qb + hid * stride_qh
    q_offsets = offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    q = tl.load(q_base + q_offsets, mask=offs_m[:, None] < S, other=0.0)

    # --- Softmax state per query row [BLOCK_M] ---
    # Start with zero-sum; accumulate into a finite-running-max scheme.
    sum_exp = tl.zeros([BLOCK_M], dtype=tl.float32)
    max_score = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc_out = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    seen_any = tl.zeros([BLOCK_M], dtype=tl.int1)

    k_base = K + bid * stride_kb + hid * stride_kh
    v_base = V + bid * stride_vb + hid * stride_vh

    # Number of KV tiles (runtime-safe via tl.cdiv in range)
    num_kv_tiles = tl.cdiv(S, BLOCK_N)

    for kv_tile in tl.range(num_kv_tiles, num_stages=0):
        start_n = kv_tile * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)           # [BLOCK_N]

        # Build masks
        n_ok = offs_n < S                                   # [BLOCK_N]
        causal = offs_m[:, None] >= offs_n[None, :]         # [BLOCK_M, BLOCK_N]

        # Load K [BLOCK_N, BLOCK_D]
        k_ptrs = k_base + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=n_ok[:, None], other=0.0)

        # Q @ K^T [BLOCK_M, BLOCK_N] fp32
        scores = tl.dot(q, k.T) * scale

        # Mask: causal AND in-bounds.  Expand n_ok to [1,BLOCK_N] for & with [BLOCK_M,BLOCK_N]
        m = causal & n_ok[None, :]                          # [BLOCK_M, BLOCK_N]

        # Set invalid entries to very negative value
        scores = tl.where(m, scores, -1e20)

        # Per-row max in this block
        block_max = tl.max(scores, axis=1)                  # [BLOCK_M]

        # Do any queries in this row have valid keys?
        valid = block_max > -1e19                            # [BLOCK_M]

        # Centered scores for exp (avoid inf-inf NaN)
        center = tl.where(valid, block_max, 0.0)            # [BLOCK_M]
        centered = scores - center[:, None]                  # [BLOCK_M, BLOCK_N]
        probs = tl.exp(centered)                             # [BLOCK_M, BLOCK_N]
        probs = tl.where(m, probs, 0.0)                      # kill invalid

        block_sum = tl.sum(probs, axis=1)                    # [BLOCK_M]

        # --- Online softmax recurrence ---
        # old_scale = exp(old_max - new_max)
        # Use previous max; clamp exponent to [-50, 0]
        old_minus_new = tl.where(valid, max_score - block_max, 0.0)
        old_minus_new = tl.clamp(old_minus_new, -50.0, 0.0)
        alpha = tl.exp(old_minus_new)                        # [BLOCK_M]

        # Recompute centered sums relative to old max
        # scaled_block = alpha * block_sum (already computed above)
        new_sum = alpha * sum_exp + block_sum                # [BLOCK_M]

        # New global max for each row
        new_max = tl.where(valid, tl.maximum(max_score, block_max), max_score)

        # Scale old output accumulation down
        scale_vec = alpha[:, None]                           # [BLOCK_M, 1]
        acc_out = acc_out * scale_vec

        # Add weighted V
        v_ptrs = v_base + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=n_ok[:, None], other=0.0)
        pv = tl.dot(probs.to(tl.bfloat16), v)               # [BLOCK_M, BLOCK_D]
        acc_out = acc_out + pv

        # Update state
        sum_exp = new_sum                                    # [BLOCK_M]
        max_score = new_max                                  # [BLOCK_M]
        seen_any = seen_any | valid                           # [BLOCK_M]

    # --- Normalize output ---
    inv_sum = tl.where(seen_any, 1.0 / sum_exp, 0.0)        # [BLOCK_M]
    out_val = acc_out * inv_sum[:, None]                     # [BLOCK_M, BLOCK_D]

    # Store O [B,H,S,D] bf16
    o_base = O + bid * stride_ob + hid * stride_oh
    o_ptrs = o_base + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(o_ptrs, out_val.to(tl.bfloat16), mask=offs_m[:, None] < S)

    # Store LSE [B,H,S] fp32
    lse = tl.where(seen_any, max_score + tl.log(sum_exp), float("-inf"))
    lse_base = LSE + bid * stride_lse_b + hid * stride_lse_h
    lse_ptrs = lse_base + offs_m * stride_lse_s
    tl.store(lse_ptrs, lse, mask=offs_m < S)


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