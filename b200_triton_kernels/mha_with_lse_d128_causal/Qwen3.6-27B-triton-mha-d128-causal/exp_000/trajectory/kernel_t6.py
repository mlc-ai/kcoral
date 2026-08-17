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

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    offs_d = tl.arange(0, BLOCK_D)                       # [BLOCK_D]

    # Load Q tile once  [BLOCK_M, BLOCK_D]
    q_base = Q + bid * stride_qb + hid * stride_qh
    q_ptrs = q_base + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    q_mask_row = offs_m[:, None] < S                     # [BLOCK_M, 1]
    q = tl.load(q_ptrs, mask=q_mask_row, other=0.0)

    # Each query row needs its own running softmax state.
    # We track whether each row has seen any valid (causal) key yet.
    row_m = tl.zeros([BLOCK_M], dtype=tl.float32)       # running max of scores
    row_lse = tl.zeros([BLOCK_M], dtype=tl.float32)     # running log-sum-exp component
    o_accum = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    # flag_i[r] == 1 iff query row r has seen at least one valid key so far
    flag_i = tl.zeros([BLOCK_M], dtype=tl.int1)

    k_base = K + bid * stride_kb + hid * stride_kh
    v_base = V + bid * stride_vb + hid * stride_vh

    num_kv_tiles = tl.cdiv(S, BLOCK_N)
    for idx in range(num_kv_tiles):
        start_n = idx * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)        # [BLOCK_N]

        # Causal mask: [BLOCK_M, BLOCK_N]  true when n <= m
        causal = (offs_m[:, None] >= offs_n[None, :])
        # In-bounds mask for keys: [BLOCK_N, 1]
        n_in_bounds = offs_n[:, None] < S
        # Combined mask for attention scores: [BLOCK_M, BLOCK_N]
        attn_mask = causal & n_in_bounds[None, :, ]

        # Load K  [BLOCK_N, BLOCK_D]
        k_ptrs = k_base + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=n_in_bounds, other=0.0)

        # Scores  [BLOCK_M, BLOCK_N]  (fp32 via Tensor Core)
        scores = tl.dot(q, k.T) * scale

        # Mask invalid positions to -inf
        scores = tl.where(attn_mask, scores, float("-inf"))

        # Per-row max
        m_ij = tl.max(scores, axis=1)                    # [BLOCK_M]
        # Does this row have any valid key in this block?
        has_valid = m_ij > float("-inf")                 # [BLOCK_M]

        # Safe centering: subtract m_ij where valid, else subtract 0
        m_center = tl.where(has_valid, m_ij, 0.0)
        centered = scores - m_center[:, None]            # [BLOCK_M, BLOCK_N]
        p = tl.exp(centered)                              # [BLOCK_M, BLOCK_N]
        p = tl.where(attn_mask, p, 0.0)                  # kill invalid

        # Sum of probabilities in this block for each row
        s_ij = tl.sum(p, axis=1)                         # [BLOCK_M]

        # ---- Online softmax update ----
        # Only rows that have valid keys in THIS block need state updates.
        # For rows with no valid key here, nothing changes.

        # Scale factor for OLD accumulator contribution
        # alpha = exp(old_m - new_m). If old had no data, treat old_m such that alpha=1.
        new_flag = flag_i | has_valid                     # [BLOCK_M]

        # Compute safe exp(old_m - new_m)
        # When !has_valid: won't matter since s_ij=0 and p=0 anyway.
        # When has_valid but !flag_i (first time seeing data): old_m is 0, new_m is real.
        #   We want alpha ≈ 0 so old (zero) accum doesn't pollute.
        exponent = row_m - tl.where(has_valid, m_ij, row_m)   # safe; =0 when !has_valid
        exponent_clamped = tl.maximum(exponent, -50.0)         # prevent underflow to inf
        alpha = tl.exp(exponent_clamped)                      # [BLOCK_M]

        # Update normalized denominator
        row_lse = alpha * row_lse + s_ij                      # [BLOCK_M]

        # Update running max
        row_m = tl.where(new_flag, tl.maximum(row_m, m_ij), row_m)

        # Rescale accumulator
        o_accum = o_accum * alpha[:, None]                    # [BLOCK_M, BLOCK_D]

        # Add new weighted V contribution
        v_ptrs = v_base + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=n_in_bounds, other=0.0)     # [BLOCK_N, BLOCK_D]
        o_add = tl.dot(p.to(tl.bfloat16), v)                  # [BLOCK_M, BLOCK_D]
        o_accum = o_accum + o_add

        # Update flags
        flag_i = new_flag

    # Finalize: divide accumulator by lse sum
    inv_lse = tl.where(row_lse > 0.0, 1.0 / row_lse, 0.0)    # [BLOCK_M]
    o_final = o_accum * inv_lse[:, None]                       # [BLOCK_M, BLOCK_D]

    # Store output O  [BLOCK_M, BLOCK_D]
    o_base = O + bid * stride_ob + hid * stride_oh
    o_ptrs = o_base + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(o_ptrs, o_final.to(tl.bfloat16), mask=offs_m[:, None] < S)

    # Store LSE
    lse_out = tl.where(flag_i, row_m + tl.log(row_lse), float("-inf"))
    lse_base = LSE + bid * stride_lse_b + hid * stride_lse_h
    lse_ptrs = lse_base + offs_m * stride_lse_s
    tl.store(lse_ptrs, lse_out, mask=offs_m < S)


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