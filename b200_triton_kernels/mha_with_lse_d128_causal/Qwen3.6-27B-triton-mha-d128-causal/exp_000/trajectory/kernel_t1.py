import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
    ],
    key=['S'],
)
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

    # Attention temperature scale (= 1 / sqrt(D))
    scale = 1.0 / tl.sqrt(D.to(tl.float32))

    # Lane offsets (block-level coordinates)
    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    off_d = tl.arange(0, BLOCK_D)                      # [BLOCK_D]

    # ------------------------------------------------------------------
    # 1) Load Q tile — loop invariant, reused across all KV tiles
    # ------------------------------------------------------------------
    q_base = Q + bid * stride_qb + hid * stride_qh
    q_ptrs = q_base + off_m[:, None] * stride_qs + off_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=(off_m[:, None] < S), other=0.0)  # bf16 [BLOCK_M, BLOCK_D]

    # ------------------------------------------------------------------
    # 2) Initialise online softmax bookkeeping
    # ------------------------------------------------------------------
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)       # weighted sum accumulator
    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32) # running max of scores
    l_i = tl.full((BLOCK_M,), 0.0, dtype=tl.float32)           # running sum of exp(score-max)

    # Loop-invariant pointer bases for K and V
    k_base = K + bid * stride_kb + hid * stride_kh
    v_base = V + bid * stride_vb + hid * stride_vh

    # ------------------------------------------------------------------
    # 3) Iterate over KV tiles
    # ------------------------------------------------------------------
    for start_n in range(0, S, BLOCK_N):
        off_n = start_n + tl.arange(0, BLOCK_N)               # [BLOCK_N]

        # --- masks ---------------------------------------------------
        # Causal: key position n must be <= query position m
        causal_mask = (off_m[:, None] >= off_n[None, :])      # [BLOCK_M, BLOCK_N]
        # Sequence boundary for keys
        n_valid = (off_n < S)                                  # [BLOCK_N]
        kv_mask = causal_mask & n_valid[None, :]               # combined mask
        n_load_mask = (off_n[:, None] < S)                     # for K/V loads

        # --- load K tile -----------------------------------------------
        k_ptrs = k_base + off_n[:, None] * stride_ks + off_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=n_load_mask, other=0.0)      # bf16 [BLOCK_N, BLOCK_D]

        # --- Q · K^T scores, scaled -----------------------------------
        qk = tl.dot(q, k.T)                                    # fp32 [BLOCK_M, BLOCK_N]
        qk = qk * scale
        qk = tl.where(kv_mask, qk, float("-inf"))             # enforce causal mask

        # --- online softmax update (BlockSoftmax) ---------------------
        m_ij = tl.max(qk, axis=1)                              # [BLOCK_M]

        # Detect rows where ALL scores are -inf (no valid causal keys in this block).
        # exp(-inf - (-inf)) = NaN, so we must guard against this.
        has_valid = m_ij != float("-inf")                      # [BLOCK_M]

        # Safe subtraction reference: use 0.0 when m_ij == -inf; result is clamped below.
        m_ij_safe = tl.where(has_valid, m_ij, 0.0)
        exp_arg = qk - m_ij_safe[:, None]                      # [BLOCK_M, BLOCK_N]
        p = tl.exp(exp_arg)
        # Zero out contributions from invalid (masked) positions
        p = tl.where(kv_mask, p, 0.0)

        l_ij = tl.sum(p, axis=1)                               # [BLOCK_M]

        # Alpha re-scaling factor
        # When has_valid=False: alpha=1 (no rescaling needed, block contributes nothing)
        # When has_valid=True: alpha = exp(old_max - new_max), clamped safely
        alpha = tl.where(
            has_valid,
            tl.exp(m_i - m_ij),
            1.0,
        )

        l_i = l_i * alpha + l_ij
        m_i = tl.where(has_valid, m_ij, m_i)

        # Scale down old accumulated output and add new contributions
        acc = acc * alpha[:, None]

        # --- load V tile and accumulate P · V ------------------------
        v_ptrs = v_base + off_n[:, None] * stride_vs + off_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=n_load_mask, other=0.0)       # bf16 [BLOCK_N, BLOCK_D]
        acc = tl.dot(p.to(tl.bfloat16), v, acc)                # fp32 accumulation

    # ------------------------------------------------------------------
    # 4) Finalise — divide by normaliser
    # ------------------------------------------------------------------
    # Guard division: if l_i==0 (shouldn't happen for valid causal attn), produce 0 output
    acc = tl.where(
        l_i > 0.0,
        acc / l_i[:, None],
        0.0,
    )

    # ------------------------------------------------------------------
    # 5) Store outputs
    # ------------------------------------------------------------------
    # Output O
    o_base = O + bid * stride_ob + hid * stride_oh
    o_ptrs = o_base + off_m[:, None] * stride_os + off_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(tl.bfloat16), mask=(off_m[:, None] < S))

    # LSE = max(scores) + log(sum(exp(scores - max))) = m_i + log(l_i)
    lse = tl.where(
        l_i > 0.0,
        m_i + tl.log(l_i),
        float("-inf"),
    )
    lse_base = LSE + bid * stride_lse_b + hid * stride_lse_h
    lse_ptrs = lse_base + off_m * stride_lse_s
    tl.store(lse_ptrs, lse, mask=off_m < S)


def run(Q, K, V, O, LSE):
    """Causal multi-head attention forward with LSE output.

    Parameters
    ----------
    Q, K, V : torch.Tensor (B, H, S, D), bfloat16
        Query, Key, Value tensors.
    O : torch.Tensor (B, H, S, D), bfloat16
        Preallocated output for the attended values.
    LSE : torch.Tensor (B, H, S), float32
        Preallocated output for the log-sum-exp of raw attention scores.

    Semantics
    ---------
    P = Q @ K^T / sqrt(D)                  (raw scores, with lower-tri causal mask)
    A = softmax(P, dim=-1)                 (normalised attention weights)
    O = A @ V
    LSE = logsumexp(P, dim=-1)             (numerically stable log-sum-exp)
    """
    torch.cuda.set_device(Q.device)

    B, H, S, D = Q.shape

    grid = lambda META: (
        triton.cdiv(S, META["BLOCK_M"]),
        B * H,
    )

    _mha_causal_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        B, H, S, D,
        BLOCK_D=D,
    )