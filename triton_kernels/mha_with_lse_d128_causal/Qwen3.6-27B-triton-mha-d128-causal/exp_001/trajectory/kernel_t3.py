import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=3),
    ],
    key=['S'],
)
@triton.jit
def _mha_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lse_b, stride_lse_h, stride_lse_s,
    B, H, S,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr,
):
    """Causal multi-head attention with online softmax and LSE."""

    pid_bh = tl.program_id(0)    # batch × head program ID
    pid_m = tl.program_id(1)     # query sequence tile ID

    batch_idx = pid_bh // H
    head_idx = pid_bh % H

    # ── query-side row offsets ─────────────────────────────────────
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)          # [BLOCK_M]
    mask_m = offs_m < S                                        # [BLOCK_M]

    # ── key/value-side column offsets (fixed within kernel) ───────
    offs_n = tl.arange(0, BLOCK_N)                             # [BLOCK_N]

    # ── head-dimension column offsets ─────────────────────────────
    offs_d = tl.arange(0, D)                                   # [D] — D is constexpr
    mask_d = offs_d < D                                        # [D]

    # ── base pointer offsets for this batch × head ────────────────
    q_off = batch_idx * stride_qb + head_idx * stride_qh
    k_off = batch_idx * stride_kb + head_idx * stride_kh
    v_off = batch_idx * stride_vb + head_idx * stride_vh
    o_off = batch_idx * stride_ob + head_idx * stride_oh
    lse_off = batch_idx * stride_lse_b + head_idx * stride_lse_h

    # ── load Q tile [BLOCK_M, D] ──────────────────────────────────
    q_ptrs = q_off + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd  # [BLOCK_M, D]
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)      # [BLOCK_M, D]

    # ── online softmax accumulators ───────────────────────────────
    acc_o = tl.zeros((BLOCK_M, D), dtype=tl.float32)           # accumulated softweighted V
    m_i = tl.full((BLOCK_M,), value=float('-inf'), dtype=tl.float32)  # running max
    d_i = tl.full((BLOCK_M,), value=1.0, dtype=tl.float32)     # running sum(exp)

    inv_sqrt_d = 1.0 / tl.sqrt(D)                              # fp32 scalar

    # ── iterate over K/V tiles along the key-sequence axis ───────
    for start_n in range(0, S, BLOCK_N):
        # Current key position offsets [BLOCK_N]
        cur_n = start_n + offs_n                                  # actual seq positions
        mask_n = cur_n < S                                      # [BLOCK_N]

        # Load K tile [BLOCK_N, D]
        k_ptrs = k_off + cur_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)

        # Load V tile [BLOCK_N, D]
        v_ptrs = v_off + cur_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)

        # ── attention scores  Q@K^T / sqrt(D)  → [BLOCK_M, BLOCK_N] ──
        scores = tl.dot(q, k.T) * inv_sqrt_d

        # Causal mask: query pos >= key pos ⇒ keep; else → -inf
        causal = (offs_m[:, None] >= cur_n[None, :])             # [BLOCK_M, BLOCK_N]
        scores = tl.where(causal, scores, float('-inf'))

        # ── online softmax update ─────────────────────────────────
        m_ij = tl.max(scores, axis=1)                           # [BLOCK_M]
        m_i_old = m_i                                           # save old max before update
        m_i = tl.maximum(m_i, m_ij)                             # new running max
        p = tl.exp(scores - m_i[:, None])                       # stabilized probs
        alpha = tl.exp(m_i_old - m_i)                           # rescaling factor
        d_i = d_i * alpha + tl.sum(p, axis=1)                   # update denom
        acc_o = acc_o * alpha[:, None] + tl.dot(p, v)           # update output acc

    # ── normalize accumulated output ──────────────────────────────
    acc_o = acc_o / d_i[:, None]                                # divide by partition fn

    # ── store O [BLOCK_M, D] ──────────────────────────────────────
    o_ptrs = o_off + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc_o.to(tl.bfloat16),
             mask=mask_m[:, None] & mask_d[None, :])

    # ── store LSE  [BLOCK_M] ──────────────────────────────────────
    lse_val = m_i + tl.log(d_i)                                 # float32
    lse_ptrs = lse_off + offs_m * stride_lse_s
    tl.store(lse_ptrs, lse_val, mask=mask_m)


def run(Q, K, V, O, LSE):
    """Compute causal multi-head attention forward with LSE output.

    Parameters
    ----------
    Q, K, V : torch.Tensor  [B, H, S, D]  bfloat16
    O       : torch.Tensor  [B, H, S, D]  bfloat16  (preallocated output)
    LSE     : torch.Tensor  [B, H, S]     float32   (preallocated output)
    """
    torch.cuda.set_device(Q.device)

    B, H, S, D = Q.shape
    n_batch_heads = B * H

    grid = lambda META: (
        n_batch_heads,
        triton.cdiv(S, META['BLOCK_M']),
    )

    _mha_fwd_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        B, H, S,
        D=D,
    )