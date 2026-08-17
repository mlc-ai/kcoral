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

    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    batch_idx = pid_bh // H
    head_idx = pid_bh % H

    # ── index vectors ─────────────────────────────────────────────
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    mask_m = offs_m < S                                 # [BLOCK_M]
    offs_d = tl.arange(0, D)                            # [D]
    mask_d = offs_d < D                                 # [D]

    inv_sqrt_d = 1.0 / tl.sqrt(D)                      # fp32 scalar

    # ── load Q tile [BLOCK_M, D], keep as bf16 for tensor-core dot ──
    q_ptrs = Q_ptr + \
             batch_idx * stride_qb + head_idx * stride_qh + \
             offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)

    # ── online softmax accumulators (fp32) ────────────────────────
    acc_o = tl.zeros((BLOCK_M, D), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), value=float('-inf'), dtype=tl.float32)
    d_i = tl.full((BLOCK_M,), value=1.0, dtype=tl.float32)

    # ── iterate over K/V tiles along sequence axis ────────────────
    for start_n in range(0, S, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)      # [BLOCK_N]
        mask_n = offs_n < S                             # [BLOCK_N]

        # Load K tile [BLOCK_N, D], keep bf16 for tensor-core dot
        k_ptrs = K_ptr + \
                 batch_idx * stride_kb + head_idx * stride_kh + \
                 offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)

        # Load V tile [BLOCK_N, D] → cast to fp32 for accumulation dot
        v_ptrs = V_ptr + \
                 batch_idx * stride_vb + head_idx * stride_vh + \
                 offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        v = v.to(tl.float32)

        # Attention scores [BLOCK_M, BLOCK_N], bf16×bf16→fp32 via tensor cores
        scores = tl.dot(q, k.T) * inv_sqrt_d

        # Causal mask: query_pos >= key_pos → keep; else → -inf
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        scores = tl.where(causal_mask, scores, float('-inf'))

        # Online softmax update (all fp32)
        m_ij = tl.max(scores, axis=1)                  # [BLOCK_M]
        m_i_old = m_i
        m_i = tl.maximum(m_i, m_ij)
        p = tl.exp(scores - m_i[:, None])               # [BLOCK_M, BLOCK_N] fp32
        alpha = tl.exp(m_i_old - m_i)                   # [BLOCK_M]
        d_i = d_i * alpha + tl.sum(p, axis=1)           # [BLOCK_M]
        acc_o = acc_o * alpha[:, None] + tl.dot(p, v)   # [BLOCK_M, D] fp32

    # ── normalize output ──────────────────────────────────────────
    acc_o = acc_o / d_i[:, None]

    # ── store O [BLOCK_M, D] ──────────────────────────────────────
    o_ptrs = O_ptr + \
             batch_idx * stride_ob + head_idx * stride_oh + \
             offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc_o.to(tl.bfloat16),
             mask=mask_m[:, None] & mask_d[None, :])

    # ── store LSE [BLOCK_M] ───────────────────────────────────────
    lse_val = m_i + tl.log(d_i)                         # float32
    lse_ptrs = LSE_ptr + \
               batch_idx * stride_lse_b + head_idx * stride_lse_h + \
               offs_m * stride_lse_s
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