import torch
import triton
import triton.language as tl


@triton.jit
def _mha_kernel(
    q_ptr, k_ptr, v_ptr,
    out_ptr, lse_ptr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lb, stride_lh, stride_ls,
    n_ctx,
    num_heads,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """Non-causal multi-head attention with online softmax."""
    pid_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    batch_idx = off_hz // num_heads
    head_idx = off_hz % num_heads

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    # Validity masks
    row_valid = offs_m < n_ctx                           # [BLOCK_M]
    col_range = tl.arange(0, BLOCK_N)                    # [BLOCK_N]

    # ---- Load Q tile [BLOCK_M, HEAD_DIM] ----
    q_ptrs = (q_ptr
              + batch_idx * stride_qb + head_idx * stride_qh
              + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd)
    q_mask = row_valid[:, None] & (offs_d < HEAD_DIM)[None, :]
    Q_tile = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # ---- Init online softmax accumulators ----
    m_i = tl.full((BLOCK_M,), -1e10, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

    # Masking constant for invalid columns (large negative, but finite)
    MASK_VAL = -1e10

    # ---- Iterate over K / V tiles ----
    num_blocks_n = tl.cdiv(n_ctx, BLOCK_N)
    for start_n in range(num_blocks_n):
        offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
        col_valid = offs_n < n_ctx                        # [BLOCK_N]

        # Load K [BLOCK_N, HEAD_DIM]
        k_ptrs = (k_ptr
                  + batch_idx * stride_kb + head_idx * stride_kh
                  + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd)
        K_tile = tl.load(k_ptrs,
                         mask=(col_valid[:, None]) & (offs_d < HEAD_DIM)[None, :],
                         other=0.0)

        # Scores [BLOCK_M, BLOCK_N] fp32
        scores = tl.dot(Q_tile, K_tile.T) * scale

        # Suppress OOB columns before max-reduction
        scores = tl.where(col_valid[None, :], scores, MASK_VAL)

        # Online softmax recurrence
        m_i_prev = m_i
        row_max = tl.max(scores, axis=1)
        m_i_new = tl.maximum(m_i, row_max)

        alpha = tl.exp(m_i_prev - m_i_new)

        # Renormalised probs [BLOCK_M, BLOCK_N]
        p = tl.exp(scores - m_i_new[:, None])

        l_i = alpha * l_i + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        # Load V [BLOCK_N, HEAD_DIM]
        v_ptrs = (v_ptr
                  + batch_idx * stride_vb + head_idx * stride_vh
                  + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd)
        V_tile = tl.load(v_ptrs,
                         mask=(col_valid[:, None]) & (offs_d < HEAD_DIM)[None, :],
                         other=0.0)

        # Accumulate p @ V  — cast p back to bf16 for Tensor Core
        acc = acc + tl.dot(p.to(tl.bfloat16), V_tile)

    # ---- Epilogue ----
    # Guard against l_i <= 0 (can happen for fully-masked rows)
    l_i_safe = tl.where(l_i > 0.0, l_i, 1.0)
    inv_l = 1.0 / l_i_safe
    acc = acc * inv_l[:, None]

    # Store output O [BLOCK_M, HEAD_DIM]
    out_ptrs = (out_ptr
                + batch_idx * stride_ob + head_idx * stride_oh
                + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od)
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=q_mask)

    # Store LSE [BLOCK_M]
    lse_val = m_i + tl.log(l_i_safe)
    lse_ptrs = (lse_ptr
                + batch_idx * stride_lb + head_idx * stride_lh
                + offs_m * stride_ls)
    tl.store(lse_ptrs, lse_val, mask=row_valid)


def run(Q, K, V, O, LSE):
    """Multi-head attention forward: O = softmax(Q K^T / sqrt(D)) V.

    Inputs  (bf16 [B, H, S, D]): Q, K, V
    Outputs (preallocated):       O [B, H, S, D] bf16, LSE [B, H, S] fp32
    """
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    scale = 1.0 / (D ** 0.5)

    BLOCK_M = 128
    BLOCK_N = 64
    grid = (triton.cdiv(S, BLOCK_M), B * H)

    _mha_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        S, H, scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        HEAD_DIM=D,
        num_warps=4,
        num_stages=3,
    )