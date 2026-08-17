import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=4, num_stages=3),
    ],
    key=["n_ctx"],
    reset_to_zero=["out_ptr", "lse_ptr"],
)
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
    head_dim,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    h_per_batch = tl.num_programs(1)
    batch_idx = off_hz // h_per_batch
    head_idx = off_hz % h_per_batch

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, head_dim)

    # Load Q tile [BLOCK_M, HEAD_DIM]
    q_offs_m = offs_m[:, None]
    q_offs_d = offs_d[None, :]
    q_ptrs = (q_ptr
              + batch_idx * stride_qb + head_idx * stride_qh
              + q_offs_m * stride_qs + q_offs_d * stride_qd)
    q_row_mask = (offs_m[:, None] < n_ctx) & (offs_d[None, :] < head_dim)
    Q_tile = tl.load(q_ptrs, mask=q_row_mask, other=0.0)

    # Online softmax accumulators
    m_i = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, head_dim), tl.float32)

    for start_n in range(tl.cdiv(n_ctx, BLOCK_N)):
        offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)

        # K mask and load
        k_col_valid = offs_n < n_ctx
        k_ptrs = (k_ptr
                  + batch_idx * stride_kb + head_idx * stride_kh
                  + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_qd)
        k_load_mask = k_col_valid[:, None] & (offs_d[None, :] < head_dim)
        K_tile = tl.load(k_ptrs, mask=k_load_mask, other=0.0)

        # Scores [BLOCK_M, BLOCK_N] in fp32
        scores = tl.dot(Q_tile, K_tile.T) * scale

        # Suppress OOB columns BEFORE the max-reduction
        col_mask = (offs_n[None, :] < n_ctx)
        scores = tl.where(col_mask, scores, float("-inf"))

        # Online softmax recurrence
        m_i_prev = m_i
        m_i_new = tl.maximum(m_i, tl.max(scores, axis=1, keep_dims=False))
        alpha = tl.exp(m_i_prev - m_i_new)

        # Renormalised probs [BLOCK_M, BLOCK_N]
        p = tl.exp(scores - m_i_new[:, None])

        # Update stats and rescale old accumulation
        l_i = alpha * l_i + tl.sum(p, axis=1, keep_dims=False)
        acc = acc * alpha[:, None]

        # V mask and load
        v_ptrs = (v_ptr
                  + batch_idx * stride_vb + head_idx * stride_vh
                  + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_qd)
        v_load_mask = k_col_valid[:, None] & (offs_d[None, :] < head_dim)
        V_tile = tl.load(v_ptrs, mask=v_load_mask, other=0.0)

        # Accumulate p @ V
        acc = tl.dot(p.to(tl.bfloat16), V_tile) + acc

    # Epilogue – normalise
    inf_inv = tl.where(l_i > 0.0, tl.reciprocal(l_i), 0.0)
    acc = acc * inf_inv[:, None]

    # Store O
    out_ptrs = (out_ptr
                + batch_idx * stride_ob + head_idx * stride_oh
                + q_offs_m * stride_os + q_offs_d * stride_od)
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=q_row_mask)

    # Store LSE
    lse_val = m_i + tl.log(l_i)
    lse_ptrs = (lse_ptr
                + batch_idx * stride_lb + head_idx * stride_lh
                + offs_m * stride_ls)
    msk = offs_m < n_ctx
    tl.store(lse_ptrs, lse_val, mask=msk)


def run(Q, K, V, O, LSE):
    """Multi-head attention forward: O = softmax(Q K^T / sqrt(D)) V.

    Inputs  (bf16 [B, H, S, D]): Q, K, V
    Outputs (preallocated):       O [B, H, S, D] bf16, LSE [B, H, S] fp32
    """
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    assert K.shape == Q.shape and V.shape == Q.shape
    assert O.shape == Q.shape
    assert LSE.shape == (B, H, S)
    assert Q.dtype == K.dtype == V.dtype == torch.bfloat16

    scale = 1.0 / (D ** 0.5)

    # Grid: axis-0 covers Q-sequence tiles, axis-1 covers flat batch×head
    grid = lambda META: (triton.cdiv(S, META["BLOCK_M"]), B * H)

    _mha_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        S, D, scale,
    )