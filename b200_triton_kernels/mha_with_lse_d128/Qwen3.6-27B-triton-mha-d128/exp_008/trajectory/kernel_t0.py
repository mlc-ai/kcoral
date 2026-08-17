import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=4, num_stages=4),
    ],
    key=["N_CTX"],
)
@triton.jit
def _flash_attention_kernel(
    Q, K, V,
    Out, LSE,
    stride_qb, stride_qh, stride_qs,
    stride_kb, stride_kh, stride_ks,
    stride_vb, stride_vh, stride_vs,
    stride_ob, stride_oh, stride_os,
    stride_lb, stride_lh, stride_ls,
    N_CTX,
    B, H,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """FlashAttention-3 style forward kernel for non-causal MHA on Hopper."""
    pid_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    batch_idx = off_hz // H
    head_idx = off_hz % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    # ---- Load Q tile (once) [BLOCK_M, HEAD_DIM] ----
    q_ptrs = (Q + batch_idx * stride_qb + head_idx * stride_qh +
              offs_m[:, None] * stride_qs + offs_d[None, :])
    q_mask = offs_m[:, None] < N_CTX                          # [BLOCK_M, 1]
    Q_tile = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # ---- Init online softmax state ----
    m_i = tl.full([BLOCK_M], -1.0e20, tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)

    # ---- Iterate over K / V tiles ----
    for start_n in range(0, tl.cdiv(N_CTX, BLOCK_N)):
        offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)

        # Masks
        kv_mask = offs_n[:, None] < N_CTX                     # [BLOCK_N, 1] for K/V loads
        score_mask = offs_n[None, :] < N_CTX                  # [1, BLOCK_N] for scores

        # Load K tile [BLOCK_N, HEAD_DIM]
        k_ptrs = (K + batch_idx * stride_kb + head_idx * stride_kh +
                  offs_n[:, None] * stride_ks + offs_d[None, :])
        K_tile = tl.load(k_ptrs, mask=kv_mask, other=0.0)

        # Scores [BLOCK_M, BLOCK_N]
        scores = tl.dot(Q_tile, K_tile.T) * scale

        # Suppress out-of-bound columns so they cannot corrupt the softmax max
        scores = tl.where(score_mask, scores, -1.0e20)

        # Online softmax recurrence
        m_i_prev = m_i
        m_i = tl.maximum(m_i, tl.max(scores, axis=1))

        alpha = tl.exp(m_i_prev - m_i)                         # rescale factor for old accum
        p = tl.exp(scores - m_i[:, None])                      # renormalised probabilities

        l_i = alpha * l_i + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        # Load V tile [BLOCK_N, HEAD_DIM]
        v_ptrs = (V + batch_idx * stride_vb + head_idx * stride_vh +
                  offs_n[:, None] * stride_vs + offs_d[None, :])
        V_tile = tl.load(v_ptrs, mask=kv_mask, other=0.0)

        # Accumulate  p[V]   — cast p back to bf16 for Tensor-Core PMAD
        acc = acc + tl.dot(p.to(tl.bfloat16), V_tile)

    # ---- Epilogue ----
    inv_l = tl.reciprocal(l_i)
    acc = acc * inv_l[:, None]

    # Write output O  [BLOCK_M, HEAD_DIM]
    o_ptrs = (Out + batch_idx * stride_ob + head_idx * stride_oh +
              offs_m[:, None] * stride_os + offs_d[None, :])
    tl.store(o_ptrs, acc.to(tl.bfloat16), mask=q_mask)

    # Write LSE  [BLOCK_M]
    lse = m_i + tl.log(l_i)
    lse_ptrs = LSE + batch_idx * stride_lb + head_idx * stride_lh + offs_m * stride_ls
    tl.store(lse_ptrs, lse, mask=offs_m < N_CTX)


def run(Q, K, V, O, LSE):
    """Multi-head attention forward: O = softmax(Q K^T / sqrt(D)) V.

    Inputs  (bf16 [B, H, S, D]): Q, K, V
    Outputs (preallocated):       O [B, H, S, D] bf16, LSE [B, H, S] fp32
    """
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    scale = 1.0 / (D ** 0.5)

    grid = lambda META: (triton.cdiv(S, META["BLOCK_M"]), B * H)

    _flash_attention_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        S, B, H, scale,
        HEAD_DIM=D,
    )