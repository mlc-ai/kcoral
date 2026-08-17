import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=4),
    ],
    key=["S"],
)
@triton.jit
def _mha_forward_kernel(
    Q, K, V, O, LSE,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    B, H, S,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
):
    """Forward MHA kernel with online softmax."""
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    bid_b = pid_bh // H
    bid_h = pid_bh % H

    # Advance base pointers to this (b, h) slice
    Q += bid_b * stride_qb + bid_h * stride_qh
    K += bid_b * stride_kb + bid_h * stride_kh
    V += bid_b * stride_vb + bid_h * stride_vh
    O += bid_b * stride_ob + bid_h * stride_oh
    LSE += bid_b * stride_lseb + bid_h * stride_lseh

    # Index arrays
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    m_mask = offs_m < S

    # Load Q tile once
    q_ptrs = Q + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    Q_tile = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0)

    # Initialize online softmax state
    acc_o = tl.zeros((BLOCK_M, D), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.full((BLOCK_M,), 1.0, dtype=tl.float32)

    # Loop over N tiles
    for start_n in range(0, S, BLOCK_N):
        n_offs = start_n + offs_n
        n_mask = n_offs < S

        # Load K [BLOCK_N, D]
        k_ptrs = K + n_offs[:, None] * stride_ks + offs_d[None, :] * stride_kd
        K_tile = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)

        # Q @ K^T in bf16 -> fp32 accumulator
        scores = tl.dot(Q_tile, K_tile.T) * scale

        # Combined mask
        mn_mask = m_mask[:, None] & n_mask[None, :]
        scores = tl.where(mn_mask, scores, float("-inf"))

        # Online softmax
        m_ij = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_ij)

        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        l_new = alpha * l_i + tl.sum(p, axis=1)

        acc_o = acc_o * alpha[:, None]

        # Load V [BLOCK_N, D]
        v_ptrs = V + n_offs[:, None] * stride_vs + offs_d[None, :] * stride_vd
        V_tile = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)

        # p->bf16 so dot stays within Tensor Core bf16 path
        acc_o = acc_o + tl.dot(p.to(tl.bfloat16), V_tile)

        m_i = m_new
        l_i = l_new

    # Normalize
    acc_o = acc_o / l_i[:, None]
    o_ptrs = O + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc_o.to(tl.bfloat16), mask=m_mask[:, None])

    # Store LSE
    lse_vals = m_i + tl.log(l_i)
    lse_ptrs = LSE + offs_m * stride_lses
    tl.store(lse_ptrs, lse_vals, mask=m_mask)


def run(Q, K, V, O, LSE):
    """Compute non-causal multi-head attention forward pass with LSE."""
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape

    scale = 1.0 / (D ** 0.5)

    grid = lambda META: (B * H, triton.cdiv(S, META["BLOCK_M"]))

    _mha_forward_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        B, H, S,
        scale,
        D=D,
    )