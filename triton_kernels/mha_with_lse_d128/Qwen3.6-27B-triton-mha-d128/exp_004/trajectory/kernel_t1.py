import torch
import triton
import triton.language as tl


@triton.jit
def _mha_kernel(
    Q,
    K,
    V,
    O,
    LSE,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    seq_len,
    num_heads,
    SCALE_INV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Tiled multi-head attention kernel with online softmax.
    Each program handles one (batch, head, query_tile) combination.
    """
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    pid_b = pid_bh // num_heads
    pid_h = pid_bh % num_heads

    # Offset arrays
    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)       # [BLOCK_M]
    off_d = tl.arange(0, BLOCK_D)                          # [BLOCK_D]

    # Base pointers for this (batch, head)
    q_base = Q + pid_b * stride_qb + pid_h * stride_qh
    k_base = K + pid_b * stride_kb + pid_h * stride_kh
    v_base = V + pid_b * stride_vb + pid_h * stride_vh
    o_base = O + pid_b * stride_ob + pid_h * stride_oh
    lse_base = LSE + pid_b * stride_lseb + pid_h * stride_lseh

    # Accumulators: output in fp32 for numerical stability
    acc_o = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), value=-1e10, dtype=tl.float32)
    l_i = tl.full((BLOCK_M,), value=1.0, dtype=tl.float32)

    # Query row validity mask
    row_mask = off_m < seq_len                              # [BLOCK_M]

    # Load Q tile once
    q_ptrs = q_base + off_m[:, None] * stride_qs + off_d[None, :] * stride_qd
    q_tile = tl.load(q_ptrs, mask=row_mask[:, None], other=0.0)

    # Iterate over K/V blocks
    num_steps = tl.cdiv(seq_len, BLOCK_N)
    for step in range(num_steps):
        start_n = step * BLOCK_N
        off_n = start_n + tl.arange(0, BLOCK_N)            # [BLOCK_N]
        col_mask = off_n < seq_len                           # [BLOCK_N]

        # Load K tile [BLOCK_N, BLOCK_D]
        k_ptrs = k_base + off_n[:, None] * stride_ks + off_d[None, :] * stride_kd
        k_tile = tl.load(k_ptrs, mask=col_mask[:, None], other=0.0)

        # Load V tile [BLOCK_N, BLOCK_D]
        v_ptrs = v_base + off_n[:, None] * stride_vs + off_d[None, :] * stride_vd
        v_tile = tl.load(v_ptrs, mask=col_mask[:, None], other=0.0)

        # Attention scores S = Q @ K^T / sqrt(D)
        s = tl.dot(q_tile, k_tile) * SCALE_INV              # [BLOCK_M, BLOCK_N]

        # Apply column mask to scores: out-of-bound cols become very negative
        s = tl.where(col_mask[None, :], s, -1e10)

        # Online softmax: update running max
        m_prev = m_i
        m_i = tl.maximum(m_i, tl.max(s, axis=1))

        # Stability clamp: avoid exp(diff) = nan when both prev and new are extreme
        diff = m_prev - m_i
        alpha = tl.where(diff > -50.0, tl.exp(diff), 0.0)

        # Softmax probabilities for this block
        p = tl.exp(s - m_i[:, None])                        # [BLOCK_M, BLOCK_N]

        # Accumulate
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc_o = acc_o * alpha[:, None] + tl.dot(p, v_tile)

    # Finalization
    l_i_safe = tl.where(l_i > 0.0, l_i, 1.0)
    acc_o = acc_o / l_i_safe[:, None]

    # LSE = m_i + log(l_i)
    lse_out = m_i + tl.log(l_i_safe)

    # Store output O [BLOCK_M, BLOCK_D]
    o_ptrs = o_base + off_m[:, None] * stride_os + off_d[None, :] * stride_od
    tl.store(o_ptrs, acc_o.to(tl.bfloat16), mask=row_mask[:, None])

    # Store LSE [BLOCK_M]
    lse_ptrs = lse_base + off_m * stride_lses
    tl.store(lse_ptrs, lse_out, mask=row_mask)


def run(Q, K, V, O, LSE):
    """
    Multi-head attention forward: O = softmax(Q @ K^T / sqrt(D)) @ V

    Inputs (destination-passing order):
        Q, K, V : bf16 [B, H, S, D]
        O       : bf16 [B, H, S, D]  (preallocated output)
        LSE     : fp32 [B, H, S]      (preallocated output)
    """
    torch.cuda.set_device(Q.device)

    B, H, S, D = Q.shape

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = 128  # Fixed head dimension

    scale_inv = 1.0 / (D ** 0.5)

    # Grid: axis 0 = batch*head, axis 1 = query tiles
    grid = (B * H, triton.cdiv(S, BLOCK_M))

    _mha_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        S,
        H,
        SCALE_INV=scale_inv,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=3,
    )