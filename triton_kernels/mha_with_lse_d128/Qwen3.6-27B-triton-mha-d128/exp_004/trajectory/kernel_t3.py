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
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    pid_b = pid_bh // num_heads
    pid_h = pid_bh % num_heads

    # Index offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)       # [BLOCK_M]
    offs_n = tl.arange(0, BLOCK_N)                          # [BLOCK_N]
    offs_d = tl.arange(0, BLOCK_D)                          # [BLOCK_D]

    # Base pointers for this (batch, head)
    q_base = Q + pid_b * stride_qb + pid_h * stride_qh
    k_base = K + pid_b * stride_kb + pid_h * stride_kh
    v_base = V + pid_b * stride_vb + pid_h * stride_vh
    o_base = O + pid_b * stride_ob + pid_h * stride_oh
    lse_base = LSE + pid_b * stride_lseb + pid_h * stride_lseh

    # Masks
    row_mask = (offs_m < seq_len)                           # [BLOCK_M]
    row_mask_2d = row_mask[:, None]                         # [BLOCK_M, 1]

    # Load Q tile once
    q_ptrs = q_base + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    q_tile = tl.load(q_ptrs, mask=row_mask_2d, other=0.0)   # [BLOCK_M, BLOCK_D] bf16

    # Initialize accumulators
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)    # [BLOCK_M, BLOCK_D]
    m_i = tl.full((BLOCK_M,), value=-float("inf"), dtype=tl.float32)
    l_i = tl.full((BLOCK_M,), value=1.0, dtype=tl.float32)

    # Loop over key/value blocks
    for start_n in range(0, seq_len, BLOCK_N):
        cur_n = start_n + offs_n                            # [BLOCK_N]
        col_mask = cur_n < seq_len                           # [BLOCK_N]
        col_mask_2d = col_mask[:, None]                      # [BLOCK_N, 1]

        # Load K tile [BLOCK_N, BLOCK_D]
        k_ptrs = k_base + cur_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        k_tile = tl.load(k_ptrs, mask=col_mask_2d, other=0.0)

        # Attention scores S = Q @ K^T / sqrt(D)
        s = tl.dot(q_tile, k_tile) * SCALE_INV               # [BLOCK_M, BLOCK_N] fp32

        # Clip invalid columns to a large negative value before softmax
        # Use row-wise mask combined with column mask
        col_mask_row = col_mask[None, :]                      # [1, BLOCK_N]
        neg_inf_f32 = tl.full((BLOCK_M, BLOCK_N), value=-1e12, dtype=tl.float32)
        s = tl.where(row_mask_2d & col_mask_row, s, neg_inf_f32)

        # Online softmax step
        m_prev = m_i                                          # [BLOCK_M]
        row_max_s = tl.max(s, axis=1)                         # [BLOCK_M]
        m_i_new = tl.maximum(m_i, row_max_s)                  # [BLOCK_M]

        # Stable exponent
        alpha = tl.exp(m_i - m_i_new)                         # [BLOCK_M]
        p_scale = tl.exp(s - m_i_new[:, None])                # [BLOCK_M, BLOCK_N]

        # Accumulate
        l_i = l_i * alpha + tl.sum(p_scale, axis=1)           # [BLOCK_M]

        # Load V tile [BLOCK_N, BLOCK_D]
        v_ptrs = v_base + cur_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
        v_tile = tl.load(v_ptrs, mask=col_mask_2d, other=0.0)

        # Weighted accumulation into output
        acc = acc * alpha[:, None] + tl.dot(p_scale.to(tl.bfloat16), v_tile)

        m_i = m_i_new

    # Finalize
    inv_sum = 1.0 / l_i                                      # [BLOCK_M]
    acc_final = acc * inv_sum[:, None]

    # LSE = m_i + log(l_i)
    lse_out = m_i + tl.log(l_i)

    # Store output O [BLOCK_M, BLOCK_D]
    o_ptrs = o_base + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc_final.to(tl.bfloat16), mask=row_mask_2d)

    # Store LSE [BLOCK_M]
    lse_ptrs = lse_base + offs_m * stride_lses
    tl.store(lse_ptrs, lse_out, mask=row_mask)


def run(Q, K, V, O, LSE):
    """
    Multi-head attention forward pass.
    O = softmax(Q @ K^T / sqrt(D)) @ V
    LSE = logsumexp(Q @ K^T / sqrt(D), dim=-1)
    
    Destination-passing: writes into preallocated O and LSE.
    """
    torch.cuda.set_device(Q.device)

    B, H, S, D = Q.shape

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = 128

    scale_inv = 1.0 / float(D ** 0.5)

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