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
    offs_d = tl.arange(0, BLOCK_D)                          # [BLOCK_D]

    # Base pointers for this (batch, head)
    q_base = Q + pid_b * stride_qb + pid_h * stride_qh
    k_base = K + pid_b * stride_kb + pid_h * stride_kh
    v_base = V + pid_b * stride_vb + pid_h * stride_vh
    o_base = O + pid_b * stride_ob + pid_h * stride_oh
    lse_base = LSE + pid_b * stride_lseb + pid_h * stride_lseh

    # Row mask for query sequence boundary
    row_mask = offs_m < seq_len                              # [BLOCK_M]

    # Load Q tile once (outside K/V loop)
    q_offs = offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    q_ptrs = q_base + q_offs
    q_tile = tl.load(q_ptrs, mask=row_mask[:, None], other=0.0)

    # Initialize accumulators for online softmax
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), value=float('-inf'), dtype=tl.float32)
    l_i = tl.full((BLOCK_M,), value=1.0, dtype=tl.float32)

    # Number of key/value blocks to iterate over
    num_kv_steps = tl.cdiv(seq_len, BLOCK_N)

    for start_n in range(0, seq_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)            # [BLOCK_N]
        col_mask = offs_n < seq_len                           # [BLOCK_N]

        # Load K tile: shape [BLOCK_N, BLOCK_D]
        k_offs = offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        k_ptrs = k_base + k_offs
        k_tile = tl.load(k_ptrs, mask=col_mask[:, None], other=0.0)

        # Compute attention scores: S[i,j] = Q[i,:] @ K[j,:] / sqrt(D)
        s = tl.dot(q_tile, k_tile) * SCALE_INV               # [BLOCK_M, BLOCK_N]

        # Apply column mask: set out-of-bounds columns to -inf before softmax
        neg_inf = tl.full((BLOCK_M, BLOCK_N), value=float('-inf'), dtype=tl.float32)
        s = tl.where(col_mask[None, :], s, neg_inf)

        # Online softmax update
        m_prev = m_i                                          # [BLOCK_M]
        row_max = tl.max(s, axis=1, keep_dims=False)          # [BLOCK_M]
        m_i = tl.maximum(m_i, row_max)                        # [BLOCK_M]

        # Correction factor: exp(m_prev - m_i)
        diff = m_prev - m_i                                   # [BLOCK_M]
        alpha = tl.exp(diff)                                  # [BLOCK_M]

        # Softmax weights for current block
        p = tl.exp(s - m_i[:, None])                          # [BLOCK_M, BLOCK_N]

        # Update running sum of softmax weights
        l_i = l_i * alpha + tl.sum(p, axis=1)                 # [BLOCK_M]

        # Load V tile: shape [BLOCK_N, BLOCK_D]
        v_offs = offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
        v_ptrs = v_base + v_offs
        v_tile = tl.load(v_ptrs, mask=col_mask[:, None], other=0.0)

        # Accumulate output: weighted sum of V
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v_tile.to(tl.bfloat16))

    # Finalize output
    inv_l_i = tl.rsqrt(l_i * l_i)                            # 1/l_i with rsqrt
    acc = acc * (1.0 / tl.where(l_i > 0.0, l_i, 1.0))[:, None]

    # LSE = m_i + log(l_i)
    l_i_safe = tl.where(l_i > 0.0, l_i, 1.0)
    lse_out = m_i + tl.log(l_i_safe)

    # Store output O
    o_offs = offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    o_ptrs = o_base + o_offs
    tl.store(o_ptrs, acc.to(tl.bfloat16), mask=row_mask[:, None])

    # Store LSE
    lse_ptrs = lse_base + offs_m * stride_lses
    tl.store(lse_ptrs, lse_out, mask=row_mask)


def run(Q, K, V, O, LSE):
    """
    Multi-head attention forward pass.
    
    O = softmax(Q @ K^T / sqrt(D)) @ V
    LSE = log(sum(exp(P))) where P = Q @ K^T / sqrt(D)
    
    Destination-passing interface: writes into preallocated O and LSE.
    """
    torch.cuda.set_device(Q.device)

    B, H, S, D = Q.shape[0], Q.shape[1], Q.shape[2], Q.shape[3]

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