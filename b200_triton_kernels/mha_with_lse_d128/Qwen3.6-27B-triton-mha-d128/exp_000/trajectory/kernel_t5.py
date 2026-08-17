import torch
import triton
import triton.language as tl


@triton.jit
def _attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    lse_ptr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    B, H, S, D,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    bid_b = pid_bh // H
    bid_h = pid_bh % H

    if bid_b >= B or bid_h >= H:
        return

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    # Query mask for storing output
    q_valid = offs_m[:, None] < S

    # Load Q tile [BLOCK_M, BLOCK_D]
    q_offsets = bid_b * stride_qb + bid_h * stride_qh \
                + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    q_load_mask = q_valid & (offs_d[None, :] < D)
    Q = tl.load(q_ptr + q_offsets, mask=q_load_mask, other=0.0)

    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.full((BLOCK_M,), 1.0, dtype=tl.float32)

    num_steps = tl.cdiv(S, BLOCK_N)

    for start_n in range(num_steps):
        offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_valid = offs_n[None, :] < S

        # Load K tile [BLOCK_N, BLOCK_D]
        k_offsets = bid_b * stride_kb + bid_h * stride_kh \
                    + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        k_mask = (offs_n[:, None] < S) & (offs_d[None, :] < D)
        K = tl.load(k_ptr + k_offsets, mask=k_mask, other=0.0)

        # Load V tile [BLOCK_N, BLOCK_D]
        v_offsets = bid_b * stride_vb + bid_h * stride_vh \
                    + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
        v_mask = (offs_n[:, None] < S) & (offs_d[None, :] < D)
        V = tl.load(v_ptr + v_offsets, mask=v_mask, other=0.0)

        # Attention scores [BLOCK_M, BLOCK_N]
        s = tl.dot(Q, K.T) * scale

        # Combined mask: both query row and key col must be valid
        full_mask = q_valid & n_valid
        s = tl.where(full_mask, s, float("-inf"))

        # Online softmax update
        m_i_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(s - m_i_new[:, None])

        # Accumulate P @ V [BLOCK_M, BLOCK_D]
        acc = acc * alpha[:, None] + tl.dot(p, V.to(tl.float32))
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_i_new

    # Normalize and convert to bf16
    inv_l = 1.0 / l_i
    O_result = acc * inv_l[:, None]

    # Store O [BLOCK_M, BLOCK_D]
    o_offsets = bid_b * stride_ob + bid_h * stride_oh \
                + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(o_ptr + o_offsets, O_result.to(tl.bfloat16), mask=q_load_mask)

    # Store LSE [BLOCK_M]
    lse_val = m_i + tl.log(l_i)
    lse_offsets = bid_b * stride_lseb + bid_h * stride_lseh + offs_m * stride_lses
    tl.store(lse_ptr + lse_offsets, lse_val, mask=(offs_m < S))


def run(Q, K, V, O, LSE):
    """Multi-head attention forward returning O and LSE."""
    torch.cuda.set_device(Q.device)

    B, H, S, D = Q.shape
    scale = 1.0 / (D ** 0.5)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = 32
    num_warps = 4
    num_stages = 3

    grid = (B * H, triton.cdiv(S, BLOCK_M))

    _attention_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        B, H, S, D, scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        num_warps=num_warps,
        num_stages=num_stages,
    )