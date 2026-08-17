import torch
import triton
import triton.language as tl


@triton.jit
def _attention_kernel(
    q_base,
    k_base,
    v_base,
    o_base,
    lse_base,
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

    q_ptrs = q_base + bid_b * stride_qb + bid_h * stride_qh + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    q_mask = (offs_m[:, None] < S) & (offs_d[None, :] < D)
    Q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    k_ptrs = k_base + bid_b * stride_kb + bid_h * stride_kh
    v_ptrs = v_base + bid_b * stride_vb + bid_h * stride_vh

    num_steps = tl.cdiv(S, BLOCK_N)

    m_ij = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_ij = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    for start_n in range(0, num_steps):
        offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)

        k_ptrs_offset = k_ptrs + offs_n[None, :] * stride_ks + offs_d[:, None] * stride_kd
        k_mask = (offs_n[None, :] < S) & (offs_d[:, None] < D)
        K = tl.load(k_ptrs_offset, mask=k_mask, other=0.0)

        v_ptrs_offset = v_ptrs + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
        v_mask = (offs_n[:, None] < S) & (offs_d[None, :] < D)
        V = tl.load(v_ptrs_offset, mask=v_mask, other=0.0)

        QK = tl.dot(Q, K) * scale

        m_ij_new = tl.maximum(m_ij, tl.max(QK, axis=1, keep_dims=False))

        alpha = tl.exp(m_ij - m_ij_new)
        p = tl.exp(QK - m_ij_new[:, None])

        acc = acc * alpha[:, None]
        acc = acc + tl.dot(p.T.to(tl.float32), V)

        l_ij = l_ij * alpha + tl.sum(p, axis=1, keep_dims=False)
        m_ij = m_ij_new

    inv_l = 1.0 / l_ij
    O_result = acc * inv_l[:, None]

    o_ptrs = o_base + bid_b * stride_ob + bid_h * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(o_ptrs, O_result, mask=q_mask)

    lse_vals = m_ij + tl.log(l_ij)
    lse_ptrs = lse_base + bid_b * stride_lseb + bid_h * stride_lseh + offs_m * stride_lses
    tl.store(lse_ptrs, lse_vals, mask=(offs_m < S))


def run(Q, K, V, O, LSE):
    """Multi-head attention forward returning O and LSE."""
    torch.cuda.set_device(Q.device)

    B, H, S, D = Q.shape[0], Q.shape[1], Q.shape[2], Q.shape[3]
    scale = 1.0 / (D ** 0.5)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = 32
    num_warps = 4
    num_stages = 3

    grid_bh = B * H
    grid_m = triton.cdiv(S, BLOCK_M)
    grid = (grid_bh, grid_m)

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