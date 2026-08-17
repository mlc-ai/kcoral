import torch
import triton
import triton.language as tl


@triton.jit
def _causal_mha_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    lse_ptr,
    stride_qs, stride_qd,
    stride_ks, stride_kd,
    stride_vs, stride_vd,
    stride_os, stride_od,
    stride_ls,
    stride_qbh,
    stride_lseb,
    S,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    bh_id = tl.program_id(0)
    pid_m = tl.program_id(1)

    # Base offsets for this (batch, head) group
    q_base = q_ptr + bh_id * stride_qbh
    k_base = k_ptr + bh_id * stride_qbh
    v_base = v_ptr + bh_id * stride_qbh
    o_base = o_ptr + bh_id * stride_qbh
    lse_base = lse_ptr + bh_id * stride_lseb

    # Query row indices for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < S

    # Head-dimension column indices
    offs_d = tl.arange(0, BLOCK_D)

    # Load Q tile: [BLOCK_M, BLOCK_D] as bf16
    q_ptrs = q_base + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    q_tile = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    # Online softmax accumulators in fp32
    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    scale = 1.0 / tl.sqrt(tl.cast(BLOCK_D, tl.float32))

    # Iterate over KV tiles
    num_steps = tl.cdiv(S, BLOCK_N)
    for step in range(num_steps):
        start_n = step * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S

        # Load K tile: physical shape [BLOCK_N, BLOCK_D], read as [BLOCK_N, BLOCK_D]
        k_ptrs = k_base + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        k_tile = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)

        # Load V tile: [BLOCK_N, BLOCK_D]
        v_ptrs = v_base + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
        v_tile = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

        # Attention scores: Q[BLOCK_M, BLOCK_D] @ K^T[BLOCK_D, BLOCK_N] -> [BLOCK_M, BLOCK_N]
        scores = tl.dot(q_tile, k_tile.T) * scale

        # Causal mask: query_pos >= key_pos (lower triangular including diagonal)
        causal = offs_m[:, None] >= offs_n[None, :]
        valid = causal & mask_m[:, None] & mask_n[None, :]

        masked_scores = tl.where(valid, scores, -float("inf"))

        # Online softmax update
        new_m = tl.maximum(m_i, tl.max(masked_scores, axis=1))

        alpha = tl.exp(m_i - new_m)

        p = tl.exp(masked_scores - new_m[:, None])

        new_l = alpha * l_i + tl.sum(p, axis=1)

        # Compute P @ V: p is [BLOCK_M, BLOCK_N] fp32, v is [BLOCK_N, BLOCK_D] bf16
        # Downcast p to bf16 for the dot, accumulate in fp32
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v_tile, out_dtype=tl.float32)

        m_i = new_m
        l_i = new_l

    # Normalize output
    denom = tl.where(l_i > 0.0, l_i, 1.0)
    acc = acc / denom[:, None]

    # Store O in bf16
    o_ptrs = o_base + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(tl.bfloat16), mask=mask_m[:, None])

    # Store LSE in fp32
    lse_val = m_i + tl.log(denom)
    lse_ptrs = lse_base + offs_m * stride_ls
    tl.store(lse_ptrs, lse_val, mask=mask_m)


def run(Q, K, V, O, LSE):
    """Causal multi-head attention forward returning O and LSE.
    
    Inputs: Q, K, V shaped (B, H, S, D) bfloat16
    Outputs: O shaped (B, H, S, D) bfloat16, LSE shaped (B, H, S) float32
    """
    torch.cuda.set_device(Q.device)

    B, H, S, D = Q.shape

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = D

    BH = B * H
    num_pid_m = triton.cdiv(S, BLOCK_M)

    grid = (BH, num_pid_m)

    _causal_mha_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(2), Q.stride(3),
        K.stride(2), K.stride(3),
        V.stride(2), V.stride(3),
        O.stride(2), O.stride(3),
        LSE.stride(2),
        Q.stride(0),
        LSE.stride(0),
        S,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=3,
    )