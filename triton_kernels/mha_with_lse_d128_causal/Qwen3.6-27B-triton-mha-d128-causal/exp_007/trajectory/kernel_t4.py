import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _causal_mha_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    lse_ptr,
    stride_bh,
    stride_s_qkv,
    stride_d_qkv,
    stride_lse_bh,
    B,
    H,
    S,
    NUM_SMS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)

    # Total number of query tiles we need to process
    num_batches_heads = B * H
    num_pid_m = tl.cdiv(S, BLOCK_M)
    num_tiles = num_batches_heads * num_pid_m

    # Persistent scheduling: each program processes tiles with stride NUM_SMS
    for tile_id in range(pid, num_tiles, NUM_SMS):
        bh_id = tile_id // num_pid_m
        pid_m = tile_id % num_pid_m

        batch_idx = bh_id // H
        head_idx = bh_id % H

        # Base offsets for this (batch, head) group
        q_base = q_ptr + bh_id * stride_bh
        k_base = k_ptr + bh_id * stride_bh
        v_base = v_ptr + bh_id * stride_bh
        o_base = o_ptr + bh_id * stride_bh
        lse_base = lse_ptr + bh_id * stride_lse_bh

        # Query row indices
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S

        offs_d = tl.arange(0, BLOCK_D)

        # Load Q tile once outside the K/V loop
        q_ptrs = q_base + offs_m[:, None] * stride_s_qkv + offs_d[None, :] * stride_d_qkv
        q_tile = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

        # Online softmax accumulators in fp32
        m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

        scale = 1.0 / tl.sqrt(tl.cast(BLOCK_D, tl.float32))

        # Iterate over KV tiles along sequence dimension
        num_k_steps = tl.cdiv(S, BLOCK_N)
        for step in range(num_k_steps):
            start_n = step * BLOCK_N
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < S

            # Load K tile: [BLOCK_N, BLOCK_D]
            k_ptrs = k_base + offs_n[:, None] * stride_s_qkv + offs_d[None, :] * stride_d_qkv
            k_tile = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)

            # Load V tile: [BLOCK_N, BLOCK_D]
            v_ptrs = v_base + offs_n[:, None] * stride_s_qkv + offs_d[None, :] * stride_d_qkv
            v_tile = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

            # Scores: [BLOCK_M, BLOCK_D] x [BLOCK_N, BLOCK_D]^T -> [BLOCK_M, BLOCK_N]
            scores = tl.dot(q_tile, k_tile.T, out_dtype=tl.float32) * scale

            # Causal mask: query_pos >= key_pos
            causal = offs_m[:, None] >= offs_n[None, :]
            valid = causal & mask_m[:, None] & mask_n[None, :]

            masked_scores = tl.where(valid, scores, -float("inf"))

            # Online softmax update
            new_m = tl.maximum(m_i, tl.max(masked_scores, axis=1))

            alpha = tl.exp(m_i - new_m)

            p = tl.exp(masked_scores - new_m[:, None])

            new_l = alpha * l_i + tl.sum(p, axis=1)

            # P @ V: convert p to bf16 for dot, accumulate in fp32
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v_tile, out_dtype=tl.float32)

            m_i = new_m
            l_i = new_l

        # Normalize and store output
        denom = tl.where(l_i > 0.0, l_i, 1.0)
        acc = acc / denom[:, None]

        o_ptrs = o_base + offs_m[:, None] * stride_s_qkv + offs_d[None, :] * stride_d_qkv
        tl.store(o_ptrs, acc.to(tl.bfloat16), mask=mask_m[:, None])

        # Store LSE
        lse_val = m_i + tl.log(denom)
        lse_ptrs = lse_base + offs_m
        tl.store(lse_ptrs, lse_val, mask=mask_m)


def run(Q, K, V, O, LSE):
    """Causal multi-head attention forward returning O and LSE."""
    torch.cuda.set_device(Q.device)

    B, H, S, D = Q.shape

    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_D = D

    # Use device count for persistent scheduling
    num_sms = 132  # Hopper/H100

    total_batches_heads = B * H
    num_pid_m = triton.cdiv(S, BLOCK_M)
    total_tiles = total_batches_heads * num_pid_m
    grid_size = min(num_sms, total_tiles)
    grid = (grid_size,)

    stride_bh = S * D
    stride_s_qkv = D
    stride_d_qkv = 1
    stride_lse_bh = S

    _causal_mha_kernel[grid](
        Q, K, V, O, LSE,
        stride_bh, stride_s_qkv, stride_d_qkv, stride_lse_bh,
        B, H, S,
        NUM_SMS=num_sms,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        num_warps=8,
        num_stages=3,
    )