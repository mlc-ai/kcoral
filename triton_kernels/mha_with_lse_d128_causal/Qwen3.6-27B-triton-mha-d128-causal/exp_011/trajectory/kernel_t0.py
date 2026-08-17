import torch
import triton
import triton.language as tl


@triton.jit
def _mha_fwd_kernel(
    Q,
    K,
    V,
    O,
    LSE,
    stride_qm, stride_qh, stride_qs, stride_qd,
    stride_km, stride_kh, stride_ks, stride_kd,
    stride_vm, stride_vh, stride_vs, stride_vd,
    stride_lm, stride_lh, stride_ls,
    S,
    D,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Causal multi-head attention kernel with online softmax."""
    pid = tl.program_id(0)

    # Decode flat program id into (batch, head, query_tile)
    B = stride_qm // stride_qh
    H = stride_qh // stride_qs
    sq_tiles = tl.cdiv(S, BLOCK_M)
    pid_sq = pid % sq_tiles
    pid_h = (pid // sq_tiles) % H
    pid_b = pid // (sq_tiles * H)

    # Query offset for this program
    off_m = pid_sq * BLOCK_M + tl.arange(0, BLOCK_M)
    q_end = pid_sq * BLOCK_M + BLOCK_M

    # For causal attention, each query attends to keys up to min(S, q_end)
    # Limit num_blocks to ceil(min(S, q_end) / BLOCK_N)
    kv_limit = tl.minimum(S, q_end)
    num_blocks = tl.cdiv(kv_limit, BLOCK_N)

    # Head dimension offsets
    off_d = tl.arange(0, BLOCK_D)

    # Base pointers for this batch/head
    base_q = Q + pid_b * stride_qh + pid_h * stride_qs
    base_k = K + pid_b * stride_qh + pid_h * stride_qs
    base_v = V + pid_b * stride_qh + pid_h * stride_qs
    base_o = O + pid_b * stride_qh + pid_h * stride_qs
    base_lse = LSE + pid_b * stride_lh + pid_h * stride_ls

    # Load Q once (outside the kv loop)
    q_ptrs = base_q + off_m[:, None] * stride_qs + off_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=(off_m[:, None] < S) & (off_d[None, :] < D), other=0.0)

    # Initialize accumulators
    acc_o = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    m_old = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    lse_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for blk_idx in range(num_blocks):
        start_n = blk_idx * BLOCK_N

        # Build K pointers
        off_n = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = base_k + off_n[:, None] * stride_ks + off_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=(off_n[:, None] < S) & (off_d[None, :] < D), other=0.0)

        # QK^T matmul
        attn = tl.dot(q, k, acc=None, out_dtype=tl.float32)

        # Causal mask: query pos must be >= key pos
        causal_mask = (off_m[:, None] >= off_n[None, :])
        attn = tl.where(causal_mask, attn, float('-inf'))

        # Online softmax: track running max
        m_new = tl.maximum(m_old, tl.max(attn, axis=1))
        alpha = tl.exp(m_old - m_new)

        # Rescale accumulated output
        acc_o = acc_o * alpha[:, None]

        # Compute normalized probabilities and update LSE sum
        p_row_j = tl.exp(attn - m_new[:, None])
        lse_sum = lse_sum * alpha + tl.sum(p_row_j, axis=1)

        # Load V and accumulate
        v_ptrs = base_v + off_n[:, None] * stride_vs + off_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=(off_n[:, None] < S) & (off_d[None, :] < D), other=0.0)
        acc_o = tl.dot(p_row_j, v, acc=acc_o, out_dtype=tl.float32)

        # Update m_old for next iteration
        m_old = m_new

    # Store output (normalize by lse_sum)
    o_ptrs = base_o + off_m[:, None] * stride_qs + off_d[None, :] * stride_qd
    mask_out = (off_m[:, None] < S) & (off_d[None, :] < D)
    tl.store(o_ptrs, (acc_o / lse_sum[:, None]).to(O.dtype.element_ty), mask=mask_out)

    # Store LSE = log(sum(exp(scores))) = m_final + log(lse_sum)
    lse_ptrs = base_lse + off_m * stride_ls
    tl.store(lse_ptrs, (m_old + tl.log(lse_sum)).to(LSE.dtype.element_ty), mask=off_m < S)


def run(Q, K, V, O, LSE):
    """Causal multi-head attention forward pass with LSE output.

    Args:
        Q: Input query tensor, shape [B, H, S, D], dtype bf16
        K: Input key tensor, shape [B, H, S, D], dtype bf16
        V: Input value tensor, shape [B, H, S, D], dtype bf16
        O: Preallocated output tensor, shape [B, H, S, D], dtype bf16
        LSE: Preallocated LSE output tensor, shape [B, H, S], dtype fp32
    """
    torch.cuda.set_device(Q.device)

    B, H, S, D = Q.shape
    scale = 1.0 / float(D) ** 0.5

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = 32

    num_tiles = B * H * triton.cdiv(S, BLOCK_M)
    grid = (num_tiles,)

    _mha_fwd_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(2), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(2), K.stride(1), K.stride(2), K.stride(3),
        V.stride(2), V.stride(1), V.stride(2), V.stride(3),
        LSE.stride(2), LSE.stride(1), LSE.stride(2),
        S, D, scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=3,
    )