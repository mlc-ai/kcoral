import torch
import triton
import triton.language as tl


@triton.jit
def _mha_forward_kernel(
    Q, K, V, O, LSE,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lse_b, stride_lse_h, stride_lse_s,
    seq_len,
    num_heads,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Causal multi-head attention forward kernel with online softmax."""
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    batch_idx = pid_bh // num_heads
    head_idx = pid_bh % num_heads

    # Base offsets for this (batch, head) slice
    q_off = Q + batch_idx * stride_qb + head_idx * stride_qh
    k_off = K + batch_idx * stride_kb + head_idx * stride_kh
    v_off = V + batch_idx * stride_vb + head_idx * stride_vh
    o_off = O + batch_idx * stride_ob + head_idx * stride_oh
    lse_off = LSE + batch_idx * stride_lse_b + head_idx * stride_lse_h

    # Index arrays
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)

    # Query valid mask [BLOCK_M]
    q_mask = offs_m < seq_len

    # Load Q tile once: [BLOCK_M, D] as bf16, reused across KV iterations
    q_ptrs = q_off + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs,
                mask=(q_mask[:, None]) & (offs_d[None, :] < D),
                other=0.0)

    # Scale factor in fp32
    scale = 1.0 / tl.sqrt(tl.cast(D, tl.float32))

    # Online-softmax state (all FP32)
    m_i = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)

    # Absolute query positions for causal comparison
    q_pos = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    num_steps = tl.cdiv(seq_len, BLOCK_N)

    for start_n in range(num_steps):
        offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
        kv_pos = start_n * BLOCK_N + tl.arange(0, BLOCK_N)

        # Keys valid mask [BLOCK_N]
        k_mask = offs_n < seq_len

        # Load K tile [BLOCK_N, D]
        k_ptrs = k_off + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs,
                    mask=(k_mask[:, None]) & (offs_d[None, :] < D),
                    other=0.0)

        # Load V tile [BLOCK_N, D]
        v_ptrs = v_off + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs,
                    mask=(k_mask[:, None]) & (offs_d[None, :] < D),
                    other=0.0)

        # Attention scores [BLOCK_M, BLOCK_N], fp32 accumulation via tl.dot
        scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale

        # Causal + bounds mask: key_pos <= query_pos AND within sequence
        causal = q_pos[:, None] >= kv_pos[None, :]
        bounds = q_mask[:, None] & k_mask[None, :]
        valid = causal & bounds

        # Mask invalid entries with -inf
        scores = tl.where(valid, scores, float('-inf'))

        # --- Online softmax update (stable form) ---
        row_max = tl.max(scores, axis=1)
        m_i_new = tl.maximum(m_i, row_max)

        # Probability weights – only compute exp where the entry is truly valid
        p = tl.where(valid, tl.exp(scores - m_i_new[:, None]), 0.0)

        # Only scale previous accumulators if we previously had valid data
        was_valid = m_i > float('-inf')
        alpha = tl.where(was_valid, tl.exp(m_i - m_i_new), 0.0)

        # Update running statistics
        l_i = tl.where(was_valid, alpha * l_i, 0.0) + tl.sum(p, axis=1)
        acc = tl.where(was_valid[:, None], alpha[:, None] * acc, 0.0) \
              + tl.dot(p.to(tl.bfloat16), v.to(tl.bfloat16), out_dtype=tl.float32)
        m_i = m_i_new

    # --- Epilogue ---
    # Normalize output: safe division with valid-check
    has_valid = l_i > 0.0
    o_val = tl.where(has_valid[:, None], acc / l_i[:, None], 0.0)
    lse_val = tl.where(has_valid, m_i + tl.log(l_i), float('-inf'))

    # Store O: bf16 [BLOCK_M, D]
    o_ptrs = o_off + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(o_ptrs, o_val.to(tl.bfloat16),
             mask=q_mask[:, None] & (offs_d[None, :] < D))

    # Store LSE: fp32 [BLOCK_M]
    lse_ptrs = lse_off + offs_m * stride_lse_s
    tl.store(lse_ptrs, lse_val, mask=q_mask)


def run(Q, K, V, O, LSE):
    """Causal multi-head attention forward with LSE output.

    Inputs  (defined): Q [B, H, S, D] bf16, K [B, H, S, D] bf16,
                       V [B, H, S, D] bf16
    Outputs (preallocated): O [B, H, S, D] bf16,
                            LSE [B, H, S] fp32
    """
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape

    BLOCK_M = 64
    BLOCK_N = 64

    grid = (triton.cdiv(S, BLOCK_M), B * H)

    _mha_forward_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        S, H,
        D=D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=3,
    )