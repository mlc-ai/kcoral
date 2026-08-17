import torch
import triton
import triton.language as tl


@triton.jit
def _mha_causal_fwd_kernel(
    Q, K, V,
    O, LSE,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lsb, stride_lsh, stride_ls,
    num_heads, seq_len,
    softmax_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    # Program identity
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    pid_b = pid_bh // num_heads
    pid_h = pid_bh % num_heads

    # Base pointers for this batch/head
    q_base = Q + pid_b * stride_qb + pid_h * stride_qh
    k_base = K + pid_b * stride_kb + pid_h * stride_kh
    v_base = V + pid_b * stride_vb + pid_h * stride_vh
    o_base = O + pid_b * stride_ob + pid_h * stride_oh
    lse_base = LSE + pid_b * stride_lsb + pid_h * stride_lsh

    # Query position indices [BLOCK_M]
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # Head dim indices [HEAD_DIM]
    offs_d = tl.arange(0, HEAD_DIM)

    # Query validity mask
    q_mask = offs_m < seq_len

    # Load Q tile [BLOCK_M, HEAD_DIM] and cast to fp32 immediately
    q_ptrs = q_base + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float32)

    # Online softmax state
    acc_o = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # Iterate over key/value tiles
    kv_tiles = tl.cdiv(seq_len, BLOCK_N)

    for ti in range(kv_tiles):
        # Key position indices [BLOCK_N]
        offs_n = ti * BLOCK_N + tl.arange(0, BLOCK_N)
        # Key validity mask
        k_mask = offs_n < seq_len

        # Load K tile [BLOCK_N, HEAD_DIM] -> fp32
        k_ptrs = k_base + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=k_mask[:, None], other=0.0).to(tl.float32)

        # Scores [BLOCK_M, BLOCK_N] = Q @ K^T * scale
        scores = tl.dot(q, k.T) * softmax_scale

        # Build combined mask: valid query AND valid key AND causal (j <= i)
        causal = (offs_m[:, None] >= offs_n[None, :]) & q_mask[:, None] & k_mask[None, :]
        scores = tl.where(causal, scores, float("-inf"))

        # --- Online softmax step ---
        m_ij = tl.max(scores, axis=1)                    # [BLOCK_M]
        m_new = tl.maximum(m_i, m_ij)                    # [BLOCK_M]
        alpha = tl.exp(m_i - m_new)                      # [BLOCK_M]
        p = tl.exp(scores - m_new[:, None])              # [BLOCK_M, BLOCK_N]

        # Decay old accumulation
        acc_o = acc_o * alpha[:, None]                   # [BLOCK_M, HEAD_DIM]

        # Load V tile [BLOCK_N, HEAD_DIM] -> fp32
        v_ptrs = v_base + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0).to(tl.float32)

        # P @ V addition
        acc_o = tl.dot(p, v, acc_o)                     # [BLOCK_M, HEAD_DIM]

        # Update normalization denominator
        p_sum = tl.sum(p, axis=1)                       # [BLOCK_M]
        l_i = alpha * l_i + p_sum                       # [BLOCK_M]
        m_i = m_new                                     # [BLOCK_M]

    # Write output O as bf16
    o_ptrs = o_base + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc_o.to(tl.bfloat16), mask=q_mask[:, None])

    # Write LSE as fp32
    lse_ptrs = lse_base + offs_m * stride_ls
    lse_val = m_i + tl.log(l_i)
    tl.store(lse_ptrs, lse_val, mask=q_mask)


def run(Q, K, V, O, LSE):
    """Compute causal MHA forward (O, LSE) into pre-allocated output tensors."""
    torch.cuda.set_device(Q.device)

    B, H, S, D = Q.shape
    scale = 1.0 / float(D ** 0.5)

    BLOCK_M = 64
    BLOCK_N = 64

    grid = (B * H, triton.cdiv(S, BLOCK_M))

    _mha_causal_fwd_kernel[grid](
        Q, K, V,
        O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        H, S,
        scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        HEAD_DIM=D,
        num_warps=4,
        num_stages=3,
    )