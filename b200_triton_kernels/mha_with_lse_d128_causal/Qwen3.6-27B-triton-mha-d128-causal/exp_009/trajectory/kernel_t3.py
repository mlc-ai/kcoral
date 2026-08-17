import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _mha_forward_kernel(
    Q, K, V, O, LSE,
    stride_qs, stride_qd,
    stride_ks, stride_kd,
    stride_vs, stride_vd,
    stride_os, stride_od,
    stride_lse_s,
    seq_len,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Causal MHA forward kernel with online softmax, using pointer loads."""
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)

    q_mask = offs_m < seq_len

    # Load Q tile once [BLOCK_M, D] as bf16
    q_ptrs = Q + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)

    scale = 1.0 / tl.sqrt(tl.cast(D, tl.float32))

    # Online-softmax state (FP32)
    m_i = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)

    q_pos = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    num_steps = tl.cdiv(seq_len, BLOCK_N)

    for start_n in range(num_steps):
        offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
        kv_pos = start_n * BLOCK_N + tl.arange(0, BLOCK_N)

        k_mask = offs_n < seq_len

        # Load K tile [BLOCK_N, D]
        k_ptrs = K + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=k_mask[:, None], other=0.0)

        # Load V tile [BLOCK_N, D]
        v_ptrs = V + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0)

        # Attention scores [BLOCK_M, BLOCK_N] – fp32 via tensor cores
        scores = tl.dot(q, tl.trans(k)) * scale

        # Causal + bounds mask
        causal = q_pos[:, None] >= kv_pos[None, :]
        valid = causal & q_mask[:, None] & k_mask[None, :]

        scores = tl.where(valid, scores, float('-inf'))

        # --- Online softmax update (stable form) ---
        row_max = tl.max(scores, axis=1)
        m_i_new = tl.maximum(m_i, row_max)

        p = tl.where(valid, tl.exp(scores - m_i_new[:, None]), 0.0)

        was_valid = m_i > float('-inf')
        alpha = tl.where(was_valid, tl.exp(m_i - m_i_new), 0.0)

        l_i = tl.where(was_valid, alpha * l_i, 0.0) + tl.sum(p, axis=1)
        acc = tl.where(was_valid[:, None], alpha[:, None] * acc, 0.0) \
              + tl.dot(p, v.to(tl.float32))
        m_i = m_i_new

    # --- Epilogue ---
    has_valid = l_i > 0.0
    o_val = tl.where(has_valid[:, None], acc / l_i[:, None], 0.0)
    lse_val = tl.where(has_valid, m_i + tl.log(l_i), float('-inf'))

    # Store O: fp32 -> bf16
    o_ptrs = O + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(o_ptrs, o_val.to(tl.bfloat16), mask=q_mask[:, None])

    # Store LSE: fp32
    lse_ptrs = LSE + offs_m * stride_lse_s
    tl.store(lse_ptrs, lse_val, mask=q_mask)


def run(Q, K, V, O, LSE):
    """Causal multi-head attention forward with LSE output.

    Inputs:  Q [B, H, S, D] bf16, K [B, H, S, D] bf16, V [B, H, S, D] bf16
    Outputs: O [B, H, S, D] bf16, LSE [B, H, S] fp32
    """
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape

    BLOCK_M = 128
    BLOCK_N = 64

    grid = (triton.cdiv(S, BLOCK_M), B * H)

    _mha_forward_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(2), Q.stride(3),
        K.stride(2), K.stride(3),
        V.stride(2), V.stride(3),
        O.stride(2), O.stride(3),
        LSE.stride(2),
        S,
        D=D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=3,
    )