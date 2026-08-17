import torch
import triton
import triton.language as tl


@triton.jit
def _mha_fwd_kernel(
    Q, K, V, Out, Lse,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    B, H, S, D,
    scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """FlashAttention forward with per-threadblock head loop for better occupancy."""
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m_base = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    m_mask_row = offs_m_base < S
    m_mask = m_mask_row[:, None]

    for h in range(H):
        # Advance base pointers to current (batch, head) slice
        Q_h = Q + pid_b * stride_qb + h * stride_qh
        K_h = K + pid_b * stride_kb + h * stride_kh
        V_h = V + pid_b * stride_vb + h * stride_vh
        Out_h = Out + pid_b * stride_ob + h * stride_oh
        Lse_h = Lse + pid_b * stride_lseb + h * stride_lseh

        # Load Q tile [BLOCK_M, D] in bf16
        q_ptrs = Q_h + offs_m_base[:, None] * stride_qs + offs_d[None, :] * stride_qd
        q = tl.load(q_ptrs, mask=m_mask, other=0.0)

        # Initialize accumulators
        acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)
        m_i = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)

        num_pid_n = tl.cdiv(S, BLOCK_N)

        for start_n in range(num_pid_n):
            offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
            n_mask_row = offs_n < S
            n_mask = n_mask_row[:, None]

            # Load K tile [BLOCK_N, D] in bf16
            k_ptrs = K_h + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
            k = tl.load(k_ptrs, mask=n_mask, other=0.0)

            # Q @ K^T: bf16 @ bf16 -> fp32 via Tensor Cores
            att = tl.dot(q, k.T, out_dtype=tl.float32) * scale

            # Online softmax update
            m_i_old = m_i
            m_i_new = tl.maximum(m_i, tl.max(att, axis=1, keep_dims=False))

            # Compute attention probabilities in fp32
            p = tl.exp(att - m_i_new[:, None])
            alpha = tl.exp(m_i_old - m_i_new)

            # Update normalization
            l_i = alpha * l_i + tl.sum(p, axis=1, keep_dims=False)
            acc = alpha[:, None] * acc

            # Load V tile [BLOCK_N, D] in bf16
            v_ptrs = V_h + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
            v = tl.load(v_ptrs, mask=n_mask, other=0.0)

            # p @ V: cast p to bf16 for fast tensor-core dot
            acc = acc + tl.dot(p.to(tl.bfloat16), v, out_dtype=tl.float32)

            m_i = m_i_new

        # Final normalization: O = acc / l_i
        acc = acc / l_i[:, None]

        # Store output O
        out_ptrs = Out_h + offs_m_base[:, None] * stride_os + offs_d[None, :] * stride_od
        tl.store(out_ptrs, acc.to(tl.bfloat16), mask=m_mask)

        # Store LSE = m_i + log(l_i)
        lse_val = m_i + tl.log(l_i)
        tl.store(Lse_h + offs_m_base * stride_lses, lse_val, mask=m_mask_row)


def run(Q, K, V, O, LSE):
    """Compute multi-head attention output O and log-sum-exp LSE."""
    torch.cuda.set_device(Q.device)

    B, H, S, D = Q.shape
    scale = 1.0 / (D ** 0.5)

    BLOCK_M = 64
    BLOCK_N = 128

    # Grid: (B, num_query_blocks) -- each CTA loops over all H heads
    grid = (B, triton.cdiv(S, BLOCK_M))

    _mha_fwd_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        B, H, S, D,
        scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=8,
        num_stages=3,
    )