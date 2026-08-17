import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
    ],
    key=['S'],
)
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
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """FlashAttention forward kernel with online softmax."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(S, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_bh = pid // num_pid_m
    pid_h = pid_bh % H
    pid_b = pid_bh // H

    # Advance base pointers to current (batch, head) slice
    Q += pid_b * stride_qb + pid_h * stride_qh
    K += pid_b * stride_kb + pid_h * stride_kh
    V += pid_b * stride_vb + pid_h * stride_vh
    Out += pid_b * stride_ob + pid_h * stride_oh
    Lse += pid_b * stride_lseb + pid_h * stride_lseh

    # Index tensors
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    # Load Q tile: [BLOCK_M, BLOCK_D]
    q_ptrs = Q + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=(offs_m[:, None] < S), other=0.0)

    # Initialize accumulators for online softmax
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    m_i = tl.zeros((BLOCK_M,), dtype=tl.float32) + float('-inf')
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)

    num_pid_n = tl.cdiv(S, BLOCK_N)

    for start_n in range(num_pid_n):
        offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n[:, None] < S

        # Load K tile: [BLOCK_N, BLOCK_D]
        k_ptrs = K + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=n_mask, other=0.0)

        # Compute attention scores: [BLOCK_M, BLOCK_D] @ [BLOCK_D, BLOCK_N] -> [BLOCK_M, BLOCK_N]
        att = tl.dot(q, k.T) * scale

        # Online softmax update
        m_i_old = m_i
        m_i = tl.maximum(m_i, tl.max(att, axis=1, keep_dims=False))

        # Rescaling factors
        p = tl.exp(att - m_i[:, None])
        alpha = tl.exp(m_i_old - m_i)

        # Update normalization denominator
        l_i = alpha * l_i + tl.sum(p, axis=1, keep_dims=False)
        # Rescale accumulated output
        acc = alpha[:, None] * acc

        # Load V tile: [BLOCK_N, BLOCK_D]
        v_ptrs = V + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=n_mask, other=0.0)

        # Accumulate: acc += p @ v  [BLOCK_M, BLOCK_N] @ [BLOCK_N, BLOCK_D] -> [BLOCK_M, BLOCK_D]
        acc = acc + tl.dot(p, v)

    # Final normalization
    acc = acc / l_i[:, None]

    # Store output O: cast to bf16
    out_ptrs = Out + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=(offs_m[:, None] < S))

    # Store LSE = m_i + log(l_i) in float32
    lse_val = m_i + tl.log(l_i)
    tl.store(Lse + offs_m * stride_lses, lse_val, mask=(offs_m < S))


def run(Q, K, V, O, LSE):
    """Compute multi-head attention output O and log-sum-exp LSE.

    Parameters
    ----------
    Q : torch.Tensor, shape [B, H, S, D], dtype bf16
    K : torch.Tensor, shape [B, H, S, D], dtype bf16
    V : torch.Tensor, shape [B, H, S, D], dtype bf16
    O : torch.Tensor, shape [B, H, S, D], dtype bf16 (preallocated output)
    LSE : torch.Tensor, shape [B, H, S], dtype float32 (preallocated output)
    """
    torch.cuda.set_device(Q.device)

    B, H, S, D = Q.shape
    scale = 1.0 / (D ** 0.5)

    # Grid: one program per (batch, head, query-tile)
    grid = lambda META: (B * H * triton.cdiv(S, META["BLOCK_M"]),)

    _mha_fwd_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        B, H, S, D,
        scale,
        BLOCK_D=D,
    )