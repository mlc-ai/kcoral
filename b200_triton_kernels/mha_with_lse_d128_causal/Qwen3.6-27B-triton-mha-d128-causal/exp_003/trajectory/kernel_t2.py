import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=3),
        triton.Config({}, num_warps=4, num_stages=4),
        triton.Config({}, num_warps=8, num_stages=4),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=2),
    ],
    key=["S", "H"],
)
@triton.heuristics(
    values={
        "BLOCK_M": lambda META: 128,
        "BLOCK_N": lambda META: 64,
        "BLOCK_D": lambda META: 128,
    }
)
@triton.jit
def _mha_kernel(
    Q, K, V, O, LSE,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lse_b, stride_lse_h, stride_lse_s,
    H, S,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Causal multi-head attention forward with online softmax."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(S, BLOCK_M)

    # Decode flat program id -> (batch, head, query-block index)
    bh_idx = pid // num_pid_m
    block_m = pid % num_pid_m
    b = bh_idx // H
    h = bh_idx % H

    # Offsets
    off_m = block_m * BLOCK_M + tl.arange(0, BLOCK_M)    # [BLOCK_M] – query positions
    off_m_last = off_m[-1]                                # largest query pos in this tile
    off_d = tl.arange(0, BLOCK_D)                         # [BLOCK_D]

    # Load Q tile [BLOCK_M, BLOCK_D] once
    q_ptrs = Q + b * stride_qb + h * stride_qh + \
             off_m[:, None] * stride_qs + off_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=(off_m[:, None] < S), other=0.0).to(tl.float32)

    # Online softmax initial state
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)
    l_i = tl.full((BLOCK_M,), 1.0, dtype=tl.float32)

    num_pid_n = tl.cdiv(S, BLOCK_N)

    # Sweep across key blocks
    for start_n in range(num_pid_n):
        n_off = start_n * BLOCK_N + tl.arange(0, BLOCK_N)

        # Early exit: if every key position exceeds all query positions
        if start_n * BLOCK_N > off_m_last:
            break

        # Load K tile [BLOCK_N, BLOCK_D]
        k_ptrs = K + b * stride_kb + h * stride_kh + \
                 n_off[:, None] * stride_ks + off_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=(n_off[:, None] < S), other=0.0).to(tl.float32)

        # Load V tile [BLOCK_N, BLOCK_D]
        v_ptrs = V + b * stride_vb + h * stride_vh + \
                 n_off[:, None] * stride_vs + off_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=(n_off[:, None] < S), other=0.0).to(tl.float32)

        # Attention scores [BLOCK_M, BLOCK_N]
        qk = tl.dot(q, k.T) * scale

        # Causal mask: key_pos <= query_pos
        is_causal = n_off[None, :] <= off_m[:, None]
        qk = tl.where(is_causal, qk, float('-inf'))

        # Online softmax update
        m_ij = tl.max(qk, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)

        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])

        acc = alpha[:, None] * acc + tl.dot(p, v)
        l_i = alpha * l_i + tl.sum(p, axis=1)
        m_i = m_i_new

    # Epilogue: normalize and store
    acc_norm = acc / l_i[:, None]

    # Write O [BLOCK_M, BLOCK_D] -> bf16
    o_ptrs = O + b * stride_ob + h * stride_oh + \
             off_m[:, None] * stride_os + off_d[None, :] * stride_od
    tl.store(o_ptrs, acc_norm.to(tl.bfloat16),
             mask=(off_m[:, None] < S))

    # Write LSE [BLOCK_M] -> fp32
    lse_val = m_i + tl.log(l_i)
    lse_ptrs = LSE + b * stride_lse_b + h * stride_lse_h + off_m * stride_lse_s
    tl.store(lse_ptrs, lse_val, mask=off_m < S)


def run(Q, K, V, O, LSE):
    """
    Causal multi-head attention forward.

    Computes:
      O   = softmax(Q @ K^T / sqrt(D)) @ V          (bf16, [B,H,S,D])
      LSE = logsumexp(Q @ K^T / sqrt(D))  per-row   (fp32,  [B,H,S])

    with lower-triangular (causal) masking over (query_pos, key_pos).
    """
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape

    scale = 1.0 / float(D ** 0.5)

    BLOCK_M = 128
    num_pid_m = triton.cdiv(S, BLOCK_M)
    grid = (B * H * num_pid_m,)

    _mha_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        H, S,
        scale,
    )