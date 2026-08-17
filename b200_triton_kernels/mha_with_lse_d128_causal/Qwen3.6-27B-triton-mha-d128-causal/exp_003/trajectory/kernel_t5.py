import torch
import triton
import triton.language as tl


@triton.jit
def _mha_kernel(
    Q, K, V, O, LSE,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lse_b, stride_lse_h, stride_lse_s,
    H, S, D,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
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
    offs_m = block_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_d = tl.arange(0, D)                              # [D]
    mask_m = offs_m[:, None] < S                          # [BLOCK_M, 1]
    mask_d = offs_d[None, :] < D                           # [1, D]

    # Load Q tile [BLOCK_M, D] once
    q_ptr = Q + b * stride_qb + h * stride_qh
    q_ptrs = q_ptr + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=mask_m & mask_d, other=0.0).to(tl.float32)

    # Online softmax state
    acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)
    n_sm_states = tl.full((BLOCK_M,), 1.0, dtype=tl.float32)

    num_pid_n = tl.cdiv(S, BLOCK_N)

    for start_n in range(num_pid_n):
        offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

        # Early stop when all keys are past all queries in this tile
        last_offs_m = block_m * BLOCK_M + BLOCK_M - 1
        if start_n * BLOCK_N > last_offs_m:
            break

        # Load K [BLOCK_N, D]
        k_ptr = K + b * stride_kb + h * stride_kh
        k_ptrs = k_ptr + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=(offs_n[:, None] < S) & mask_d, other=0.0).to(tl.float32)

        # Load V [BLOCK_N, D]
        v_ptr = V + b * stride_vb + h * stride_vh
        v_ptrs = v_ptr + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=(offs_n[:, None] < S) & mask_d, other=0.0).to(tl.float32)

        # Compute scores [BLOCK_M, BLOCK_N]
        scores = tl.dot(q, k.T) * scale

        # Causal mask
        causal_mask = offs_n[None, :] <= offs_m[:, None]
        scores = tl.where(causal_mask, scores, float('-inf'))

        # Online softmax step
        m_ij = tl.max(scores, axis=1)
        new_m = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - new_m)
        p = tl.exp(scores - new_m[:, None])

        acc = alpha[:, None] * acc + tl.dot(p, v)
        n_sm_states = alpha * n_sm_states + tl.sum(p, axis=1)
        m_i = new_m

    # Epilogue
    inv_denom = 1.0 / n_sm_states
    acc_norm = acc * inv_denom[:, None]

    # Store O
    o_ptr = O + b * stride_ob + h * stride_oh
    o_ptrs = o_ptr + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc_norm.to(tl.bfloat16), mask=mask_m & mask_d)

    # Store LSE
    lse_val = m_i + tl.log(n_sm_states)
    lse_ptr = LSE + b * stride_lse_b + h * stride_lse_h
    lse_ptrs = lse_ptr + offs_m * stride_lse_s
    tl.store(lse_ptrs, lse_val, mask=offs_m < S)


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
    BLOCK_N = 64

    num_pid_m = triton.cdiv(S, BLOCK_M)
    grid = (B * H * num_pid_m,)

    _mha_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        H, S, D,
        scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=3,
    )