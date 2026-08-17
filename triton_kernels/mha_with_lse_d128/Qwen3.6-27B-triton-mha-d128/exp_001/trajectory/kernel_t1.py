import torch
import triton
import triton.language as tl


@triton.jit
def _mha_fwd_kernel(
    Q, K, V, O, LSE,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    S, D,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
):
    """
    Multi-head attention forward using flash attention online softmax algorithm.
    Grid axes: (batch, head, m_block)
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)

    # Indexing vectors
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    m_mask = offs_m < S
    d_mask = offs_d < D

    # Base pointer offsets for this (batch, head)
    q_base = Q + pid_b * stride_qb + pid_h * stride_qh
    k_base = K + pid_b * stride_kb + pid_h * stride_kh
    v_base = V + pid_b * stride_vb + pid_h * stride_vh
    o_base = O + pid_b * stride_ob + pid_h * stride_oh
    lse_base = LSE + pid_b * stride_lseb + pid_h * stride_lseh

    # Load Q tile: [BLOCK_M, BLOCK_DMODEL]
    q_ptrs = q_base + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)

    # Attention scale factor: 1/sqrt(D), computed in fp32
    d_float = tl.cast(D, tl.float32)
    qk_scale = tl.reciprocal(tl.sqrt(d_float))

    # Running state for online softmax (all fp32)
    m_i = tl.full([BLOCK_M], -1e20, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # Iterate over KV sequence in blocks
    for start_n in range(0, S, BLOCK_N):
        n_idx = start_n + offs_n
        n_mask = n_idx < S

        # Load K: physical layout [N, D], want [D, N] for dot with Q[M,D]
        k_ptrs = k_base + offs_d[:, None] * stride_kd + n_idx[None, :] * stride_ks
        k = tl.load(k_ptrs, mask=d_mask[:, None] & n_mask[None, :], other=0.0)

        # Load V: [N, D]
        v_ptrs = v_base + n_idx[:, None] * stride_vs + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)

        # Compute scores: Q @ K^T -> [M, N]
        qk = tl.dot(q, k)
        qk = qk * qk_scale

        # --- Online softmax update ---
        row_max = tl.max(qk, axis=1)
        new_m = tl.maximum(m_i, row_max)
        alpha = tl.exp(m_i - new_m)
        p = tl.exp(qk - new_m[:, None])

        # Update accumulator: acc = alpha*acc + p @ V
        # p must be bf16 for tensor core; v is bf16
        acc = alpha[:, None] * acc + tl.dot(p.to(tl.bfloat16), v)
        l_i = alpha * l_i + tl.sum(p, axis=1)
        m_i = new_m

    # Final normalization
    inv_sum = tl.where(l_i > 0.0, tl.reciprocal(l_i), 0.0)
    acc = acc * inv_sum[:, None]

    # Write O: [M, D] in bf16
    o_ptrs = o_base + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(tl.bfloat16), mask=m_mask[:, None] & d_mask[None, :])

    # Write LSE: log-sum-exp per query token in fp32
    lse_val = tl.where(l_i > 0.0, m_i + tl.log(l_i), tl.full([], float('-inf'), dtype=tl.float32))
    lse_ptrs = lse_base + offs_m * stride_lses
    tl.store(lse_ptrs, lse_val, mask=m_mask)


def run(Q, K, V, O, LSE):
    """
    Compute non-causal multi-head attention forward pass.

    O = softmax(Q @ K^T / sqrt(D)) @ V      [bf16, (B,H,S,D)]
    LSE = logsumexp(Q @ K^T / sqrt(D))       [fp32, (B,H,S)]

    Destination-passing: writes into preallocated O and LSE tensors.
    """
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_DMODEL = 128

    grid = (B, H, triton.cdiv(S, BLOCK_M))

    _mha_fwd_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        S, D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=4,
        num_stages=3,
    )