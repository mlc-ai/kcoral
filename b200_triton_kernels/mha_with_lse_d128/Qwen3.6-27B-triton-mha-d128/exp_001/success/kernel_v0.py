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
    S, D, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
):
    """Multi-head attention forward using flash attention online softmax."""
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)

    # Index vectors
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    m_mask = offs_m < S
    d_mask = offs_d < D

    # Base pointers for this (batch, head)
    q_base = Q + pid_b * stride_qb + pid_h * stride_qh
    k_base = K + pid_b * stride_kb + pid_h * stride_kh
    v_base = V + pid_b * stride_vb + pid_h * stride_vh
    o_base = O + pid_b * stride_ob + pid_h * stride_oh
    lse_base = LSE + pid_b * stride_lseb + pid_h * stride_lseh

    # Load Q tile: [BLOCK_M, BLOCK_DMODEL]
    q_ptrs = q_base + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)

    # Flash attention state (fp32)
    m_i = tl.full([BLOCK_M], -1e20, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # Iterate over KV blocks
    for start_n in range(0, S, BLOCK_N):
        n_idx = start_n + offs_n
        n_mask = n_idx < S

        # Load K transposed: physical [N,D], load as [D, N]
        k_ptrs = k_base + offs_d[:, None] * stride_kd + n_idx[None, :] * stride_ks
        k = tl.load(k_ptrs, mask=d_mask[:, None] & n_mask[None, :], other=0.0)

        # Load V: [BLOCK_N, BLOCK_DMODEL]
        v_ptrs = v_base + n_idx[:, None] * stride_vs + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)

        # Attention scores: [BLOCK_M, BLOCK_N]
        qk = tl.dot(q, k)
        qk = qk * scale

        # Online softmax update
        max_k = tl.max(qk, axis=1)
        new_m = tl.maximum(m_i, max_k)
        alpha = tl.exp(m_i - new_m)
        p = tl.exp(qk - new_m[:, None])

        # Update accumulators
        acc = alpha[:, None] * acc + tl.dot(p.to(tl.bfloat16), v)
        l_i = alpha * l_i + tl.sum(p, axis=1)
        m_i = new_m

    # Normalize output
    mask_safe = l_i > 0.0
    inv_l = tl.where(mask_safe, 1.0 / l_i, 0.0)
    acc = acc * inv_l[:, None]

    # Store O [bf16]
    o_ptrs = o_base + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(tl.bfloat16), mask=m_mask[:, None] & d_mask[None, :])

    # Store LSE [fp32]
    lse_val = tl.where(mask_safe, m_i + tl.log(l_i), -1e20)
    lse_ptrs = lse_base + offs_m * stride_lses
    tl.store(lse_ptrs, lse_val, mask=m_mask)


def run(Q, K, V, O, LSE):
    """
    Multi-head attention forward pass.
    O = softmax(Q @ K^T / sqrt(D)) @ V      [bf16, (B,H,S,D)]
    LSE = logsumexp(Q @ K^T / sqrt(D))       [fp32, (B,H,S)]
    """
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_DMODEL = 128

    # Pre-compute scale factor in fp32 on host
    scale = 1.0 / float(D) ** 0.5

    grid = (B, H, triton.cdiv(S, BLOCK_M))

    _mha_fwd_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        S, D, scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=4,
        num_stages=3,
    )