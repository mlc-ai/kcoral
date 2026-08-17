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
    B, H, S, D,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
):
    """
    Multi-head attention forward kernel using flash attention online softmax.

    Grid axes: (batch, head, m_block)
    Each program processes BLOCK_M query positions for one (batch, head) pair,
    iterating over all key positions in BLOCK_N chunks.
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)

    # Indexing tensors
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)       # [BLOCK_M]
    offs_n = tl.arange(0, BLOCK_N)                           # [BLOCK_N]
    offs_d = tl.arange(0, BLOCK_DMODEL)                      # [BLOCK_DMODEL]

    # Base offsets for this (batch, head)
    q_offset = pid_b * stride_qb + pid_h * stride_qh
    k_offset = pid_b * stride_kb + pid_h * stride_kh
    v_offset = pid_b * stride_vb + pid_h * stride_vh
    o_offset = pid_b * stride_ob + pid_h * stride_oh
    lse_offset = pid_b * stride_lseb + pid_h * stride_lseh

    # Load Q tile: [BLOCK_M, BLOCK_DMODEL]
    q_ptrs = (Q + q_offset + offs_m[:, None] * stride_qs
              + offs_d[None, :] * stride_qd)
    q = tl.load(q_ptrs,
                mask=(offs_m[:, None] < S) & (offs_d[None, :] < D),
                other=0.0)

    # Scaling factor for attention logits
    qk_scale = 1.0 / tl.sqrt(D)

    # Flash attention accumulators (fp32 for numerical stability)
    m_i = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)  # running max
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)               # running sum
    acc = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)  # weighted V sum

    # Iterate over key/value blocks
    for start_n in range(0, S, BLOCK_N):
        n_idx = start_n + offs_n
        n_mask = n_idx < S

        # Load K tile: [BLOCK_DMODEL, BLOCK_N]
        k_ptrs = (K + k_offset + offs_d[:, None] * stride_kd
                  + n_idx[None, :] * stride_ks)
        k = tl.load(k_ptrs,
                    mask=(offs_d[:, None] < D) & n_mask[None, :],
                    other=0.0)

        # Load V tile: [BLOCK_N, BLOCK_DMODEL]
        v_ptrs = (V + v_offset + n_idx[:, None] * stride_vs
                  + offs_d[None, :] * stride_vd)
        v = tl.load(v_ptrs,
                    mask=n_mask[:, None] & (offs_d[None, :] < D),
                    other=0.0)

        # Compute attention scores Q @ K^T: [BLOCK_M, BLOCK_N]
        qk = tl.dot(q, k)                   # fp32 output by default
        qk = qk * qk_scale

        # Online softmax update
        max_qk = tl.max(qk, axis=1)         # [BLOCK_M]
        new_m = tl.maximum(m_i, max_qk)     # new running max
        alpha = tl.exp(m_i - new_m)         # old normalization factor
        p = tl.exp(qk - new_m[:, None])     # stabilized probs [BLOCK_M, BLOCK_N]

        # Update accumulated output: acc = alpha * acc + p @ V
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)

        # Update running sum
        l_i = alpha * l_i + tl.sum(p, axis=1)
        m_i = new_m

    # Final normalization
    inv_l = tl.where(l_i > 0, 1.0 / l_i, 0.0)
    acc = acc * inv_l[:, None]

    # Store output O: [BLOCK_M, BLOCK_DMODEL] -> bf16
    o_ptrs = (O + o_offset + offs_m[:, None] * stride_os
              + offs_d[None, :] * stride_od)
    tl.store(o_ptrs, acc.to(tl.bfloat16),
             mask=(offs_m[:, None] < S) & (offs_d[None, :] < D))

    # Store LSE: log-sum-exp in fp32
    lse = tl.where(l_i > 0, m_i + tl.log(l_i), float('-inf'))
    lse_ptrs = LSE + lse_offset + offs_m * stride_lses
    tl.store(lse_ptrs, lse, mask=offs_m < S)


def run(Q, K, V, O, LSE):
    """
    Compute multi-head attention:
      O = softmax(Q @ K^T / sqrt(D)) @ V      [bf16, (B,H,S,D)]
      LSE = logsumexp(Q @ K^T / sqrt(D))       [fp32, (B,H,S)]

    Destination-passing: writes into preallocated O and LSE tensors.
    """
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_DMODEL = 128  # must be >= D (head dimension)

    grid = (B, H, triton.cdiv(S, BLOCK_M))

    _mha_fwd_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        B, H, S, D,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=4, num_stages=3,
    )