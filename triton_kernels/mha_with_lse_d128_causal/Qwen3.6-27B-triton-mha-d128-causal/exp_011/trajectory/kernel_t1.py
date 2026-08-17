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
    stride_lsb, stride_lsh, stride_lss,
    S,
    D,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)

    # Row indices for query positions
    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # Column indices covering full head dimension
    off_d = tl.arange(0, D)

    # Base pointers for this batch and head
    q_offs = pid_b * stride_qb + pid_h * stride_qh
    k_offs = pid_b * stride_kb + pid_h * stride_kh
    v_offs = pid_b * stride_vb + pid_h * stride_vh
    o_offs = pid_b * stride_ob + pid_h * stride_oh
    lse_offs = pid_b * stride_lsb + pid_h * stride_lsh

    q_base = Q + q_offs
    k_base = K + k_offs
    v_base = V + v_offs
    o_base = O + o_offs
    lse_base = LSE + lse_offs

    # Mask for valid query rows within sequence length
    q_mask = off_m[:, None] < S

    # Load Q tile: [BLOCK_M, D]
    q_ptrs = q_base + off_m[:, None] * stride_qs + off_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # Initialize online softmax accumulators
    acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), fill_value=-float('inf'), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # Number of key/value block tiles
    num_n_tiles = tl.cdiv(S, BLOCK_N)

    for start_n in range(num_n_tiles):
        off_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)

        # Load K tile: [BLOCK_N, D]
        k_ptrs = k_base + off_n[:, None] * stride_ks + off_d[None, :] * stride_kd
        k_mask = off_n[:, None] < S
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        # Compute attention scores: [BLOCK_M, D] x [D, BLOCK_N] -> [BLOCK_M, BLOCK_N]
        scores = tl.dot(q, tl.trans(k)) * scale

        # Apply causal mask: query_pos >= key_pos
        causal_mask = off_m[:, None] >= off_n[None, :]
        scores = tl.where(causal_mask, scores, float('-inf'))

        # Online softmax: update running max
        m_new = tl.maximum(m_i, tl.max(scores, axis=1))

        # Rescale old accumulator by exp(m_old - m_new)
        alpha = tl.exp(m_i - m_new)
        acc = acc * alpha[:, None]

        # Compute stabilized probabilities
        p = tl.exp(scores - m_new[:, None])

        # Load V tile: [BLOCK_N, D]
        v_ptrs = v_base + off_n[:, None] * stride_vs + off_d[None, :] * stride_vd
        v_mask = off_n[:, None] < S
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)

        # Accumulate output: p [@][BLOCK_M, BLOCK_N] V [@][BLOCK_N, D] -> acc [BLOCK_M, D]
        acc = tl.dot(p, v, acc=acc)

        # Update LSE running sum
        l_i = l_i * alpha + tl.sum(p, axis=1)

        m_i = m_new

    # Final normalization: O = acc / l_i
    lse_vals = m_i + tl.log(l_i)
    acc = acc / l_i[:, None]

    # Store output O: [BLOCK_M, D] -> bf16
    o_ptrs = o_base + off_m[:, None] * stride_os + off_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(dtype=O.dtype.element_ty), mask=q_mask)

    # Store LSE: [BLOCK_M] -> fp32
    lse_ptrs = lse_base + off_m * stride_lss
    tl.store(lse_ptrs, lse_vals.to(dtype=LSE.dtype.element_ty), mask=off_m < S)


def run(Q, K, V, O, LSE):
    """Causal multi-head attention forward pass returning O and LSE.

    Args:
        Q:  [B, H, S, D] bf16 query tensor
        K:  [B, H, S, D] bf16 key tensor
        V:  [B, H, S, D] bf16 value tensor
        O:  [B, H, S, D] bf16 preallocated output tensor
        LSE:[B, H, S]   fp32 preallocated LSE tensor
    """
    torch.cuda.set_device(Q.device)

    B, H, S, D = Q.shape
    scale = 1.0 / float(D) ** 0.5

    BLOCK_M = 64
    BLOCK_N = 64

    grid = (B, H, triton.cdiv(S, BLOCK_M))

    _mha_fwd_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        S,
        D,
        scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=3,
    )