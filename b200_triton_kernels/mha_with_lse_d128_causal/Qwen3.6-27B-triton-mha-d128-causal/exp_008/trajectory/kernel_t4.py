import torch
import triton
import triton.language as tl


@triton.jit
def _flash_attn_fwd_kernel(
    Q,
    K,
    V,
    O,
    LSE,
    seqlen,
    scale,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lse_b, stride_lse_h, stride_lse_s,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Causal multi-head attention with online softmax and FP32 accumulation."""

    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)

    # Index arrays
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)

    # Base offsets for this (b, h) pair
    base_q = Q + pid_b * stride_qb + pid_h * stride_qh
    base_k = K + pid_b * stride_kb + pid_h * stride_kh
    base_v = V + pid_b * stride_vb + pid_h * stride_vh
    base_o = O + pid_b * stride_ob + pid_h * stride_oh
    base_lse = LSE + pid_b * stride_lse_b + pid_h * stride_lse_h

    # --- Load Q tile once: [BLOCK_M, D] in bf16, cast to fp32 ---
    q_ptrs = base_q + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    q_mask = (offs_m[:, None] < seqlen)
    q_tile = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.float32)

    # --- Online softmax accumulators (FP32) ---
    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    # Number of key/value blocks
    num_blocks_n = tl.cdiv(seqlen, BLOCK_N)

    for start_n in range(num_blocks_n):
        offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = (offs_n[:, None] < seqlen)

        # Load K tile: [BLOCK_N, D] in bf16, cast to fp32
        k_ptrs = base_k + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        k_tile = tl.load(k_ptrs, mask=n_mask, other=0.0).to(tl.float32)

        # Attention scores: [BLOCK_M, D] @ [D, BLOCK_N] = [BLOCK_M, BLOCK_N]
        s = tl.dot(q_tile, tl.trans(k_tile)) * scale

        # Causal mask: only allow attending to past/self
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        s = tl.where(causal_mask, s, -float("inf"))

        # Update running max per row
        m_i_old = m_i
        m_i_new = tl.maximum(m_i, tl.max(s, axis=1))

        # Compute normalized attention weights
        p = tl.exp(s - m_i_new[:, None])

        # Update the denominator using the new max
        alpha = tl.exp(m_i_old - m_i_new)
        l_i = alpha * l_i + tl.sum(p, axis=1)

        # Load V tile: [BLOCK_N, D] in bf16, cast to fp32
        v_ptrs = base_v + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
        v_tile = tl.load(v_ptrs, mask=n_mask, other=0.0).to(tl.float32)

        # Accumulate weighted V: rescale old accumulator, add new contribution
        acc = alpha[:, None] * acc + tl.dot(p, v_tile)

        # Advance m_i
        m_i = m_i_new

    # Store output: convert fp32 accumulator to bf16
    o_ptrs = base_o + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    o_mask = (offs_m[:, None] < seqlen)
    tl.store(o_ptrs, acc.to(tl.bfloat16), mask=o_mask)

    # Store LSE: log-sum-exp of attention scores (fp32)
    lse_ptrs = base_lse + offs_m * stride_lse_s
    lse_mask = (offs_m < seqlen)
    final_lse = m_i + tl.log(l_i)
    tl.store(lse_ptrs, final_lse, mask=lse_mask)


def run(Q, K, V, O, LSE):
    """Causal multi-head attention forward pass with LSE output.

    Computes:
      O = softmax(Q @ K^T / sqrt(D), causal=True) @ V
      LSE = logsumexp(Q @ K^T / sqrt(D), causal=True, axis=-1)

    All inputs/outputs follow destination-passing convention.
    """
    torch.cuda.set_device(Q.device)

    bsz, num_heads, seqlen, head_dim = Q.shape

    # Precompute attention scale on host
    scale = 1.0 / (head_dim ** 0.5)

    # Extract element-wise strides
    sq = Q.stride()
    sk = K.stride()
    sv = V.stride()
    so = O.stride()
    sl = LSE.stride()

    # Tile dimensions
    BLOCK_M = 64
    BLOCK_N = 64

    # Grid: (query_tiles, heads, batches)
    num_tiles_m = triton.cdiv(seqlen, BLOCK_M)
    grid = (num_tiles_m, num_heads, bsz)

    _flash_attn_fwd_kernel[grid](
        Q, K, V, O, LSE,
        seqlen,
        scale,
        stride_qb=sq[0], stride_qh=sq[1], stride_qs=sq[2], stride_qd=sq[3],
        stride_kb=sk[0], stride_kh=sk[1], stride_ks=sk[2], stride_kd=sk[3],
        stride_vb=sv[0], stride_vh=sv[1], stride_vs=sv[2], stride_vd=sv[3],
        stride_ob=so[0], stride_oh=so[1], stride_os=so[2], stride_od=so[3],
        stride_lse_b=sl[0], stride_lse_h=sl[1], stride_lse_s=sl[2],
        D=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=3,
    )