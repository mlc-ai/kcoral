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

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)

    # Base offsets for this (b, h) pair
    base_q = Q + pid_b * stride_qb + pid_h * stride_qh
    base_k = K + pid_b * stride_kb + pid_h * stride_kh
    base_v = V + pid_b * stride_vb + pid_h * stride_vh
    base_o = O + pid_b * stride_ob + pid_h * stride_oh
    base_lse = LSE + pid_b * stride_lse_b + pid_h * stride_lse_h

    # --- Load Q tile once, keep as bf16 for tensor-core dot ---
    q_ptrs = base_q + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    q_mask = (offs_m[:, None] < seqlen)
    q_tile = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # Online softmax accumulators (FP32)
    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    num_blocks_n = tl.cdiv(seqlen, BLOCK_N)

    for start_n in range(num_blocks_n):
        offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask_row = offs_n < seqlen
        n_mask = n_mask_row[:, None]

        # Load K tile, keep as bf16 for tensor-core dot
        k_ptrs = base_k + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        k_tile = tl.load(k_ptrs, mask=n_mask, other=0.0)

        # bf16 dot -> fp32 accumulator output
        s = tl.dot(q_tile, tl.trans(k_tile)) * scale

        # Causal mask
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        s = tl.where(causal_mask, s, -float("inf"))

        # Update running max
        m_i_old = m_i
        m_i_new = tl.maximum(m_i, tl.max(s, axis=1))

        # Normalized probabilities in fp32
        p = tl.exp(s - m_i_new[:, None])

        alpha = tl.exp(m_i_old - m_i_new)
        l_i = alpha * l_i + tl.sum(p, axis=1)

        # Load V tile and convert to fp32 for second dot (p is fp32)
        v_ptrs = base_v + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
        v_tile = tl.load(v_ptrs, mask=n_mask, other=0.0).to(tl.float32)

        # Accumulate weighted contribution
        acc = alpha[:, None] * acc + tl.dot(p, v_tile)

        m_i = m_i_new

    # Normalize output
    acc = acc / l_i[:, None]

    # Store output bf16
    o_ptrs = base_o + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(tl.bfloat16), mask=q_mask)

    # Store LSE fp32
    lse_ptrs = base_lse + offs_m * stride_lse_s
    tl.store(lse_ptrs, m_i + tl.log(l_i), mask=offs_m < seqlen)


def run(Q, K, V, O, LSE):
    """Causal multi-head attention forward pass with LSE output."""
    torch.cuda.set_device(Q.device)

    bsz, num_heads, seqlen, head_dim = Q.shape
    scale = 1.0 / (head_dim ** 0.5)

    sq = Q.stride()
    sk = K.stride()
    sv = V.stride()
    so = O.stride()
    sl = LSE.stride()

    # Try several tile configs and pick the best via manual tuning
    # For D=128, S=4096: larger BLOCK_M gives fewer blocks to iterate
    BLOCK_M = 128
    BLOCK_N = 64

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
        num_warps=8,
        num_stages=4,
    )