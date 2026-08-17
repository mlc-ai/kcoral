import torch
import triton
import triton.language as tl


@triton.jit
def _mha_fwd_kernel(
    Q, K, V,
    Out, Lse,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lb, stride_lh, stride_ls,
    num_heads,
    seq_len,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_zh = tl.program_id(1)

    off_b = off_zh // num_heads
    off_h = off_zh % num_heads

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_SIZE)

    q_valid_m = offs_m < seq_len

    # Base pointers for this (batch, head)
    q_base = Q + off_b * stride_qb + off_h * stride_qh
    k_base = K + off_b * stride_kb + off_h * stride_kh
    v_base = V + off_b * stride_vb + off_h * stride_vh
    o_base = Out + off_b * stride_ob + off_h * stride_oh
    lse_base = Lse + off_b * stride_lb + off_h * stride_lh

    # Load Q tile once [BLOCK_M, HEAD_SIZE], bf16
    Q_tile = tl.load(q_base + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd,
                     mask=q_valid_m[:, None], other=0.0)
    dtype_in = Q_tile.dtype

    # Online-softmax state — all FP32
    m_i = tl.full([BLOCK_M], value=float('-inf'), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], value=1.0, dtype=tl.float32)
    acc_o = tl.zeros([BLOCK_M, HEAD_SIZE], dtype=tl.float32)

    num_steps = tl.cdiv(seq_len, BLOCK_N)

    for step in range(num_steps):
        n_idx = step * BLOCK_N + offs_n
        n_valid = n_idx < seq_len

        # Load K and V tiles [BLOCK_N, HEAD_SIZE]
        K_tile = tl.load(k_base + n_idx[:, None] * stride_ks + offs_d[None, :] * stride_kd,
                         mask=n_valid[:, None], other=0.0)
        V_tile = tl.load(v_base + n_idx[:, None] * stride_vs + offs_d[None, :] * stride_vd,
                         mask=n_valid[:, None], other=0.0)

        # Scores [BLOCK_M, BLOCK_N], FP32 accumulate via tensor cores
        scores = tl.dot(Q_tile, K_tile.T) * scale

        # Pad-masked KV columns → -inf
        scores = tl.where(n_valid[None, :], scores, float('-inf'))

        # --- Online softmax (FP32) ---
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        p = tl.exp(scores - m_ij[:, None])
        alpha = tl.exp(m_i - m_ij)

        acc_o = acc_o * alpha[:, None] + tl.dot(p.to(dtype_in), V_tile)

        l_ij = tl.sum(p, axis=1)
        l_i = l_i * alpha + l_ij
        m_i = m_ij

    # Epilogue
    o_final = acc_o / l_i[:, None]

    out_ptrs = o_base + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(out_ptrs, o_final.to(dtype_in), mask=q_valid_m[:, None])

    lse_ptrs = lse_base + offs_m * stride_ls
    tl.store(lse_ptrs, m_i + tl.log(l_i), mask=q_valid_m)


def run(Q, K, V, O, LSE):
    """Destination-passing entry point."""
    torch.cuda.set_device(Q.device)

    B, H, S, D = Q.shape
    scale = 1.0 / float(D ** 0.5)

    BLOCK_M = 64
    BLOCK_N = 64
    num_warps = 4
    num_stages = 3

    # Query-major grid — empirically faster for these shapes
    grid = (triton.cdiv(S, BLOCK_M), B * H)

    _mha_fwd_kernel[grid](
        Q, K, V,
        O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        H,
        S,
        scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        HEAD_SIZE=D,
        num_warps=num_warps,
        num_stages=num_stages,
    )