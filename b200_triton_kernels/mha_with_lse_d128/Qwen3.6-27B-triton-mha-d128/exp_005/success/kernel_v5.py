import torch
import triton
import triton.language as tl


def _alloc_fn(size: int, alignment: int, stream):
    """Allocator for device-created tensor descriptors (infrastructure only)."""
    return torch.empty(size, device="cuda", dtype=torch.int8)


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
    """FlashAttention forward via online softmax with device tensor descriptors.

    Descriptors enable TMA async DMA on Hopper, overlapping data movement
    with WGMMA tensor-core computation.
    """
    pid_m = tl.program_id(0)
    pid_zh = tl.program_id(1)

    off_b = pid_zh // num_heads
    off_h = pid_zh % num_heads

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_SIZE)

    q_valid = offs_m < seq_len

    # Byte-offset base for (batch, head) slice
    q_base = Q + off_b * stride_qb + off_h * stride_qh
    k_base = K + off_b * stride_kb + off_h * stride_kh
    v_base = V + off_b * stride_vb + off_h * stride_vh
    o_base = Out + off_b * stride_ob + off_h * stride_oh
    lse_base = Lse + off_b * stride_lb + off_h * stride_lh

    # Build device tensor descriptors — TMA-enabled on Hopper
    q_desc = tl.make_tensor_descriptor(q_base, [seq_len, HEAD_SIZE],
                                        [stride_qs, stride_qd],
                                        [BLOCK_M, HEAD_SIZE], padding_option="zero")
    k_desc = tl.make_tensor_descriptor(k_base, [seq_len, HEAD_SIZE],
                                        [stride_ks, stride_kd],
                                        [BLOCK_N, HEAD_SIZE], padding_option="zero")
    v_desc = tl.make_tensor_descriptor(v_base, [seq_len, HEAD_SIZE],
                                        [stride_vs, stride_vd],
                                        [BLOCK_N, HEAD_SIZE], padding_option="zero")
    o_desc = tl.make_tensor_descriptor(o_base, [seq_len, HEAD_SIZE],
                                        [stride_os, stride_od],
                                        [BLOCK_M, HEAD_SIZE])
    lse_desc = tl.make_tensor_descriptor(lse_base, [seq_len], [stride_ls], [BLOCK_M])

    # Load Q once
    Q_tile = q_desc.load([pid_m * BLOCK_M, 0])
    dtype_in = Q_tile.dtype

    # Online-softmax accumulators (FP32)
    m_i = tl.full([BLOCK_M], float('-inf'), tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, tl.float32)
    acc_o = tl.zeros([BLOCK_M, HEAD_SIZE], tl.float32)

    num_steps = tl.cdiv(seq_len, BLOCK_N)

    for step in range(num_steps):
        K_tile = k_desc.load([step * BLOCK_N, 0])
        V_tile = v_desc.load([step * BLOCK_N, 0])

        scores = tl.dot(Q_tile, K_tile.T) * scale

        # Pad-mask for incomplete last-KV block
        col_off = step * BLOCK_N + offs_n
        n_valid = col_off < seq_len
        scores = tl.where(n_valid[None, :], scores, float('-inf'))

        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        p = tl.exp(scores - m_ij[:, None])
        alpha = tl.exp(m_i - m_ij)
        acc_o = acc_o * alpha[:, None] + tl.dot(p.to(dtype_in), V_tile)

        l_ij = tl.sum(p, axis=1)
        l_i = l_i * alpha + l_ij
        m_i = m_ij

    o_final = acc_o / l_i[:, None]
    o_desc.store([pid_m * BLOCK_M, 0], o_final.to(dtype_in))
    lse_desc.store([pid_m * BLOCK_M], m_i + tl.log(l_i))


def run(Q, K, V, O, LSE):
    """Destination-passing entry point."""
    torch.cuda.set_device(Q.device)

    # Must install allocator BEFORE first kernel launch (descriptor infrastructure)
    triton.set_allocator(_alloc_fn)

    B, H, S, D = Q.shape
    scale = 1.0 / float(D ** 0.5)

    BLOCK_M = 64
    BLOCK_N = 64
    num_warps = 4
    num_stages = 3

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