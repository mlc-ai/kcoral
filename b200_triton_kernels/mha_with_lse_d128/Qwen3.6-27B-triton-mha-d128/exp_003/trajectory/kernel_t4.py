import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=4),
    ],
    key=["S"],
)
@triton.jit
def _mha_forward_kernel(
    Q_desc_ptr, K_desc_ptr, V_desc_ptr, O_desc_ptr, LSE_desc_ptr,
    stride_qs, stride_ks, stride_vs, stride_os, stride_lses,
    B, H, S,
    scale,
    NUM_PID_BH: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Forward MHA kernel using device-side tensor descriptors for TMA."""
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    bid_b = pid_bh // H
    bid_h = pid_h = pid_bh % H

    # ----- Create device tensor descriptors for this (b,h) slice -----
    q_base = Q_desc_ptr + bid_b * stride_qs * S + bid_h * stride_qs
    k_base = K_desc_ptr + bid_b * stride_ks * S + bid_h * stride_ks
    v_base = V_desc_ptr + bid_b * stride_vs * S + bid_h * stride_vs
    o_base = O_desc_ptr + bid_b * stride_os * S + bid_h * stride_os
    lse_base = LSE_desc_ptr + bid_b * stride_lses * S + bid_h * stride_lses

    D = 128

    Q_tile_size = [BLOCK_M, D]
    KV_tile_size = [BLOCK_N, D]
    O_tile_size = [BLOCK_M, D]

    Q_desc = tl.make_tensor_descriptor(q_base, shape=[S, D], strides=[stride_qs, 1],
                                        block_shape=Q_tile_size, padding_option="zero")
    K_desc = tl.make_tensor_descriptor(k_base, shape=[S, D], strides=[stride_ks, 1],
                                        block_shape=KV_tile_size, padding_option="zero")
    V_desc = tl.make_tensor_descriptor(v_base, shape=[S, D], strides=[stride_vs, 1],
                                        block_shape=KV_tile_size, padding_option="zero")
    O_desc = tl.make_tensor_descriptor(o_base, shape=[S, D], strides=[stride_os, 1],
                                        block_shape=O_tile_size, padding_option="zero")
    LSE_desc = tl.make_tensor_descriptor(lse_base, shape=[S], strides=[stride_lses],
                                          block_shape=[BLOCK_M], padding_option="zero")

    offs_m = pid_m * BLOCK_M
    offs_d = tl.arange(0, D)

    m_mask = offs_m + tl.arange(0, BLOCK_M) < S

    # Load Q once via descriptor
    Q_tile = Q_desc.load([offs_m, 0])                       # [BLOCK_M, D] bf16

    acc_o = tl.zeros((BLOCK_M, D), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.full((BLOCK_M,), 1.0, dtype=tl.float32)

    # Loop over K/V sequence tiles
    num_n_tiles = tl.cdiv(S, BLOCK_N)
    for tn in range(num_n_tiles):
        offset_n = tn * BLOCK_N

        K_tile = K_desc.load([offset_n, 0])                  # [BLOCK_N, D] bf16
        scores = tl.dot(Q_tile, K_tile.T) * scale             # [BLOCK_M, BLOCK_N] fp32

        n_mask = (offset_n + tl.arange(0, BLOCK_N)) < S

        # Apply masks
        scores = tl.where(m_mask[:, None], scores, float("-inf"))
        scores = tl.where(n_mask[None, :], scores, float("-inf"))

        m_ij = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_ij)

        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        l_new = alpha * l_i + tl.sum(p, axis=1)

        acc_o = acc_o * alpha[:, None]

        V_tile = V_desc.load([offset_n, 0])                  # [BLOCK_N, D] bf16
        acc_o = acc_o + tl.dot(p.to(tl.bfloat16), V_tile)

        m_i = m_new
        l_i = l_new

    acc_o = acc_o / l_i[:, None]
    O_desc.store([offs_m, 0], acc_o.to(tl.bfloat16))

    lse_vals = m_i + tl.log(l_i)
    LSE_desc.store([offs_m], lse_vals)


def run(Q, K, V, O, LSE):
    """Compute non-causal multi-head attention forward pass with LSE.

    O = softmax(Q @ K^T / sqrt(D)) @ V      (bf16, [B, H, S, D])
    LSE = logsumexp(Q @ K^T / sqrt(D))       (fp32, [B, H, S])
    """
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape

    scale = 1.0 / (D ** 0.5)

    # Allocate descriptor storage on device
    def alloc_fn(size: int, alignment: int, stream):
        return torch.empty(size, device="cuda", dtype=torch.int8)
    triton.set_allocator(alloc_fn)

    NUM_PID_BH = B * H

    grid = lambda META: (NUM_PID_BH, triton.cdiv(S, META["BLOCK_M"]))

    _mha_forward_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(2), K.stride(2), V.stride(2), O.stride(2), LSE.stride(2),
        B, H, S,
        scale,
        NUM_PID_BH=NUM_PID_BH,
    )