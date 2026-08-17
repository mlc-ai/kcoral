import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _mha_causal_tma(
    Q_desc, K_desc, V_desc, O_desc,
    LSE,
    S,
    stride_lsb, stride_lsh, stride_ls,
    softmax_scale,
    B: tl.constexpr,
    H: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    n_sms = tl.num_programs(0)

    num_blocks_m = tl.cdiv(S, BLOCK_M)
    total_tiles = B * H * num_blocks_m

    for tile_idx in range(pid, total_tiles, n_sms):
        bh = tile_idx // num_blocks_m
        bid_m = tile_idx % num_blocks_m
        bid_b = bh // H
        bid_h = bh % H

        offset_m = bid_m * BLOCK_M

        # Load Q once per query block
        q_fp32 = Q_desc.load([bid_b, bid_h, offset_m, 0]).to(tl.float32)

        # Online softmax accumulators
        m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
        d_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc_o = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

        num_kv = tl.cdiv(S, BLOCK_N)

        for start_n in range(num_kv):
            offset_n = start_n * BLOCK_N

            k_fp32 = K_desc.load([bid_b, bid_h, offset_n, 0]).to(tl.float32)
            v_fp32 = V_desc.load([bid_b, bid_h, offset_n, 0]).to(tl.float32)

            scores = tl.dot(q_fp32, k_fp32.T) * softmax_scale

            row_pos = offset_m + tl.arange(0, BLOCK_M)
            col_pos = offset_n + tl.arange(0, BLOCK_N)
            causal = row_pos[:, None] >= col_pos[None, :]
            q_in = row_pos[:, None] < S
            k_in = col_pos[None, :] < S
            valid = causal & q_in & k_in
            scores = tl.where(valid, scores, float("-inf"))

            m_ij = tl.max(scores, axis=1, keep_dims=False)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new[:, None])

            acc_o = acc_o * alpha[:, None]
            acc_o = tl.dot(p, v_fp32, acc_o)

            p_sum = tl.sum(p, axis=1, keep_dims=False)
            d_i = alpha * d_i + p_sum
            m_i = m_new

        acc_o = acc_o / d_i[:, None]
        O_desc.store([bid_b, bid_h, offset_m, 0], acc_o.to(tl.bfloat16))

        lse_val = m_i + tl.log(d_i)
        off_lse = offset_m + tl.arange(0, BLOCK_M)
        lse_ptr = LSE + bid_b * stride_lsb + bid_h * stride_lsh + off_lse * stride_ls
        tl.store(lse_ptr, lse_val, mask=(off_lse < S))


def run(Q, K, V, O, LSE):
    """Compute causal MHA forward (O, LSE) into pre-allocated output tensors."""
    torch.cuda.set_device(Q.device)

    B, H, S, D = Q.shape
    softmax_scale = 1.0 / float(D ** 0.5)

    BLOCK_M = 64
    BLOCK_N = 64

    q_desc = TensorDescriptor.from_tensor(Q, [B, H, BLOCK_M, D])
    k_desc = TensorDescriptor.from_tensor(K, [B, H, BLOCK_N, D])
    v_desc = TensorDescriptor.from_tensor(V, [B, H, BLOCK_N, D])
    o_desc = TensorDescriptor.from_tensor(O, [B, H, BLOCK_M, D])

    num_sm = 132
    num_blocks_m = triton.cdiv(S, BLOCK_M)
    total_tiles = B * H * num_blocks_m
    grid_size = min(num_sm, total_tiles)

    _mha_causal_tma[(grid_size,)](
        q_desc, k_desc, v_desc, o_desc,
        LSE,
        S=S,
        stride_lsb=LSE.stride(0), stride_lsh=LSE.stride(1), stride_ls=LSE.stride(2),
        softmax_scale=softmax_scale,
        B=B, H=H,
        HEAD_DIM=D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=4,
    )