import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=8, num_stages=4),
    ],
    key=["S", "D"],
)
@triton.jit
def _mha_forward_persistent(
    Q, K, V, O, LSE,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lb, stride_lh, stride_ls,
    B, H, S, D,
    SCALE,
    NUM_SM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Persistent-scheduling Flash-Attention forward kernel with LSE."""

    pid = tl.program_id(0)

    # Total number of (batch, head, query-row-tile) work items
    num_bh = B * H
    num_pid_m = tl.cdiv(S, BLOCK_M)
    num_tiles = num_bh * num_pid_m

    start_n_tiles = tl.arange(0, BLOCK_D)   # not used; just satisfy BLOCK_D constexpr

    # Each program iterates over tiles in stride NUM_SM
    for tile_idx in range(pid, num_tiles, NUM_SM):
        bh = tile_idx // num_pid_m
        pid_m = tile_idx % num_pid_m
        b = bh // H
        h = bh % H

        off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        off_d = tl.arange(0, BLOCK_D)

        # --- Load Q tile [BLOCK_M, BLOCK_D] ---
        q_offs = (b * stride_qb + h * stride_qh
                   + off_m[:, None] * stride_qs
                   + off_d[None, :] * stride_qd)
        q_mask = (off_m[:, None] < S) & (off_d[None, :] < D)
        Q_tile = tl.load(Q + q_offs, mask=q_mask, other=0.0)

        # --- Online-softmax accumulators (fp32) ---
        acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
        m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
        l_i = tl.full((BLOCK_M,), 1.0, dtype=tl.float32)

        # --- Iterate over K/V tiles ---
        num_k_steps = tl.cdiv(S, BLOCK_N)
        for start_n in range(0, num_k_steps):
            cur_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)

            # Load K tile [BLOCK_N, BLOCK_D]
            k_offs = (b * stride_kb + h * stride_kh
                       + cur_n[:, None] * stride_ks
                       + off_d[None, :] * stride_kd)
            k_mask = (cur_n[:, None] < S) & (off_d[None, :] < D)
            K_tile = tl.load(K + k_offs, mask=k_mask, other=0.0)

            # Attention scores [BLOCK_M, BLOCK_N]
            p = tl.dot(Q_tile, K_tile.T) * SCALE

            # Mask invalid positions
            mask_p = (off_m[:, None] < S) & (cur_n[None, :] < S)
            p = tl.where(mask_p, p, float("-inf"))

            # Online softmax
            m_i_new = tl.maximum(m_i, tl.max(p, axis=1))
            alpha = tl.exp(m_i - m_i_new)
            beta = tl.exp(p - m_i_new[:, None])

            acc = acc * alpha[:, None]

            # Load V tile [BLOCK_N, BLOCK_D]
            v_offs = (b * stride_vb + h * stride_vh
                       + cur_n[:, None] * stride_vs
                       + off_d[None, :] * stride_vd)
            v_mask = (cur_n[:, None] < S) & (off_d[None, :] < D)
            V_tile = tl.load(V + v_offs, mask=v_mask, other=0.0)

            acc = acc + tl.dot(beta.to(tl.float32), V_tile.to(tl.float32))

            l_i = l_i * alpha + tl.sum(beta, axis=1)
            m_i = m_i_new

        # --- Write back ---
        acc = acc / l_i[:, None]

        o_offs = (b * stride_ob + h * stride_oh
                  + off_m[:, None] * stride_os
                  + off_d[None, :] * stride_od)
        tl.store(O + o_offs, acc.to(tl.bfloat16), mask=q_mask)

        lse_offs = b * stride_lb + h * stride_lh + off_m * stride_ls
        lse = m_i + tl.log(l_i)
        tl.store(LSE + lse_offs, lse, mask=(off_m < S))


def run(Q, K, V, O, LSE):
    """Non-causal MHA forward: O = softmax(Q@K^T / sqrt(D)) @ V, plus LSE."""
    torch.cuda.set_device(Q.device)

    B, H, S, D = Q.shape
    scale = 1.0 / float(D ** 0.5)

    # Cap persistent grid at actual SM count for this device
    sm_count = torch.cuda.get_device_properties(Q.device).multi_processor_count
    BLOCK_D = 128

    grid = (min(sm_count, B * H * triton.cdiv(S, 64)),)

    _mha_forward_persistent[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        B, H, S, D,
        scale,
        NUM_SM=sm_count,
        BLOCK_D=BLOCK_D,
    )