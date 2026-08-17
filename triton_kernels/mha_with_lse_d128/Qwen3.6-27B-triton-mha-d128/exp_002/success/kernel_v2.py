import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _mha_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr,
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
    pid = tl.program_id(0)

    num_bh = B * H
    num_pid_m = tl.cdiv(S, BLOCK_M)
    num_tiles = num_bh * num_pid_m

    for tile_idx in range(pid, num_tiles, NUM_SM):
        bh = tile_idx // num_pid_m
        pid_m = tile_idx % num_pid_m
        b = bh // H
        h = bh % H

        off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        off_d = tl.arange(0, BLOCK_D)
        off_n_base = tl.arange(0, BLOCK_N)

        # --- Load Q once [BLOCK_M, BLOCK_D] ---
        q_ptrs = (q_ptr +
                  b * stride_qb + h * stride_qh +
                  off_m[:, None] * stride_qs + off_d[None, :] * stride_qd)
        q_mask = (off_m[:, None] < S) & (off_d[None, :] < D)
        Q_tile = tl.load(q_ptrs, mask=q_mask, other=0.0,
                         eviction_policy="evict_last")

        # --- Online softmax accumulators ---
        acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
        m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
        l_i = tl.full((BLOCK_M,), 1.0, dtype=tl.float32)

        # --- K/V loop ---
        n_kvsteps = tl.cdiv(S, BLOCK_N)
        for sn in range(n_kvsteps):
            cur_n = sn * BLOCK_N + off_n_base

            # Load K [BLOCK_N, BLOCK_D]
            k_ptrs = (k_ptr +
                      b * stride_kb + h * stride_kh +
                      cur_n[:, None] * stride_ks + off_d[None, :] * stride_kd)
            k_mask = (cur_n[:, None] < S) & (off_d[None, :] < D)
            K_tile = tl.load(k_ptrs, mask=k_mask, other=0.0,
                             eviction_policy="evict_first")

            # Scores [BLOCK_M, BLOCK_N]
            scores = tl.dot(Q_tile, K_tile.T) * SCALE

            # Mask
            seq_mask = (off_m[:, None] < S) & (cur_n[None, :] < S)
            scores = tl.where(seq_mask, scores, float("-inf"))

            # Online softmax step
            m_new = tl.maximum(m_i, tl.max(scores, axis=1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new[:, None])

            acc = acc * alpha[:, None]

            # Load V [BLOCK_N, BLOCK_D]
            v_ptrs = (v_ptr +
                      b * stride_vb + h * stride_vh +
                      cur_n[:, None] * stride_vs + off_d[None, :] * stride_vd)
            v_mask = (cur_n[:, None] < S) & (off_d[None, :] < D)
            V_tile = tl.load(v_ptrs, mask=v_mask, other=0.0,
                             eviction_policy="evict_first")

            acc += tl.dot(p, V_tile.to(tl.float32))

            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_new

        # --- Output ---
        acc = acc / l_i[:, None]

        o_ptrs = (o_ptr +
                  b * stride_ob + h * stride_oh +
                  off_m[:, None] * stride_os + off_d[None, :] * stride_od)
        tl.store(o_ptrs, acc.to(tl.bfloat16), mask=q_mask)

        lse_ptrs = lse_ptr + b * stride_lb + h * stride_lh + off_m * stride_ls
        tl.store(lse_ptrs, m_i + tl.log(l_i), mask=(off_m < S))


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)

    B, H, S, D = Q.shape
    scale = 1.0 / float(D ** 0.5)
    sm = torch.cuda.get_device_properties(Q.device).multi_processor_count

    BLOCK_M = 64
    BLOCK_N = 32
    BLOCK_D = 128

    grid_size = min(sm, B * H * triton.cdiv(S, BLOCK_M))
    grid = (grid_size,)

    _mha_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        B, H, S, D,
        scale,
        NUM_SM=sm,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=3,
    )