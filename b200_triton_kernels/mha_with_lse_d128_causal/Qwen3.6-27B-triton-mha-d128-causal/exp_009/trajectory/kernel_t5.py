import torch
import triton
import triton.language as tl


@triton.jit
def _mha_forward_kernel(
    Q, K, V, O, LSE,
    stride_qbh, stride_qs, stride_qd,
    stride_kbh, stride_ks, stride_kd,
    stride_vbh, stride_vs, stride_vd,
    stride_obh, stride_os, stride_od,
    stride_lse_bh, stride_lse_s,
    seq_len,
    num_heads_total,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    N_SM: tl.constexpr,
):
    """Persistent-grid causal multi-head attention forward kernel."""
    pid = tl.program_id(0)

    num_pid_m = tl.cdiv(seq_len, BLOCK_M)
    num_bh = num_heads_total
    num_tiles = num_pid_m * num_bh

    # Process tiles in stride-N_SM increments for balanced scheduling
    for tile_idx in range(pid, num_tiles, N_SM):
        bh_idx = tile_idx // num_pid_m
        m_idx = tile_idx % num_pid_m

        offs_m = m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, D)

        q_mask = offs_m < seq_len

        # Base offsets for this (batch*head) slice
        q_off = Q + bh_idx * stride_qbh
        k_off = K + bh_idx * stride_kbh
        v_off = V + bh_idx * stride_vbh
        o_off = O + bh_idx * stride_obh
        lse_off = LSE + bh_idx * stride_lse_bh

        # Load Q tile once [BLOCK_M, D]
        q_ptrs = q_off + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
        q = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)

        scale = 1.0 / tl.sqrt(tl.cast(D, tl.float32))

        # Online-softmax state (FP32)
        m_i = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)

        q_pos = m_idx * BLOCK_M + tl.arange(0, BLOCK_M)

        num_steps = tl.cdiv(seq_len, BLOCK_N)

        for start_n in range(num_steps):
            offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
            kv_pos = start_n * BLOCK_N + tl.arange(0, BLOCK_N)

            k_mask = offs_n < seq_len

            # Load K tile [BLOCK_N, D]
            k_ptrs = k_off + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
            k = tl.load(k_ptrs, mask=k_mask[:, None], other=0.0)

            # Load V tile [BLOCK_N, D]
            v_ptrs = v_off + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
            v = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0)

            # Attention scores [BLOCK_M, BLOCK_N] – fp32 via tensor cores
            scores = tl.dot(q, tl.trans(k)) * scale

            # Causal + bounds mask
            causal = q_pos[:, None] >= kv_pos[None, :]
            valid = causal & q_mask[:, None] & k_mask[None, :]

            scores = tl.where(valid, scores, float('-inf'))

            # --- Online softmax update (stable form) ---
            row_max = tl.max(scores, axis=1)
            m_i_new = tl.maximum(m_i, row_max)

            p = tl.where(valid, tl.exp(scores - m_i_new[:, None]), 0.0)

            was_valid = m_i > float('-inf')
            alpha = tl.where(was_valid, tl.exp(m_i - m_i_new), 0.0)

            l_i = tl.where(was_valid, alpha * l_i, 0.0) + tl.sum(p, axis=1)
            acc = tl.where(was_valid[:, None], alpha[:, None] * acc, 0.0) \
                  + tl.dot(p, v.to(tl.float32))
            m_i = m_i_new

        # --- Epilogue ---
        has_valid = l_i > 0.0
        o_val = tl.where(has_valid[:, None], acc / l_i[:, None], 0.0)
        lse_val = tl.where(has_valid, m_i + tl.log(l_i), float('-inf'))

        # Store O: fp32 -> bf16
        o_ptrs = o_off + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
        tl.store(o_ptrs, o_val.to(tl.bfloat16), mask=q_mask[:, None])

        # Store LSE: fp32
        lse_ptrs = lse_off + offs_m * stride_lse_s
        tl.store(lse_ptrs, lse_val, mask=q_mask)


def run(Q, K, V, O, LSE):
    """Causal multi-head attention forward with LSE output.

    Inputs:  Q [B, H, S, D] bf16, K [B, H, S, D] bf16, V [B, H, S, D] bf16
    Outputs: O [B, H, S, D] bf16, LSE [B, H, S] fp32
    """
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape

    BLOCK_M = 128
    BLOCK_N = 64
    N_SM = 132  # Hopper SXM-class target

    num_bh = B * H
    num_pid_m = triton.cdiv(S, BLOCK_M)
    num_tiles = num_pid_m * num_bh

    grid = (min(N_SM, num_tiles),)

    _mha_forward_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0) * H + Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0) * H + K.stride(1), K.stride(2), K.stride(3),
        V.stride(0) * H + V.stride(1), V.stride(2), V.stride(3),
        O.stride(0) * H + O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0) * H + LSE.stride(1), LSE.stride(2),
        S, num_bh,
        D=D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        N_SM=N_SM,
        num_warps=8,
        num_stages=3,
    )