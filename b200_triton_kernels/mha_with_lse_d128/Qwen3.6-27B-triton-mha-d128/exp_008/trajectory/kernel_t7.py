import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Aggressive staging for Hopper
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64},
                      num_warps=8, num_stages=5),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64},
                      num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64},
                      num_warps=8, num_stages=5),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64},
                      num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64},
                      num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128},
                      num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32},
                      num_warps=8, num_stages=5),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128},
                      num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32},
                      num_warps=4, num_stages=5),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128},
                      num_warps=8, num_stages=3),
    ],
    key=["n_ctx"],
    reset_to_zero=["out_ptr", "lse_ptr"],
)
@triton.jit
def _mha_kernel_persistent(
    q_ptr, k_ptr, v_ptr,
    out_ptr, lse_ptr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lb, stride_lh, stride_ls,
    n_ctx,
    num_heads,
    scale,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Persistent-group MHA kernel with tiled-inner-loops."""
    # Map each SM to a set of (batch, head, m_tile) tiles via a group id
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)

    # Total number of (hz, m) tile pairs
    hz_count = num_heads  # batch_h is flattened; num_programs is just 1 dim
    num_m_tiles = tl.cdiv(n_ctx, BLOCK_M)
    total_tiles = num_heads * num_m_tiles

    start_pid = pid
    stride_loop = max(num_pids, 1)

    for tile_id in range(start_pid, total_tiles, stride_loop):
        off_hz = tile_id // num_m_tiles
        pid_m = tile_id % num_m_tiles

        batch_idx = off_hz // num_heads
        head_idx = off_hz % num_heads

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        row_valid = offs_m < n_ctx

        # ---- Load Q [BLOCK_M, HEAD_DIM] ----
        q_ptrs = (q_ptr
                  + batch_idx * stride_qb + head_idx * stride_qh
                  + offs_m[:, None] * stride_qs
                  + tl.arange(0, HEAD_DIM)[None, :] * stride_qd)
        q_load_mask = row_valid[:, None]
        Q_tile = tl.load(q_ptrs, mask=q_load_mask, other=0.0)

        # ---- Init accumulators ----
        m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

        num_n_tiles = tl.cdiv(n_ctx, BLOCK_N)

        for start_n in range(num_n_tiles):
            offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
            col_valid = offs_n < n_ctx

            # Load K [BLOCK_N, HEAD_DIM]
            k_ptrs = (k_ptr
                      + batch_idx * stride_kb + head_idx * stride_kh
                      + offs_n[:, None] * stride_ks
                      + tl.arange(0, HEAD_DIM)[None, :] * stride_kd)
            K_tile = tl.load(k_ptrs, mask=(col_valid[:, None]), other=0.0)

            # Scores [BLOCK_M, BLOCK_N]
            scores = tl.dot(Q_tile, K_tile.T) * scale
            scores = tl.where(col_valid[None, :], scores, float("-inf"))

            # Online softmax
            m_i_prev = m_i
            m_i = tl.maximum(m_i, tl.max(scores, axis=1))
            alpha = tl.exp(m_i_prev - m_i)
            p = tl.exp(scores - m_i[:, None])

            l_i = alpha * l_i + tl.sum(p, axis=1)
            acc = acc * alpha[:, None]

            # Load V [BLOCK_N, HEAD_DIM]
            v_ptrs = (v_ptr
                      + batch_idx * stride_vb + head_idx * stride_vh
                      + offs_n[:, None] * stride_vs
                      + tl.arange(0, HEAD_DIM)[None, :] * stride_vd)
            V_tile = tl.load(v_ptrs, mask=(col_valid[:, None]), other=0.0)

            acc = acc + tl.dot(p.to(tl.bfloat16), V_tile)

        # ---- Epilogue ----
        inv_l = tl.where(l_i > 0.0, 1.0 / l_i, 1.0)
        acc = acc * inv_l[:, None]

        out_ptrs = (out_ptr
                    + batch_idx * stride_ob + head_idx * stride_oh
                    + offs_m[:, None] * stride_os
                    + tl.arange(0, HEAD_DIM)[None, :] * stride_od)
        tl.store(out_ptrs, acc.to(tl.bfloat16), mask=q_load_mask)

        lse_val = m_i + tl.log(l_i)
        lse_ptrs = (lse_ptr
                    + batch_idx * stride_lb + head_idx * stride_lh
                    + offs_m * stride_ls)
        tl.store(lse_ptrs, lse_val, mask=row_valid)


def run(Q, K, V, O, LSE):
    """Multi-head attention forward: O = softmax(Q K^T / sqrt(D)) V."""
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    scale = 1.0 / (D ** 0.5)

    num_sms = triton.runtime.driver.active.utils.get_device_properties(
        torch.cuda.current_device()
    ).MULTIPROCESSOR_COUNT if hasattr(triton.runtime.driver, 'active') else 132
    # Fallback query via torch
    try:
        import subprocess
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            stderr=subprocess.DEVNULL).decode().strip()
        if 'Hopper' in out or 'H100' in out or 'H200' in out or 'H800' in out:
            pass  # keep default 132
    except Exception:
        pass

    total_tiles = B * H * triton.cdiv(S, 128)  # estimated with typical BLOCK_M
    grid_size = min(num_sms, max(1, total_tiles))
    grid = (grid_size,)

    _mha_kernel_persistent[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        S, H, scale,
        HEAD_DIM=D,
    )