import torch
import triton
import triton.language as tl


def run(Q, K, V, O, LSE):
    """Causal multi-head attention forward returning O and LSE."""
    torch.cuda.set_device(Q.device)

    B, H, S, D = Q.shape
    assert D == 128

    _impl(Q, K, V, O, LSE, B, H, S, D)


def _impl(Q, K, V, O, LSE, B, H, S, D):
    import triton.runtime.autotuner as autotuner

    configs = [
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_D": 128}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_D": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_D": 128}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_D": 128}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_D": 128}, num_warps=8, num_stages=3),
    ]

    @triton.autotune(
        configs=configs,
        key=["S", "D"],
    )
    @triton.jit
    def _kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        o_ptr,
        lse_ptr,
        stride_s,
        stride_d,
        stride_bh,
        S,
        D,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        bh_id = tl.program_id(0)
        pid_m = tl.program_id(1)

        q_base = q_ptr + bh_id * stride_bh
        k_base = k_ptr + bh_id * stride_bh
        v_base = v_ptr + bh_id * stride_bh
        o_base = o_ptr + bh_id * stride_bh
        lse_base = lse_ptr + bh_id * stride_bh

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S
        offs_d = tl.arange(0, BLOCK_D)

        q_ptrs = q_base + offs_m[:, None] * stride_s + offs_d[None, :] * stride_d
        q_tile = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

        m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

        scale = 1.0 / tl.sqrt(tl.cast(D, tl.float32))

        num_k_steps = tl.cdiv(S, BLOCK_N)
        for step in tl.range(0, num_k_steps, num_stages=3):
            start_n = step * BLOCK_N
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < S

            k_ptrs = k_base + offs_n[:, None] * stride_s + offs_d[None, :] * stride_d
            k_tile = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)

            v_ptrs = v_base + offs_n[:, None] * stride_s + offs_d[None, :] * stride_d
            v_tile = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

            scores = tl.dot(q_tile, k_tile.T, out_dtype=tl.float32) * scale

            causal = offs_m[:, None] >= offs_n[None, :]
            valid = causal & mask_m[:, None] & mask_n[None, :]
            masked_scores = tl.where(valid, scores, -float("inf"))

            new_m = tl.maximum(m_i, tl.max(masked_scores, axis=1))
            alpha = tl.exp(m_i - new_m)
            p = tl.exp(masked_scores - new_m[:, None])

            new_l = alpha * l_i + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v_tile, out_dtype=tl.float32)

            m_i = new_m
            l_i = new_l

        denom = tl.where(l_i > 0.0, l_i, 1.0)
        acc = acc / denom[:, None]

        o_ptrs = o_base + offs_m[:, None] * stride_s + offs_d[None, :] * stride_d
        tl.store(o_ptrs, acc.to(tl.bfloat16), mask=mask_m[:, None])

        lse_val = m_i + tl.log(denom)
        lse_ptrs = lse_base + offs_m
        tl.store(lse_ptrs, lse_val, mask=mask_m)

    BH = B * H
    num_pid_m = triton.cdiv(S, BLOCK_M) if 'BLOCK_M' in dir() else triton.cdiv(S, 128)
    grid = (BH, triton.cdiv(S, 128))

    stride_s = D
    stride_d = 1
    stride_bh = S * D

    _kernel[grid](
        Q, K, V, O, LSE,
        stride_s, stride_d, stride_bh,
        S, D,
        BLOCK_M=128,
        BLOCK_N=64,
        BLOCK_D=128,
    )