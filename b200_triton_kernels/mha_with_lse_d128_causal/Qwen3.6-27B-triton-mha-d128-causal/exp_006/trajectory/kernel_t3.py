import torch
import triton
import triton.language as tl


def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


triton.set_allocator(alloc_fn)


@triton.jit
def _mha_causal_fwd_kernel(
    Q,
    K,
    V,
    Out,
    LSE,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lsb, stride_lsh, stride_lss,
    B, H, S,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    pid_b = pid_bh // H
    pid_h = pid_bh % H

    # Query offsets (absolute sequence positions)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_mask_m = offs_m < S

    # Head dimension offsets
    offs_d = tl.arange(0, D)

    # Base pointers for this batch/head combination
    q_base = Q + pid_b * stride_qb + pid_h * stride_qh
    k_base = K + pid_b * stride_kb + pid_h * stride_kh
    v_base = V + pid_b * stride_vb + pid_h * stride_vh
    o_base = Out + pid_b * stride_ob + pid_h * stride_oh
    lse_base = LSE + pid_b * stride_lsb + pid_h * stride_lsh

    # ===== Load Q tile once: shape [BLOCK_M, D] =====
    q_ptrs = q_base + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    Q_tile = tl.load(q_ptrs, mask=off_mask_m[:, None], other=0.0)

    # ===== Initialize accumulators =====
    acc_O = tl.zeros((BLOCK_M, D), dtype=tl.float32)
    acc_m = tl.full((BLOCK_M,), value=-float("inf"), dtype=tl.float32)
    acc_l = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # ===== Main loop over key/value tiles =====
    start_n = 0
    stop_n = tl.cdiv(S, BLOCK_N) * BLOCK_N

    while start_n < stop_n:
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < S

        # === Load K tile [BLOCK_N, D] ===
        K_tile = tl.load(
            k_base + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd,
            mask=n_mask[:, None], other=0.0)

        # === Load V tile [BLOCK_N, D] ===
        V_tile_fp32 = tl.load(
            v_base + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd,
            mask=n_mask[:, None], other=0.0).to(tl.float32)

        # === Compute attention scores [BLOCK_M, BLOCK_N] ===
        scores = tl.dot(Q_tile, K_tile.T, input_precision="tf32") * scale

        # === Apply causal mask ===
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        full_mask = causal_mask & n_mask[None, :] & off_mask_m[:, None]
        scores = tl.where(full_mask, scores, float("-inf"))

        # === Online softmax ===
        cur_m = tl.max(scores, axis=1)
        new_m = tl.maximum(acc_m, cur_m)
        alpha = tl.exp(acc_m - new_m)
        p = tl.exp(scores - new_m[:, None])

        # Update output accumulator: acc_O = alpha * acc_O + p @ V_tile
        acc_O = alpha[:, None] * acc_O + tl.dot(p, V_tile_fp32)

        # Update running log-sum-exp
        p_sum = tl.sum(p, axis=1)
        acc_l = alpha * acc_l + p_sum
        acc_m = new_m

        start_n += BLOCK_N

    # Handle last partial key tile (if S not divisible by BLOCK_N)
    tail_start = tl.cdiv(S, BLOCK_N) * BLOCK_N
    if tail_start < S:
        offs_n_tail = tail_start + tl.arange(0, BLOCK_N)
        n_mask_tail = offs_n_tail < S

        K_tile_tail = tl.load(
            k_base + offs_n_tail[:, None] * stride_ks + offs_d[None, :] * stride_kd,
            mask=n_mask_tail[:, None], other=0.0)

        V_tile_tail_fp32 = tl.load(
            v_base + offs_n_tail[:, None] * stride_vs + offs_d[None, :] * stride_vd,
            mask=n_mask_tail[:, None], other=0.0).to(tl.float32)

        scores_tail = tl.dot(Q_tile, K_tile_tail.T, input_precision="tf32") * scale
        causal_mask_tail = offs_m[:, None] >= offs_n_tail[None, :]
        full_mask_tail = causal_mask_tail & n_mask_tail[None, :] & off_mask_m[:, None]
        scores_tail = tl.where(full_mask_tail, scores_tail, float("-inf"))

        cur_m_tail = tl.max(scores_tail, axis=1)
        new_m_tail = tl.maximum(acc_m, cur_m_tail)
        alpha_tail = tl.exp(acc_m - new_m_tail)
        p_tail = tl.exp(scores_tail - new_m_tail[:, None])

        acc_O = alpha_tail[:, None] * acc_O + tl.dot(p_tail, V_tile_tail_fp32)
        p_sum_tail = tl.sum(p_tail, axis=1)
        acc_l = alpha_tail * acc_l + p_sum_tail
        acc_m = new_m_tail

    # ===== Normalize output and compute LSE =====
    denom_inv = tl.where(acc_l > 0.0, 1.0 / acc_l, 0.0)
    Out_val = acc_O * denom_inv[:, None]

    # Store output [BLOCK_M, D]
    o_ptrs = o_base + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(o_ptrs, Out_val.to(tl.bfloat16), mask=off_mask_m[:, None])

    # Store LSE
    lse_ptrs = lse_base + offs_m * stride_lss
    LSE_val = tl.where(acc_l > 0.0, acc_m + tl.log(acc_l), float("-inf"))
    tl.store(lse_ptrs, LSE_val, mask=off_mask_m)


def run(Q, K, V, O, LSE):
    """
    Causal multi-head attention forward pass with optimized tiling.
    """
    torch.cuda.set_device(Q.device)

    B, H, S, D = Q.shape
    assert K.shape == (B, H, S, D)
    assert V.shape == (B, H, S, D)
    assert O.shape == (B, H, S, D)
    assert LSE.shape == (B, H, S)

    stride_qb, stride_qh, stride_qs, stride_qd = Q.stride()
    stride_kb, stride_kh, stride_ks, stride_kd = K.stride()
    stride_vb, stride_vh, stride_vs, stride_vd = V.stride()
    stride_ob, stride_oh, stride_os, stride_od = O.stride()
    stride_lsb, stride_lsh, stride_lss = LSE.stride()

    scale = 1.0 / (float(D) ** 0.5)

    # Tile sizes optimized for D=128, Hopper
    BLOCK_M = 128
    BLOCK_N = 64

    num_pid_bh = B * H
    num_pid_m = triton.cdiv(S, BLOCK_M)
    grid = (num_pid_bh, num_pid_m)

    _mha_causal_fwd_kernel[grid](
        Q, K, V, O, LSE,
        stride_qb, stride_qh, stride_qs, stride_qd,
        stride_kb, stride_kh, stride_ks, stride_kd,
        stride_vb, stride_vh, stride_vs, stride_vd,
        stride_ob, stride_oh, stride_os, stride_od,
        stride_lsb, stride_lsh, stride_lss,
        B, H, S, scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        D=D,
        num_warps=4,
        num_stages=3,
    )