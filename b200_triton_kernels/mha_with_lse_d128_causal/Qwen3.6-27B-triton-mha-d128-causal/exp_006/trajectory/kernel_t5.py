import torch
import triton
import triton.language as tl


def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


triton.set_allocator(alloc_fn)


@triton.jit
def _mha_causal_fwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    Out_ptr,
    LSE_ptr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lsb, stride_lsh, stride_lss,
    B, H, S,
    scale,
    NUM_SMS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
):
    """
    Persistent-scheduling causal MHA forward kernel.
    Uses pointer arithmetic (not descriptors) to keep things correct
    while still using a work-stealing grid pattern.
    """
    pid = tl.program_id(0)

    # Total number of (batch, head, query_tile) combinations
    num_pid_bh = B * H
    num_pid_m = tl.cdiv(S, BLOCK_M)
    num_tiles = num_pid_bh * num_pid_m

    offs_d = tl.arange(0, D)
    offs_n_local = tl.arange(0, BLOCK_N)

    for tile_id in tl.range(pid, num_tiles, NUM_SMS, flatten=False):
        pid_bh = tile_id // num_pid_m
        pid_m = tile_id % num_pid_m
        pid_b = pid_bh // H
        pid_h = pid_bh % H

        # ===== Query offsets =====
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        off_mask_m = offs_m < S

        # Base pointers for this batch/head combination
        q_base = Q_ptr + pid_b * stride_qb + pid_h * stride_qh
        k_base = K_ptr + pid_b * stride_kb + pid_h * stride_kh
        v_base = V_ptr + pid_b * stride_vb + pid_h * stride_vh
        o_base = Out_ptr + pid_b * stride_ob + pid_h * stride_oh
        lse_base = LSE_ptr + pid_b * stride_lsb + pid_h * stride_lsh

        # ===== Load Q tile once: shape [BLOCK_M, D] =====
        q_ptrs = q_base + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
        Q_tile = tl.load(q_ptrs, mask=off_mask_m[:, None], other=0.0)

        # Initialize accumulators
        acc_O = tl.zeros((BLOCK_M, D), dtype=tl.float32)
        acc_m = tl.full((BLOCK_M,), value=-float("inf"), dtype=tl.float32)
        acc_l = tl.zeros((BLOCK_M,), dtype=tl.float32)

        # Main loop over key/value tiles
        num_kv_tiles = tl.cdiv(S, BLOCK_N)

        for kv_idx in range(num_kv_tiles):
            offs_n = kv_idx * BLOCK_N + offs_n_local
            n_mask = offs_n < S

            # Load K tile [BLOCK_N, D]
            K_tile = tl.load(
                k_base + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd,
                mask=n_mask[:, None], other=0.0)

            # Load V tile [BLOCK_N, D] and cast to fp32
            V_tile_fp32 = tl.load(
                v_base + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd,
                mask=n_mask[:, None], other=0.0).to(tl.float32)

            # Compute attention scores [BLOCK_M, BLOCK_N]
            scores = tl.dot(Q_tile, K_tile.T, input_precision="tf32") * scale

            # Apply causal mask: query_pos >= key_pos
            causal_mask = offs_m[:, None] >= offs_n[None, :]
            full_mask = causal_mask & n_mask[None, :] & off_mask_m[:, None]
            scores = tl.where(full_mask, scores, float("-inf"))

            # Online softmax update
            cur_m = tl.max(scores, axis=1)
            new_m = tl.maximum(acc_m, cur_m)
            alpha = tl.exp(acc_m - new_m)
            p = tl.exp(scores - new_m[:, None])

            acc_O = alpha[:, None] * acc_O + tl.dot(p, V_tile_fp32)

            p_sum = tl.sum(p, axis=1)
            acc_l = alpha * acc_l + p_sum
            acc_m = new_m

        # Normalize output
        denom_inv = tl.where(acc_l > 0.0, 1.0 / acc_l, 0.0)
        Out_val = (acc_O * denom_inv[:, None]).to(tl.bfloat16)

        # Store output [BLOCK_M, D]
        o_ptrs = o_base + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
        tl.store(o_ptrs, Out_val, mask=off_mask_m[:, None])

        # Store LSE [BLOCK_M]
        lse_ptrs = lse_base + offs_m * stride_lss
        LSE_val = tl.where(acc_l > 0.0, acc_m + tl.log(acc_l), float("-inf"))
        tl.store(lse_ptrs, LSE_val, mask=off_mask_m)


def run(Q, K, V, O, LSE):
    """
    Causal multi-head attention forward pass with persistent scheduling.
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

    # Number of SMs for persistent scheduling
    num_sms = torch.cuda.get_device_properties(Q.device).multi_processor_count

    # Tile sizes for D=128, Hopper — larger BLOCK_M for better WGMMA usage
    BLOCK_M = 128
    BLOCK_N = 64

    num_pid_bh = B * H
    num_pid_m = triton.cdiv(S, BLOCK_M)
    total_tiles = num_pid_bh * num_pid_m
    grid_size = min(num_sms, total_tiles)
    grid = (grid_size,)

    _mha_causal_fwd_kernel[grid](
        Q, K, V, O, LSE,
        stride_qb, stride_qh, stride_qs, stride_qd,
        stride_kb, stride_kh, stride_ks, stride_kd,
        stride_vb, stride_vh, stride_vs, stride_vd,
        stride_ob, stride_oh, stride_os, stride_od,
        stride_lsb, stride_lsh, stride_lss,
        B, H, S, scale,
        NUM_SMS=grid_size,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        D=D,
        num_warps=8,
        num_stages=3,
    )