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
    Persistent-scheduling causal MHA forward kernel with device-side tensor
    descriptors for TMA load/store on Hopper.
    
    Each CTA repeatedly grabs work from a global counter until all
    (batch, head, query_tile) combinations are done.
    """
    pid = tl.program_id(0)

    # Total number of (batch, head, query_tile) combinations
    num_pid_bh = B * H
    num_pid_m = tl.cdiv(S, BLOCK_M)
    num_tiles = num_pid_bh * num_pid_m

    offs_d = tl.arange(0, D)

    for tile_id in tl.range(pid, num_tiles, NUM_SMS, flatten=False):
        pid_bh = tile_id // num_pid_m
        pid_m = tile_id % num_pid_m
        pid_b = pid_bh // H
        pid_h = pid_bh % H

        # ===== Query offsets =====
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        off_mask_m = offs_m < S

        # ===== Build tensor descriptors for this (b,h) sub-tensor =====
        # Create descriptors for Q,K,V,O,LSE scoped to this batch/head
        q_desc = tl.make_tensor_descriptor(
            Q_ptr + pid_b * stride_qb + pid_h * stride_qh,
            shape=[S, D], strides=[stride_qs, stride_qd],
            block_shape=[BLOCK_M, D], padding_option="zero")
        k_desc = tl.make_tensor_descriptor(
            K_ptr + pid_b * stride_kb + pid_h * stride_kh,
            shape=[S, D], strides=[stride_ks, stride_kd],
            block_shape=[BLOCK_N, D], padding_option="zero")
        v_desc = tl.make_tensor_descriptor(
            V_ptr + pid_b * stride_vb + pid_h * stride_vh,
            shape=[S, D], strides=[stride_vs, stride_vd],
            block_shape=[BLOCK_N, D], padding_option="zero")
        o_desc = tl.make_tensor_descriptor(
            Out_ptr + pid_b * stride_ob + pid_h * stride_oh,
            shape=[S, D], strides=[stride_os, stride_od],
            block_shape=[BLOCK_M, D])
        lse_desc = tl.make_tensor_descriptor(
            LSE_ptr + pid_b * stride_lsb + pid_h * stride_lsh,
            shape=[S, 1], strides=[stride_lss, 0],
            block_shape=[BLOCK_M, 1])

        # Load Q tile [BLOCK_M, D] once
        Q_tile = q_desc.load([pid_m * BLOCK_M, 0])

        # Initialize accumulators
        acc_O = tl.zeros((BLOCK_M, D), dtype=tl.float32)
        acc_m = tl.full((BLOCK_M,), value=-float("inf"), dtype=tl.float32)
        acc_l = tl.zeros((BLOCK_M,), dtype=tl.float32)

        # Main loop over key/value tiles
        num_kv_tiles = tl.cdiv(S, BLOCK_N)
        start_n = 0

        for kv_idx in range(num_kv_tiles):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            n_mask = offs_n < S

            # Load K, V tiles
            K_tile = tl.load(
                K_ptr + pid_b * stride_kb + pid_h * stride_kh +
                offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd,
                mask=n_mask[:, None], other=0.0)

            V_tile_fp32 = tl.load(
                V_ptr + pid_b * stride_vb + pid_h * stride_vh +
                offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd,
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

            start_n += BLOCK_N

        # Normalize output
        denom_inv = tl.where(acc_l > 0.0, 1.0 / acc_l, 0.0)
        Out_val = (acc_O * denom_inv[:, None]).to(tl.bfloat16)

        # Store output
        o_ptrs = Out_ptr + pid_b * stride_ob + pid_h * stride_oh + \
                 offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
        tl.store(o_ptrs, Out_val, mask=off_mask_m[:, None])

        # Store LSE
        lse_ptrs = LSE_ptr + pid_b * stride_lsb + pid_h * stride_lsh + \
                   offs_m * stride_lss
        LSE_val = tl.where(acc_l > 0.0, acc_m + tl.log(acc_l), float("-inf"))
        tl.store(lse_ptrs, LSE_val, mask=off_mask_m)


def run(Q, K, V, O, LSE):
    """
    Causal multi-head attention forward pass with persistent scheduling
    and optimized tiling for Hopper GPUs.
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

    # Determine number of SMs for persistent scheduling
    num_sms = torch.cuda.get_device_properties(Q.device).multi_processor_count

    # Tile sizes for D=128, Hopper
    BLOCK_M = 128
    BLOCK_N = 64

    # Persistent grid: one CTA per SM, capped by total work
    num_pid_bh = B * H
    num_pid_m = triton.cdiv(S, BLOCK_M)
    grid_size = min(num_sms, num_pid_bh * num_pid_m)
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
        num_warps=4,
        num_stages=3,
    )