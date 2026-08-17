import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


triton.set_allocator(alloc_fn)


@triton.jit
def _mha_causal_fwd_kernel(
    Q_desc, K_desc, V_desc, O_desc,
    LSE_ptr,
    stride_lsb, stride_lsh, stride_lss,
    B, H, S,
    scale,
    NUM_SMS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
):
    """
    Persistent-scheduling causal MHA forward kernel with TMA descriptors.
    """
    pid = tl.program_id(0)

    num_pid_bh = B * H
    num_pid_m = tl.cdiv(S, BLOCK_M)
    num_tiles = num_pid_bh * num_pid_m

    offs_d = tl.arange(0, D)

    for tile_id in tl.range(pid, num_tiles, NUM_SMS, flatten=False):
        pid_bh = tile_id // num_pid_m
        pid_m = tile_id % num_pid_m
        pid_b = pid_bh // H
        pid_h = pid_bh % H

        # Query offsets and boundary
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        m_mask = offs_m < S
        start_m = pid_m * BLOCK_M

        # Load Q tile [BLOCK_M, D] via TMA
        Q_tile = Q_desc.load([pid_m * BLOCK_M, 0])

        # Initialize accumulators
        acc_O = tl.zeros((BLOCK_M, D), dtype=tl.float32)
        acc_m = tl.full((BLOCK_M,), value=-float("inf"), dtype=tl.float32)
        acc_l = tl.zeros((BLOCK_M,), dtype=tl.float32)

        # Main loop over key/value tiles
        num_kv_tiles = tl.cdiv(S, BLOCK_N)

        for kv_idx in range(num_kv_tiles):
            start_n = kv_idx * BLOCK_N

            # Compute causal boundary: how many keys in this tile are valid
            # Causal: key_pos <= query_pos. For query at start_m, keys up to start_m are visible.
            # Within a key tile starting at start_n with width BLOCK_N:
            #   valid_key_end = min(start_n + BLOCK_N, S)
            #   For each query row r (0..BLOCK_M-1): query_pos = start_m + r
            #   Keys visible in this tile: key_pos in [start_n, min(start_n+BLOCK_N, S)] AND key_pos <= query_pos
            
            # Boundary: if entire key tile is after all queries in this Q tile, skip
            # This happens when start_n > start_m + BLOCK_M - 1, i.e., start_n >= start_m + BLOCK_M
            tl.device_assert(False)  # placeholder removed below

            # Load K and V tiles via TMA
            K_tile = K_desc.load([start_n, 0])
            V_tile_fp32 = V_desc.load([start_n, 0]).to(tl.float32)

            # Compute attention scores [BLOCK_M, BLOCK_N]
            scores = tl.dot(Q_tile, K_tile.T, input_precision="tf32") * scale

            # Build causal mask efficiently
            # For query offset r in [0, BLOCK_M), key offset c in [0, BLOCK_N):
            #   causal iff (start_m + r) >= (start_n + c)
            #   i.e. c <= (start_m - start_n + r)
            # We compute per-row valid key count
            offs_n_local = tl.arange(0, BLOCK_N)
            # n_boundary handles out-of-sequence positions
            n_mask_global = (start_n + offs_n_local) < S
            
            # Causal mask: each query row has different boundary within key tile
            # causal_mask[r, c] = (start_m + r) >= (start_n + c)
            #                  = c - r <= start_m - start_n
            #                  = c <= (start_m - start_n + r)
            causal_boundary = start_m - start_n  # scalar shift
            # For each query row r: keys with local index c where c <= causal_boundary + r
            col_idx = offs_n_local[None, :]  # [1, BLOCK_N]
            row_idx = offs_m[:, None]        # [BLOCK_M, 1]
            causal_mask = (start_n + col_idx) <= (row_idx)
            
            full_mask = causal_mask & n_mask_global[None, :] & m_mask[:, None]
            scores = tl.where(full_mask, scores, float("-inf"))

            # Online softmax
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

        # Store output via descriptor
        O_desc.store([start_m, 0], Out_val)

        # Store LSE via pointers
        lse_base = LSE_ptr + pid_b * stride_lsb + pid_h * stride_lsh
        lse_ptrs = lse_base + offs_m * stride_lss
        LSE_val = tl.where(acc_l > 0.0, acc_m + tl.log(acc_l), float("-inf"))
        tl.store(lse_ptrs, LSE_val, mask=m_mask)


def run(Q, K, V, O, LSE):
    """
    Causal multi-head attention forward pass with TMA descriptors and persistent scheduling.
    """
    torch.cuda.set_device(Q.device)

    B, H, S, D = Q.shape
    assert K.shape == (B, H, S, D)
    assert V.shape == (B, H, S, D)
    assert O.shape == (B, H, S, D)
    assert LSE.shape == (B, H, S)

    stride_lsb, stride_lsh, stride_lss = LSE.stride()
    scale = 1.0 / (float(D) ** 0.5)

    num_sms = torch.cuda.get_device_properties(Q.device).multi_processor_count

    # Tile sizes
    BLOCK_M = 128
    BLOCK_N = 64

    # Create host-side tensor descriptors for Q, K, V, O
    # Descriptors operate on [S, D] sub-tensors; we'll dispatch per (b,h) inside kernel
    # Since descriptors must know absolute addresses, we create one per needed block shape
    
    # Actually: TensorDescriptors describe contiguous physical tensors. 
    # For per-(b,h) slicing, we need to either create descriptors per (b,h) or handle offsetting.
    # Let's create descriptors for the full tensors and pass base offsets.
    
    # Simpler approach: create flattened views and adjust. 
    # Better: just create descriptors matching the full [B,H,S,D] layout.
    
    # For simplicity, reshape to [BH, S, D] and create descriptors over that
    Q_reshaped = Q.reshape(B * H, S, D)
    K_reshaped = K.reshape(B * H, S, D)  
    V_reshaped = V.reshape(B * H, S, D)
    O_reshaped = O.reshape(B * H, S, D)
    
    q_desc = TensorDescriptor.from_tensor(Q_reshaped.contiguous(), [BLOCK_M, D])
    k_desc = TensorDescriptor.from_tensor(K_reshaped.contiguous(), [BLOCK_N, D])
    v_desc = TensorDescriptor.from_tensor(V_reshaped.contiguous(), [BLOCK_N, D])
    o_desc = TensorDescriptor.from_tensor(O_reshaped.contiguous(), [BLOCK_M, D])
    
    num_pid_bh = B * H
    num_pid_m = triton.cdiv(S, BLOCK_M)
    total_tiles = num_pid_bh * num_pid_m
    grid_size = min(num_sms, total_tiles)
    grid = (grid_size,)

    _mha_causal_fwd_kernel[grid](
        q_desc, k_desc, v_desc, o_desc,
        LSE,
        stride_lsb, stride_lsh, stride_lss,
        B, H, S, scale,
        NUM_SMS=grid_size,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        D=D,
        num_warps=8,
        num_stages=3,
    )