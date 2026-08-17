import math

import torch
import triton
import triton.language as tl


@triton.jit
def _mha_fwd_kernel(
    Q, K, V, O, LSE,
    H, S, D,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    """Optimized non-causal multi-head attention forward kernel."""
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    
    b_idx = pid_y // H
    h_idx = pid_y % H
    
    # Compute base pointers for the specific head
    q_ptr = Q + b_idx * H * S * D + h_idx * S * D
    k_ptr = K + b_idx * H * S * D + h_idx * S * D
    v_ptr = V + b_idx * H * S * D + h_idx * S * D
    o_ptr = O + b_idx * H * S * D + h_idx * S * D
    lse_ptr = LSE + b_idx * H * S + h_idx * S
    
    # Setup TMA descriptors to efficiently pull [S, D] matrices onto the chip
    q_desc = tl.make_tensor_descriptor(
        q_ptr, shape=[S, D], strides=[D, 1],
        block_shape=[BLOCK_Q, D], padding_option="zero")
    k_desc = tl.make_tensor_descriptor(
        k_ptr, shape=[S, D], strides=[D, 1],
        block_shape=[BLOCK_KV, D], padding_option="zero")
    v_desc = tl.make_tensor_descriptor(
        v_ptr, shape=[S, D], strides=[D, 1],
        block_shape=[BLOCK_KV, D], padding_option="zero")
    
    q_start = pid_x * BLOCK_Q
    col_D = tl.arange(0, D)
    
    # Load Q tile once; it will be re-used across all inner loop reductions
    q = q_desc.load([q_start, 0])
    
    o_acc = tl.zeros((BLOCK_Q, D), tl.float32)
    m = tl.full((BLOCK_Q, 1), 0.0, tl.float32)
    l = tl.full((BLOCK_Q, 1), 0.0, tl.float32)
    valid_flag = tl.full((BLOCK_Q, 1), 0.0, tl.float32)
    
    scale = 1.0 / tl.sqrt(D)
    num_kv_tiles = tl.cdiv(S, BLOCK_KV)
    
    for kv_idx in range(num_kv_tiles):
        kv_start = kv_idx * BLOCK_KV
        
        # Fetch K and V blocks from HBM asynchronously via TMA hardware units
        k = k_desc.load([kv_start, 0])
        v = v_desc.load([kv_start, 0])
        
        # 1. Compute Unnormalized Attention Scores (Q @ K^T)
        s_acc = tl.zeros((BLOCK_Q, BLOCK_KV), tl.float32)
        s = tl.dot(q, k.T, acc=s_acc) * scale
        
        # 2. Track Running Maximum for numerical stability
        m_prev = m
        m = tl.maximum(m, tl.max(s, axis=1, keep_dims=True))
        
        # 3. Compute Exp(Scores - Max)
        p = tl.exp(s - m)
        
        # 4. Update Normalization Factor
        l_scale = tl.exp(m_prev - m)
        l = l * l_scale + tl.sum(p, axis=1, keep_dims=True)
        valid_flag = tl.maximum(valid_flag, (s > -float('inf')).to(tl.float32))
        
        # 5. Accumulate Output Representation (P @ V)
        o_acc = o_acc * l_scale + tl.dot(p, v, acc=o_acc)

    # Finalize Output Matrix
    o_final = (o_acc / l).to(tl.bfloat16)
    q_row = q_start + tl.arange(0, BLOCK_Q)
    out_ptr = o_ptr + q_row[:, None] * D + col_D[None, :]
    tl.store(out_ptr, o_final, mask=q_row[:, None] < S)
    
    # Finalize Log Sum Exp Vector
    lse = m + tl.log(l)
    lse_val = tl.where(valid_flag, lse.squeeze(), -float('inf'))
    q_row_1d = q_start + tl.arange(0, BLOCK_Q)
    tl.store(lse_ptr + q_row_1d, lse_val, mask=q_row_1d < S)


def run(Q, K, V, O, LSE):
    """Compute non-causal MHA forward pass returning O and LSE."""
    torch.cuda.set_device(Q.device)
    B, H, S = Q.shape[0], Q.shape[1], Q.shape[2]
    D = 128
    
    # Required for kernels allocating local storage via `make_tensor_descriptor`
    def alloc_fn(size: int, alignment: int, stream):
        return torch.empty(size, device="cuda", dtype=torch.int8)
    triton.set_allocator(alloc_fn)
    
    BLOCK_Q = 64
    BLOCK_KV = 64
    
    grid = (triton.cdiv(S, BLOCK_Q), B * H)
    
    _mha_fwd_kernel[grid](
        Q, K, V, O, LSE,
        H, S, D,
        BLOCK_Q=BLOCK_Q,
        BLOCK_KV=BLOCK_KV,
        num_warps=8,
        num_stages=3,
    )