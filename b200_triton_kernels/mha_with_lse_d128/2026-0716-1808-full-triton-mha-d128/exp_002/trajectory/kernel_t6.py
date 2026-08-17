import torch
import triton
import triton.language as tl
import math


@triton.jit
def _attention_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    S, scale, num_kv_tiles,
    BLOCK_Q: tl.constexpr, BLOCK_KV: tl.constexpr, HALF_D: tl.constexpr,
):
    """
    Optimized non-causal attention kernel utilizing FlashAttention's online softmax algorithm.
    
    Replaced raw global memory loads with `tl.make_tensor_descriptor` to resolve 
    the memcheck Out Of Bounds issues caused by manually calculated pointer offsets. 
    Utilizes TMA boundary checks to safely avoid loading out of bounds.
    """
    
    pid_q = tl.program_id(0)
    head_id = tl.program_id(1)
    
    row_idx = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    col_kv_base = tl.arange(0, BLOCK_KV)
    
    row_start = pid_q * BLOCK_Q
    col_offs_0 = tl.arange(0, HALF_D)
    col_offs_1 = tl.arange(HALF_D, HALF_D * 2)
    
    # Create Tensor Descriptors dynamically to leverage TMA hardware on Hopper
    Q_desc = tl.make_tensor_descriptor(
        Q_ptr, shape=[head_id * S + S, HALF_D * 2], strides=[HALF_D * 2, 1],
        block_shape=[BLOCK_Q, HALF_D], padding_option="zero")
    K_desc = tl.make_tensor_descriptor(
        K_ptr, shape=[head_id * S + S, HALF_D * 2], strides=[HALF_D * 2, 1],
        block_shape=[BLOCK_KV, HALF_D], padding_option="zero")
    V_desc = tl.make_tensor_descriptor(
        V_ptr, shape=[head_id * S + S, HALF_D * 2], strides=[HALF_D * 2, 1],
        block_shape=[BLOCK_KV, HALF_D], padding_option="zero")
        
    # Load Q tiles safely using TMA
    q0 = Q_desc.load([row_start, 0])
    q1 = Q_desc.load([row_start, HALF_D])
    
    global_max = tl.full((BLOCK_Q,), -float('inf'), tl.float32)
    global_sum = tl.full((BLOCK_Q,), 0.0, tl.float32)
    acc_O_0 = tl.zeros((BLOCK_Q, HALF_D), tl.float32)
    acc_O_1 = tl.zeros((BLOCK_Q, HALF_D), tl.float32)
    
    for i in range(num_kv_tiles):
        col_start = i * BLOCK_KV
        
        # Load K and V blocks using TMA, boundary checks prevent OOB
        k0 = K_desc.load([col_start, 0])
        k1 = K_desc.load([col_start, HALF_D])
        v0 = V_desc.load([col_start, 0])
        v1 = V_desc.load([col_start, HALF_D])
        
        # Accumulate dot products across the two independently tiled Head Dimensions
        p = (tl.dot(q0, k0.T) + tl.dot(q1, k1.T)) * scale
        
        # Explicit boundary masking to remove OOB contributions in Softmax
        p = tl.where((col_kv_base[None, :] + col_start < S) & (row_idx[:, None] < S), p, -float('inf'))
        
        block_max = tl.max(p, dim=1)
        block_sum = tl.sum(tl.exp(p - block_max[:, None]), dim=1)
        
        old_max = global_max
        new_max = tl.maximum(global_max, block_max)
        global_sum *= tl.exp(global_max - new_max)
        global_sum += block_sum * tl.exp(block_max - new_max)
        global_max = new_max
        
        exp_p = tl.exp(p - new_max[:, None])
        
        acc_O_0 *= tl.exp(old_max[:, None] - new_max[:, None])
        acc_O_0 += tl.dot(exp_p, v0)
        
        acc_O_1 *= tl.exp(old_max[:, None] - new_max[:, None])
        acc_O_1 += tl.dot(exp_p, v1)
    
    acc_O_0 /= global_sum[:, None]
    acc_O_1 /= global_sum[:, None]
    
    # Use native strides for O pointer (D=128)
    tl.store(O_ptr + head_id * S * (HALF_D * 2) + row_idx[:, None] * (HALF_D * 2) + col_offs_0[None, :],
             acc_O_0.to(tl.bfloat16), mask=(row_idx[:, None] < S))
    tl.store(O_ptr + head_id * S * (HALF_D * 2) + row_idx[:, None] * (HALF_D * 2) + col_offs_1[None, :],
             acc_O_1.to(tl.bfloat16), mask=(row_idx[:, None] < S))
    
    lse = global_max + tl.log(global_sum)
    tl.store(LSE_ptr + head_id * S + row_idx,
             lse, mask=(row_idx < S))


def run(Q, K, V, O, LSE):
    """Compute attention with LSE output."""
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    scale = 1.0 / math.sqrt(D)
    num_kv_tiles = triton.cdiv(S, 128)
    grid = (triton.cdiv(S, 128), B * H)
    
    _attention_kernel[grid](
        Q, K, V, O, LSE,
        S, scale, num_kv_tiles,
        BLOCK_Q=128, BLOCK_KV=128, HALF_D=64,
        num_warps=4, num_stages=2
    )