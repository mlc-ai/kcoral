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
    
    Fixed the TMA offset error and memcheck issues by replacing TensorDescriptors with robust raw pointer arithmetic.
    Eliminated the massive temporary P matrix allocation by utilizing a single-pass online softmax.
    Tiles the Head Dimension (D=128) into two independent 64-element matrices to manage register pressure.
    """
    
    pid_q = tl.program_id(0)
    head_id = tl.program_id(1)
    
    row_idx = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    col_offs_0 = tl.arange(0, HALF_D)
    col_offs_1 = tl.arange(HALF_D, HALF_D * 2)
    
    # Load Q tiles safely using raw pointers. Zero padding ensures robust boundary handling.
    q0 = tl.load(Q_ptr + head_id * S * (HALF_D * 2) + row_idx[:, None] * (HALF_D * 2) + col_offs_0[None, :],
                 mask=(row_idx[:, None] < S), other=0.0)
    q1 = tl.load(Q_ptr + head_id * S * (HALF_D * 2) + row_idx[:, None] * (HALF_D * 2) + col_offs_1[None, :],
                 mask=(row_idx[:, None] < S), other=0.0)
    
    global_max = -float('inf')
    global_sum = 0.0
    acc_O_0 = tl.zeros((BLOCK_Q, HALF_D), tl.float32)
    acc_O_1 = tl.zeros((BLOCK_Q, HALF_D), tl.float32)
    
    for i in range(num_kv_tiles):
        col_kv = i * BLOCK_KV + tl.arange(0, BLOCK_KV)
        
        # Load K and V blocks
        k0 = tl.load(K_ptr + head_id * S * (HALF_D * 2) + col_kv[:, None] * (HALF_D * 2) + col_offs_0[None, :],
                     mask=(col_kv[:, None] < S), other=0.0)
        k1 = tl.load(K_ptr + head_id * S * (HALF_D * 2) + col_kv[:, None] * (HALF_D * 2) + col_offs_1[None, :],
                     mask=(col_kv[:, None] < S), other=0.0)
        
        v0 = tl.load(V_ptr + head_id * S * (HALF_D * 2) + col_kv[:, None] * (HALF_D * 2) + col_offs_0[None, :],
                     mask=(col_kv[:, None] < S), other=0.0)
        v1 = tl.load(V_ptr + head_id * S * (HALF_D * 2) + col_kv[:, None] * (HALF_D * 2) + col_offs_1[None, :],
                     mask=(col_kv[:, None] < S), other=0.0)
        
        # Accumulate dot products across the two independently tiled Head Dimensions
        p = (tl.dot(q0, k0.T) + tl.dot(q1, k1.T)) * scale
        
        block_max = tl.max(p, axis=1)
        
        new_max = max(global_max, block_max[row_idx])
        global_sum *= math.exp(global_max - new_max)
        global_sum += tl.sum(math.exp(p - new_max))
        global_max = new_max
        
        old_max = global_max
        
        acc_O_0 *= math.exp(old_max - new_max)
        acc_O_0 += tl.dot(math.exp(p - new_max) * v0, tl.ones((BLOCK_KV, HALF_D)))
        
        acc_O_1 *= math.exp(old_max - new_max)
        acc_O_1 += tl.dot(math.exp(p - new_max) * v1, tl.ones((BLOCK_KV, HALF_D)))
    
    acc_O_0 /= math.exp(global_max)
    acc_O_1 /= math.exp(global_max)
    
    # Use native strides for O pointer (D=128)
    tl.store(O_ptr + head_id * S * (HALF_D * 2) + row_idx[:, None] * (HALF_D * 2) + col_offs_0[None, :],
             acc_O_0.to(tl.bfloat16), mask=(row_idx[:, None] < S))
    tl.store(O_ptr + head_id * S * (HALF_D * 2) + row_idx[:, None] * (HALF_D * 2) + col_offs_1[None, :],
             acc_O_1.to(tl.bfloat16), mask=(row_idx[:, None] < S))
    
    lse = global_max + math.log(global_sum)
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
    )