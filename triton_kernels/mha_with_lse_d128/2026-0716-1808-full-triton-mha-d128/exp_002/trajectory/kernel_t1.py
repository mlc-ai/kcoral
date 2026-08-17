import torch
import triton
import triton.language as tl
import math
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _attention_kernel(
    Q_desc, K_desc, V_desc, O_ptr, LSE_ptr,
    S, scale, num_kv_tiles,
    BLOCK_Q: tl.constexpr, BLOCK_KV: tl.constexpr, HALF_D: tl.constexpr,
):
    """
    Optimized non-causal attention kernel utilizing FlashAttention's online softmax algorithm.
    
    Replaced raw pointer arithmetic with robust TMA TensorDescriptors. 
    Reduced register pressure and shared memory usage by tiling the Head Dimension (D) 
    into two independent 64-element matrices.
    """
    
    pid_q = tl.program_id(0)
    head_id = tl.program_id(1)
    
    row_offs = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    
    # Load Q tiles safely using TMA. Zero padding ensures robust boundary handling.
    q0 = Q_desc.load([head_id * S + row_offs, 0])
    q1 = Q_desc.load([head_id * S + row_offs, HALF_D])
    
    global_max = -float('inf')
    global_sum = 0.0
    acc_O_0 = tl.zeros((BLOCK_Q, HALF_D), tl.float32)
    acc_O_1 = tl.zeros((BLOCK_Q, HALF_D), tl.float32)
    
    for i in range(num_kv_tiles):
        col_kv = i * BLOCK_KV + tl.arange(0, BLOCK_KV)
        
        # Load K and V blocks using TMA
        k0 = K_desc.load([head_id * S + col_kv, 0])
        k1 = K_desc.load([head_id * S + col_kv, HALF_D])
        v0 = V_desc.load([head_id * S + col_kv, 0])
        v1 = V_desc.load([head_id * S + col_kv, HALF_D])
        
        # Accumulate dot products across the two independently tiled Head Dimensions
        p = (tl.dot(q0, k0.T) + tl.dot(q1, k1.T)) * scale
        
        # Explicit boundary masking 
        p = tl.where((col_kv[None, :] < S), p, -float('inf'))
        
        block_max = tl.max(p, dim=1)
        block_sum = tl.sum(tl.exp(p - block_max[:, None]), dim=1)
        
        old_max = global_max
        new_max = tl.maximum(global_max, block_max)
        global_sum = global_sum * tl.exp(global_max - new_max)
        global_sum = global_sum + block_sum * tl.exp(block_max - new_max)
        global_max = new_max
        
        acc_O_0 = acc_O_0 * tl.exp(old_max[:, None] - new_max[:, None])
        acc_O_0 += tl.dot(tl.exp(p - new_max[:, None]), v0)
        
        acc_O_1 = acc_O_1 * tl.exp(old_max[:, None] - new_max[:, None])
        acc_O_1 += tl.dot(tl.exp(p - new_max[:, None]), v1)
    
    acc_O_0 = acc_O_0 / global_sum[:, None]
    acc_O_1 = acc_O_1 / global_sum[:, None]
    
    col_offs_0 = tl.arange(0, HALF_D)
    col_offs_1 = tl.arange(HALF_D, HALF_D * 2)
    
    tl.store(O_ptr + head_id * S * (HALF_D * 2) + row_offs[:, None] * (HALF_D * 2) + col_offs_0[None, :],
             acc_O_0.to(tl.bfloat16), mask=(row_offs[:, None] < S))
    tl.store(O_ptr + head_id * S * (HALF_D * 2) + row_offs[:, None] * (HALF_D * 2) + col_offs_1[None, :],
             acc_O_1.to(tl.bfloat16), mask=(row_offs[:, None] < S))
    
    tl.store(LSE_ptr + head_id * S + row_offs,
             global_max + tl.log(global_sum), mask=(row_offs < S))


def run(Q, K, V, O, LSE):
    """Compute attention with LSE output."""
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    scale = 1.0 / math.sqrt(D)
    num_kv_tiles = triton.cdiv(S, 128)
    grid = (triton.cdiv(S, 128), B * H)
    
    Q_desc = TensorDescriptor.from_tensor(Q, [128, 64])
    K_desc = TensorDescriptor.from_tensor(K, [128, 64])
    V_desc = TensorDescriptor.from_tensor(V, [128, 64])
    
    _attention_kernel[grid](
        Q_desc, K_desc, V_desc, O, LSE,
        S, scale, num_kv_tiles,
        BLOCK_Q=128, BLOCK_KV=128, HALF_D=64,
        num_warps=4, num_stages=2
    )