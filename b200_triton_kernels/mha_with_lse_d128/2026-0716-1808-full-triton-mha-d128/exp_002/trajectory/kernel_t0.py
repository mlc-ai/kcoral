import torch
import triton
import triton.language as tl
import math


@triton.jit
def _attention_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    S, scale, num_kv_tiles,
    BLOCK_Q: tl.constexpr, BLOCK_KV: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    """
    Optimized non-causal attention kernel utilizing FlashAttention's online softmax algorithm.
    
    Logic & Layout:
    1. We conceptualize the (B, H) dimensions merged into a single flattened dimension of size 192.
    2. The grid assigns one program per query block (128 rows) per head. 
    3. Inside the kernel, we loop sequentially across the Key/Value blocks.
    4. Using online exponentials tracking (`global_max`, `global_sum`, `acc_O`), 
       we avoid materializing the giant unnormalized attention probability matrix P.
    
    Masking Strategy:
    - Out of bounds Loads: Handled gracefully by passing `other=0.0` in `tl.load`. 
    - Softmax Numerator Correction: Crucially, inside the softmax calculation we inject 
      `-float('inf')` for any sequence index >= S. This guarantees `exp(-inf) = 0`, 
      ensuring fully masked rows contribute exactly zero to the denominator and output.
    """
    
    pid_q = tl.program_id(0)
    head_id = tl.program_id(1)
    
    stride_head = S * HEAD_DIM
    row_offs = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    col_offs = tl.arange(0, HEAD_DIM)
    
    q = tl.load(Q_ptr + head_id * stride_head + row_offs[:, None] * HEAD_DIM + col_offs[None, :],
                mask=(row_offs[:, None] < S), other=0.0)
    
    global_max = tl.full((BLOCK_Q,), -float('inf'), tl.float32)
    global_sum = tl.full((BLOCK_Q,), 0.0, tl.float32)
    acc_O = tl.zeros((BLOCK_Q, HEAD_DIM), tl.float32)
    
    for i in range(num_kv_tiles):
        col_kv = i * BLOCK_KV + tl.arange(0, BLOCK_KV)
        
        k = tl.load(K_ptr + head_id * stride_head + col_kv[:, None] * HEAD_DIM + col_offs[None, :],
                    mask=(col_kv[:, None] < S), other=0.0)
        v = tl.load(V_ptr + head_id * stride_head + col_kv[:, None] * HEAD_DIM + col_offs[None, :],
                    mask=(col_kv[:, None] < S), other=0.0)
        
        p = tl.dot(q, k.T) * scale
        
        # Replacing standard boundary masking with an explicit `-inf` injection 
        # preserves correct normalization dynamics across software pipelined iterations
        p = tl.where((col_kv[None, :] < S), p, -float('inf'))
        
        block_max = tl.max(p, dim=1)
        block_sum = tl.sum(tl.exp(p - block_max[:, None]), dim=1)
        
        old_max = global_max
        new_max = tl.maximum(global_max, block_max)
        global_sum *= tl.exp(global_max - new_max)
        global_sum += block_sum * tl.exp(block_max - new_max)
        global_max = new_max
        
        acc_O *= tl.exp(old_max[:, None] - new_max[:, None])
        acc_O += tl.dot(tl.exp(p - new_max[:, None]), v)
    
    acc_O /= global_sum[:, None]
    
    tl.store(O_ptr + head_id * stride_head + row_offs[:, None] * HEAD_DIM + col_offs[None, :],
             acc_O.to(tl.bfloat16), mask=(row_offs[:, None] < S))
    
    tl.store(LSE_ptr + head_id * S + row_offs,
             global_max + tl.log(global_sum), mask=(row_offs < S))


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
        BLOCK_Q=128, BLOCK_KV=128, HEAD_DIM=128,
        num_warps=4, num_stages=3
    )