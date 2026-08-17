import torch
import triton
import triton.language as tl
import math
from triton.tools.tensor_descriptor import TensorDescriptor


def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


@triton.jit
def _attention_pass1(
    Q_desc, K_desc, P_ptr, LSE_ptr, S, scale,
    BLOCK_Q: tl.constexpr, BLOCK_KV: tl.constexpr, HALF_D: tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_kv = tl.program_id(1)
    head_id = tl.program_id(2)
    
    row_start = pid_q * BLOCK_Q
    col_start = pid_kv * BLOCK_KV
    
    # Replaced problematic tensor offsets with clean scalar offsets 
    q0 = Q_desc.load([head_id * S + row_start, 0])
    q1 = Q_desc.load([head_id * S + row_start, HALF_D])
    
    k0 = K_desc.load([head_id * S + col_start, 0])
    k1 = K_desc.load([head_id * S + col_start, HALF_D])
    
    p = (tl.dot(q0, k0.T) + tl.dot(q1, k1.T)) * scale
    
    row_idx = row_start + tl.arange(0, BLOCK_Q)
    col_idx = col_start + tl.arange(0, BLOCK_KV)
    
    p = tl.where((col_idx[None, :] < S) & (row_idx[:, None] < S), p, -float('inf'))
    
    P_ptr_base = P_ptr + head_id * S * S
    ptr = P_ptr_base + row_idx[:, None] * S + col_idx[None, :]
    tl.store(ptr, p, mask=(row_idx[:, None] < S) & (col_idx[None, :] < S))


@triton.jit
def _lse_kernel(P_ptr, LSE_ptr, S, total_rows, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    if row >= total_rows:
        return
    
    # Evaluation found standard single-shot load triggered massive register spills.
    # Switched to an aggregated 2-pass approach mapping directly over concrete chunks.
    max_val = -float('inf')
    sum_val = 0.0
    
    for i in range(0, S, BLOCK):
        offsets = tl.arange(0, BLOCK)
        mask = (i + offsets) < S
        p = tl.load(P_ptr + row * S + i + offsets, mask=mask, other=-float('inf'))
        
        block_max = tl.max(p)
        block_sum = tl.sum(tl.exp(p - block_max))
        
        new_max = tl.maximum(max_val, block_max)
        sum_val = sum_val * tl.exp(max_val - new_max) + block_sum * tl.exp(block_max - new_max)
        max_val = new_max
    
    lse = max_val + tl.log(sum_val)
    
    tl.store(LSE_ptr + row, lse)


@triton.jit
def _attention_pass2(
    P_ptr, V_desc, O_ptr, S, scale, num_kv_tiles,
    BLOCK_Q: tl.constexpr, BLOCK_KV: tl.constexpr, HALF_D: tl.constexpr,
):
    pid_q = tl.program_id(0)
    head_id = tl.program_id(1)
    
    row_start = pid_q * BLOCK_Q
    row_idx = row_start + tl.arange(0, BLOCK_Q)
    
    global_max = tl.full((BLOCK_Q,), -float('inf'), tl.float32)
    global_sum = tl.full((BLOCK_Q,), 0.0, tl.float32)
    acc_O_0 = tl.zeros((BLOCK_Q, HALF_D), tl.float32)
    acc_O_1 = tl.zeros((BLOCK_Q, HALF_D), tl.float32)
    
    for i in range(num_kv_tiles):
        col_start = i * BLOCK_KV
        col_kv = col_start + tl.arange(0, BLOCK_KV)
        
        mask_p = (row_idx[:, None] < S) & (col_kv[None, :] < S)
        p = tl.load(P_ptr + head_id * S * S + row_idx[:, None] * S + col_kv[None, :],
                    mask=mask_p, other=-float('inf'))
        
        block_max = tl.max(p, dim=1)
        block_sum = tl.sum(tl.exp(p - block_max[:, None]), dim=1)
        
        old_max = global_max
        new_max = tl.maximum(global_max, block_max)
        global_sum = global_sum * tl.exp(global_max - new_max) + block_sum * tl.exp(block_max - new_max)
        global_max = new_max
        
        exp_p = tl.exp(p - new_max[:, None])
        
        v0 = V_desc.load([head_id * S + col_start, 0])
        v1 = V_desc.load([head_id * S + col_start, HALF_D])
        
        acc_O_0 = acc_O_0 * tl.exp(old_max[:, None] - new_max[:, None]) + tl.dot(exp_p, v0)
        acc_O_1 = acc_O_1 * tl.exp(old_max[:, None] - new_max[:, None]) + tl.dot(exp_p, v1)
    
    acc_O_0 /= global_sum[:, None]
    acc_O_1 /= global_sum[:, None]
    
    col_offs_0 = tl.arange(0, HALF_D)
    col_offs_1 = tl.arange(HALF_D, HALF_D * 2)
    
    tl.store(O_ptr + head_id * S * (HALF_D * 2) + row_idx[:, None] * (HALF_D * 2) + col_offs_0[None, :],
             acc_O_0.to(tl.bfloat16), mask=(row_idx[:, None] < S))
    tl.store(O_ptr + head_id * S * (HALF_D * 2) + row_idx[:, None] * (HALF_D * 2) + col_offs_1[None, :],
             acc_O_1.to(tl.bfloat16), mask=(row_idx[:, None] < S))


def run(Q, K, V, O, LSE):
    """Compute attention with LSE output."""
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    scale = 1.0 / math.sqrt(D)
    BLOCK_Q = 128
    BLOCK_KV = 128
    HALF_D = 64
    
    Q_f = Q.flatten(0, 2)
    K_f = K.flatten(0, 2)
    V_f = V.flatten(0, 2)
    
    Q_desc = TensorDescriptor.from_tensor(Q_f, [BLOCK_Q, HALF_D])
    K_desc = TensorDescriptor.from_tensor(K_f, [BLOCK_KV, HALF_D])
    V_desc = TensorDescriptor.from_tensor(V_f, [BLOCK_KV, HALF_D])
    
    P_size = B * H * S * S * 4
    P_ptr = alloc_fn(P_size, 256, torch.cuda.current_stream())
    P_f32 = P_ptr.view(torch.float32)
    
    grid1 = (triton.cdiv(S, BLOCK_Q), triton.cdiv(S, BLOCK_KV), B * H)
    _attention_pass1[grid1](Q_desc, K_desc, P_f32, LSE, S, scale, BLOCK_Q, BLOCK_KV, HALF_D)
    
    total_rows = B * H * S
    _lse_kernel[(total_rows,)](P_f32, LSE, S, total_rows, BLOCK=256)
    
    num_kv_tiles = triton.cdiv(S, BLOCK_KV)
    grid2 = (triton.cdiv(S, BLOCK_Q), B * H)
    _attention_pass2[grid2](P_f32, V_desc, O, S, scale, num_kv_tiles, BLOCK_Q, BLOCK_KV, HALF_D)