import torch
import triton
import triton.language as tl
import math
from triton.tools.tensor_descriptor import TensorDescriptor


def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


@triton.jit
def make_mask(pid_q, pid_kv, S, BLOCK_Q: tl.constexpr, BLOCK_KV: tl.constexpr):
    row_idx = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    col_idx = pid_kv * BLOCK_KV + tl.arange(0, BLOCK_KV)
    return (row_idx[:, None] < S) & (col_idx[None, :] < S)


@triton.jit
def _attention_pass1(
    Q_desc, K_desc, P_ptr, S, scale,
    BLOCK_Q: tl.constexpr, BLOCK_KV: tl.constexpr, HALF_D: tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_kv = tl.program_id(1)
    head_id = tl.program_id(2)
    
    row_offs = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    col_kv = pid_kv * BLOCK_KV + tl.arange(0, BLOCK_KV)
    
    q0 = Q_desc.load([head_id * S + row_offs, 0])
    q1 = Q_desc.load([head_id * S + row_offs, HALF_D])
    
    k0 = K_desc.load([head_id * S + col_kv, 0])
    k1 = K_desc.load([head_id * S + col_kv, HALF_D])
    
    p = (tl.dot(q0, k0.T) + tl.dot(q1, k1.T)) * scale
    
    p = tl.where((col_kv[None, :] < S) & (row_offs[:, None] < S), p, -float('inf'))
    
    P_ptr_base = P_ptr + head_id * S * S
    row_idx = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    col_idx = pid_kv * BLOCK_KV + tl.arange(0, BLOCK_KV)
    ptr = P_ptr_base + row_idx[:, None] * S + col_idx[None, :]
    tl.store(ptr, p, mask=make_mask(pid_q, pid_kv, S, BLOCK_Q, BLOCK_KV))


@triton.jit
def _lse_kernel(P_ptr, LSE_ptr, S, total_rows, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    if row >= total_rows:
        return
    
    offsets = tl.arange(0, BLOCK)
    mask = offsets < S
    p = tl.load(P_ptr + row * S + offsets, mask=mask, other=-float('inf'))
    
    max_val = tl.max(p)
    sum_val = tl.sum(tl.exp(p - max_val))
    lse = max_val + tl.log(sum_val)
    
    tl.store(LSE_ptr + row, lse)


@triton.jit
def _attention_pass2(
    P_ptr, V_desc, O_ptr, S, scale, num_kv_tiles,
    BLOCK_Q: tl.constexpr, BLOCK_KV: tl.constexpr, HALF_D: tl.constexpr,
):
    pid_q = tl.program_id(0)
    head_id = tl.program_id(1)
    
    row_offs = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    
    col_offs_all = tl.arange(0, S)
    p_vals = tl.load(P_ptr + head_id * S * S + row_offs[:, None] * S + col_offs_all[None, :],
                     mask=(row_offs[:, None] < S) & (col_offs_all[None, :] < S), other=-float('inf'))
    
    row_max = tl.max(p_vals, dim=1)
    row_sum = tl.sum(tl.exp(p_vals - row_max[:, None]), dim=1)
    
    acc_O_0 = tl.zeros((BLOCK_Q, HALF_D), tl.float32)
    acc_O_1 = tl.zeros((BLOCK_Q, HALF_D), tl.float32)
    
    for i in range(num_kv_tiles):
        col_kv = i * BLOCK_KV + tl.arange(0, BLOCK_KV)
        
        mask_p = (row_offs[:, None] < S) & (col_kv[None, :] < S)
        p = tl.load(P_ptr + head_id * S * S + row_offs[:, None] * S + col_kv[None, :],
                    mask=mask_p, other=-float('inf'))
        
        p = tl.exp(p - row_max[:, None])
        p /= row_sum[:, None]
        
        v0 = V_desc.load([head_id * S + col_kv, 0])
        v1 = V_desc.load([head_id * S + col_kv, HALF_D])
        
        acc_O_0 += tl.dot(p, v0)
        acc_O_1 += tl.dot(p, v1)
    
    col_offs_0 = tl.arange(0, HALF_D)
    col_offs_1 = tl.arange(HALF_D, HALF_D * 2)
    
    tl.store(O_ptr + head_id * S * (HALF_D * 2) + row_offs[:, None] * (HALF_D * 2) + col_offs_0[None, :],
             acc_O_0.to(tl.bfloat16), mask=(row_offs[:, None] < S))
    tl.store(O_ptr + head_id * S * (HALF_D * 2) + row_offs[:, None] * (HALF_D * 2) + col_offs_1[None, :],
             acc_O_1.to(tl.bfloat16), mask=(row_offs[:, None] < S))


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
    triton.set_allocator(alloc_fn)
    P_ptr = alloc_fn(P_size, 256, torch.cuda.current_stream())
    P_f32 = P_ptr.view(torch.float32)
    
    grid1 = (triton.cdiv(S, BLOCK_Q), triton.cdiv(S, BLOCK_KV), B * H)
    _attention_pass1[grid1](Q_desc, K_desc, P_f32, S, scale, BLOCK_Q, BLOCK_KV, HALF_D)
    
    total_rows = B * H * S
    _lse_kernel[(total_rows,)](P_f32, LSE, S, total_rows, BLOCK=S)
    
    num_kv_tiles = triton.cdiv(S, BLOCK_KV)
    grid2 = (triton.cdiv(S, BLOCK_Q), B * H)
    _attention_pass2[grid2](P_f32, V_desc, O, S, scale, num_kv_tiles, BLOCK_Q, BLOCK_KV, HALF_D)