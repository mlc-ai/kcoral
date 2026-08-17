import torch
import triton
import triton.language as tl
import math


@triton.jit
def rowmax_finite(x):
    x_safe = tl.where(tl.isinf(x), 0.0, x)
    return tl.max(x_safe, axis=1, keepdims=True)


@triton.jit
def _mha_kernel(
    Q_ptr, 
    K_ptr, 
    V_ptr, 
    O_ptr, 
    LSE_ptr, 
    H, 
    S, 
    D,
    scale,
    BLOCK_Q: tl.constexpr, 
    BLOCK_K: tl.constexpr,
):
    b_idx = tl.program_id(2)
    h_idx = tl.program_id(1)
    i = tl.program_id(0)
    
    base_offset = (b_idx * H + h_idx) * S * D
    
    q_desc = tl.make_tensor_descriptor(Q_ptr + base_offset, [S, D], [D, 1], [64, 64], "zero")
    k_desc = tl.make_tensor_descriptor(K_ptr + base_offset, [S, D], [D, 1], [64, 64], "zero")
    v_desc = tl.make_tensor_descriptor(V_ptr + base_offset, [S, D], [D, 1], [64, 64], "zero")
    o_desc = tl.make_tensor_descriptor(O_ptr + base_offset, [S, D], [D, 1], [64, 64])
    
    Q_i_0 = q_desc.load([i * 64, 0])
    Q_i_1 = q_desc.load([i * 64, 64])
    
    O_0 = tl.zeros((64, 64), tl.float32)
    O_1 = tl.zeros((64, 64), tl.float32)
    m = tl.full((64, 1), -float('inf'), tl.float32)
    l = tl.full((64, 1), 0.0, tl.float32)
    
    global_q = i * 64 + tl.arange(0, 64)
    q_valid = (global_q < S)[:, None]
    
    for j in range(0, i + 1):
        K_j_0 = k_desc.load([j * 64, 0])
        K_j_1 = k_desc.load([j * 64, 64])
        V_j_0 = v_desc.load([j * 64, 0])
        V_j_1 = v_desc.load([j * 64, 64])
        
        S_att = (tl.dot(Q_i_0, K_j_0.T) + tl.dot(Q_i_1, K_j_1.T)) * scale 
        
        valid_mask = None
        if j == i:
            global_k = j * 64 + tl.arange(0, 64)
            valid_mask = (global_q[:, None] >= global_k[None, :]) & \
                         (global_q[:, None] < S) & \
                         (global_k[None, :] < S)
            S_att = tl.where(valid_mask, S_att, -float('inf'))
        else:
            S_att = tl.where(q_valid, S_att, -float('inf'))
        
        m_prev = m
        m_new = rowmax_finite(S_att)
        m = tl.maximum(m, m_new)
        P = tl.exp(S_att - m)
        
        if j == i:
            P = tl.where(valid_mask, P, 0.0)
        
        l = l * tl.exp(m_prev - m) + tl.sum(P, axis=1, keepdims=True)
        O_0 = O_0 * tl.exp(m_prev - m) + tl.dot(P, V_j_0)
        O_1 = O_1 * tl.exp(m_prev - m) + tl.dot(P, V_j_1)
        
    inv_l = 1.0 / l
    
    O_0 = tl.where(global_q[:, None] < S, O_0 * inv_l, 0.0)
    O_1 = tl.where(global_q[:, None] < S, O_1 * inv_l, 0.0)
    
    o_desc.store([i * 64, 0], O_0.to(tl.bfloat16))
    o_desc.store([i * 64, 64], O_1.to(tl.bfloat16))
    
    m_squeezed = m.squeeze(1)
    l_squeezed = l.squeeze(1)
    lse = m_squeezed + tl.log(l_squeezed)
    lse = tl.where(global_q < S, lse, float('nan'))
    
    q_idx = i * 64 + tl.arange(0, 64)
    lse_ptr = LSE_ptr + b_idx * H * S + h_idx * S + q_idx
    tl.store(lse_ptr, lse, mask=(q_idx < S))


def run(Q, K, V, O, LSE):
    """Execute optimized causal multi-head attention computation returning precise outputs and accurate LSE"""
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    scale = 1.0 / math.sqrt(D)
    
    def alloc_fn(size: int, alignment: int, stream):
        return torch.empty(size, device="cuda", dtype=torch.int8)
    triton.set_allocator(alloc_fn)
    
    num_blocks_q = triton.cdiv(S, 64)
    grid = (num_blocks_q, H, B)
    
    _mha_kernel[grid](
        Q, K, V, O, LSE,
        H, S, D, scale,
        BLOCK_Q=64, BLOCK_K=64,
        num_warps=4, num_stages=3,
    )