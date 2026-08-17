import torch
import triton
import triton.language as tl
import math


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
    
    num_tiles_s = S // 64
    if i < num_tiles_s:
        Q_i_0 = q_desc.load([i * 64, 0])
        Q_i_1 = q_desc.load([i * 64, 64])
    else:
        Q_i_0 = tl.zeros((64, 64), tl.bfloat16)
        Q_i_1 = tl.zeros((64, 64), tl.bfloat16)
    
    O_0 = tl.zeros((64, 64), tl.float32)
    O_1 = tl.zeros((64, 64), tl.float32)
    m_init = tl.full((64, 1), -1e20, tl.float32)
    m = m_init
    l = tl.full((64, 1), 0.0, tl.float32)
    valid_q = tl.zeros((64, 1), dtype=tl.int1)
    
    global_q = i * 64 + tl.arange(0, 64)
    
    if i < num_tiles_s:
        for j in range(0, i):
            valid_q = valid_q | (global_q < S)[:, None]
            first_valid = ~valid_q
            
            K_j_0 = k_desc.load([j * 64, 0])
            K_j_1 = k_desc.load([j * 64, 64])
            
            S_att = (tl.dot(Q_i_0, K_j_0.T) + tl.dot(Q_i_1, K_j_1.T)) * scale
            S_att = tl.where(global_q[:, None] < S, S_att, -float('inf'))
            
            m_prev = m
            m = tl.maximum(m, tl.max(S_att, axis=1, keepdims=True))
            P = tl.exp(S_att - m)
            P = tl.where(global_q[:, None] < S, P, 0.0)
            
            V_j_0 = v_desc.load([j * 64, 0])
            V_j_1 = v_desc.load([j * 64, 64])
            
            scale_factor = tl.exp(m_prev - m)
            l = l * scale_factor + tl.sum(P, axis=1, keepdims=True)
            O_0 = O_0 * scale_factor + tl.dot(P, V_j_0)
            O_1 = O_1 * scale_factor + tl.dot(P, V_j_1)
            
            m = tl.where(first_valid, m_init, m)
            l = tl.where(first_valid, 0.0, l)
        else:
            j = i
            valid_q = valid_q | (global_q < S)[:, None]
            first_valid = ~valid_q
            
            K_j_0 = k_desc.load([j * 64, 0])
            K_j_1 = k_desc.load([j * 64, 64])
            
            S_att = (tl.dot(Q_i_0, K_j_0.T) + tl.dot(Q_i_1, K_j_1.T)) * scale
            
            global_k = j * 64 + tl.arange(0, 64)
            causal_mask = (global_q[:, None] >= global_k[None, :]) & \
                          (global_q[:, None] < S) & \
                          (global_k[None, :] < S)
            S_att = tl.where(causal_mask, S_att, -float('inf'))
            
            m_prev = m
            m = tl.maximum(m, tl.max(S_att, axis=1, keepdims=True))
            P = tl.exp(S_att - m)
            
            V_j_0 = v_desc.load([j * 64, 0])
            V_j_1 = v_desc.load([j * 64, 64])
            
            scale_factor = tl.exp(m_prev - m)
            l = l * scale_factor + tl.sum(P, axis=1, keepdims=True)
            O_0 = O_0 * scale_factor + tl.dot(P, V_j_0)
            O_1 = O_1 * scale_factor + tl.dot(P, V_j_1)
            
            m = tl.where(first_valid, m_init, m)
            l = tl.where(first_valid, 0.0, l)
    
    inv_l = 1.0 / l
    inv_l = tl.where(l > 0, inv_l, 0.0)
    
    O_0 = O_0 * inv_l
    O_1 = O_1 * inv_l
    
    q_valid = (global_q < S)[:, None]
    o_desc.store([i * 64, 0], O_0.to(tl.bfloat16), mask=q_valid)
    o_desc.store([i * 64, 64], O_1.to(tl.bfloat16), mask=q_valid)
    
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
        Q.contiguous(), K.contiguous(), V.contiguous(), O.contiguous(), LSE,
        H, S, D, scale,
        BLOCK_Q=64, BLOCK_K=64,
        num_warps=4, num_stages=3,
    )