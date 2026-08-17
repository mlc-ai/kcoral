import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
import math


@triton.jit
def _mha_kernel(
    Q_desc, 
    K_desc, 
    V_desc, 
    O_desc, 
    LSE_ptr, 
    H, 
    S, 
    scale,
    BLOCK_Q: tl.constexpr, 
    BLOCK_K: tl.constexpr,
):
    b_idx = tl.program_id(2)
    h_idx = tl.program_id(1)
    i = tl.program_id(0)
    
    b_h_idx = b_idx * H + h_idx
    
    Q_i_0 = Q_desc.load([b_h_idx, i * 64, 0])
    Q_i_1 = Q_desc.load([b_h_idx, i * 64, 64])
    
    O_0 = tl.zeros((64, 64), tl.float32)
    O_1 = tl.zeros((64, 64), tl.float32)
    m = tl.full((64, 1), -float('inf'), tl.float32)
    l = tl.full((64, 1), 0.0, tl.float32)
    
    for j in range(0, i + 1):
        K_j_0 = K_desc.load([b_h_idx, j * 64, 0])
        K_j_1 = K_desc.load([b_h_idx, j * 64, 64])
        V_j_0 = V_desc.load([b_h_idx, j * 64, 0])
        V_j_1 = V_desc.load([b_h_idx, j * 64, 64])
        
        S_att = (tl.dot(Q_i_0, K_j_0.T) + tl.dot(Q_i_1, K_j_1.T)) * scale 
        
        valid_mask = None
        if j == i:
            global_q = i * 64 + tl.arange(0, 64)
            global_k = j * 64 + tl.arange(0, 64)
            valid_mask = (global_q[:, None] >= global_k[None, :]) & \
                         (global_q[:, None] < S) & \
                         (global_k[None, :] < S)
            S_att = tl.where(valid_mask, S_att, -float('inf'))
        
        m_prev = m
        m = tl.maximum(m, tl.max(S_att, axis=1, keepdims=True))
        P = tl.exp(S_att - m)
        
        if j == i:
            P = tl.where(valid_mask, P, 0.0)
        
        l = l * tl.exp(m_prev - m) + tl.sum(P, axis=1, keepdims=True)
        
        O_0 = O_0 * tl.exp(m_prev - m) + tl.dot(P, V_j_0)
        O_1 = O_1 * tl.exp(m_prev - m) + tl.dot(P, V_j_1)
        
    inv_l = 1.0 / l
    
    global_q = i * 64 + tl.arange(0, 64)
    O_0 = tl.where(global_q[:, None] < S, O_0 * inv_l, 0.0)
    O_1 = tl.where(global_q[:, None] < S, O_1 * inv_l, 0.0)
    
    o_desc.store([b_h_idx, i * 64, 0], O_0.to(tl.bfloat16))
    o_desc.store([b_h_idx, i * 64, 64], O_1.to(tl.bfloat16))
    
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
    
    Q_desc = TensorDescriptor.from_tensor(Q.contiguous(), [1, 64, 64])
    K_desc = TensorDescriptor.from_tensor(K.contiguous(), [1, 64, 64])
    V_desc = TensorDescriptor.from_tensor(V.contiguous(), [1, 64, 64])
    O_desc = TensorDescriptor.from_tensor(O.contiguous(), [1, 64, 64])
    
    num_blocks_q = triton.cdiv(S, 64)
    grid = (num_blocks_q, H, B)
    
    _mha_kernel[grid](
        Q_desc, K_desc, V_desc, O_desc, LSE,
        H, S, scale,
        BLOCK_Q=64, BLOCK_K=64,
        num_warps=4, num_stages=3,
    )