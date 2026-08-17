import torch
import triton
import triton.language as tl
import math


@triton.jit
def _mha_fwd(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    LSE_ptr,
    S_len,
    D_dim,
    H_dim,
    B_dim,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D_CHUNK: tl.constexpr,
):
    pid_b = tl.program_id(2)
    pid_h = tl.program_id(1)
    pid_q = tl.program_id(0)
    
    q_start = pid_q * BLOCK_M
    q_idx = q_start + tl.arange(0, 64)
    
    b_offset = pid_b * H_dim * S_len * D_dim
    h_offset = pid_h * S_len * D_dim
    base_offset = b_offset + h_offset
    
    O_acc_left = tl.zeros((64, 64), dtype=tl.float32)
    O_acc_right = tl.zeros((64, 64), dtype=tl.float32)
    
    m_prev = float('-inf')
    l_prev = 0.0
    
    num_kv_blocks = tl.cdiv(S_len, 64)
    
    for k_step in range(num_kv_blocks):
        k_start = k_step * 64
        offs_k = k_start + tl.arange(0, 64)
        k_idx = offs_k[None, :]
        k_idx_expanded = k_idx
        q_idx_expanded = q_idx[None, :]
        
        S_acc = tl.zeros((64, 64), dtype=tl.float32)
        
        for d_offset in range(D_dim // D_CHUNK):
            d_idx = d_offset + tl.arange(0, 64)
            
            Q = tl.load(Q_ptr + base_offset + q_start + q_idx[None, :] * D_dim + d_idx[None, :],
                        mask=(q_idx[None, :] < S_len) & (d_idx[None, :] < D_dim), other=0.0)
            K = tl.load(K_ptr + base_offset + k_start + offs_k[None, :] * D_dim + d_idx[None, :],
                        mask=(k_idx < S_len) & (d_idx[None, :] < D_dim), other=0.0)
            
            S_acc = tl.dot(Q, K.T, S_acc)
            
            V = tl.load(V_ptr + base_offset + k_start + offs_k[None, :] * D_dim + d_idx[None, :],
                        mask=(k_idx < S_len) & (d_idx[None, :] < D_dim), other=0.0)
            
            if d_offset == 0:
                O_acc_left = tl.dot(tl.zeros((64, 64), dtype=tl.float32), V, O_acc_left)
            else:
                O_acc_right = tl.dot(tl.zeros((64, 64), dtype=tl.float32), V, O_acc_right)
        
        S = S_acc * scale
        
        mask = (k_idx_expanded >= S_len) | (q_idx_expanded >= S_len)
        S = tl.where(mask, float('-inf'), S)
        
        m_local = tl.max(S, axis=1)
        m_new = tl.maximum(m_prev, m_local)
        
        P = tl.exp(S - m_new[:, None])
        
        l_local = tl.sum(P, axis=1)
        l_new = tl.exp(m_prev - m_new) * l_prev + l_local
        
        O_acc_left = O_acc_left * tl.exp(m_prev - m_new)[:, None]
        O_acc_right = O_acc_right * tl.exp(m_prev - m_new)[:, None]
        
        for d_offset in range(D_dim // D_CHUNK):
            d_idx = d_offset + tl.arange(0, 64)
            V = tl.load(V_ptr + base_offset + k_start + offs_k[None, :] * D_dim + d_idx[None, :],
                        mask=(k_idx < S_len) & (d_idx[None, :] < D_dim), other=0.0)
            
            if d_offset == 0:
                O_acc_left = tl.dot(P, V, O_acc_left)
            else:
                O_acc_right = tl.dot(P, V, O_acc_right)
                
        m_prev = m_new
        l_prev = l_new
        
    l_prev = l_prev[:, None]
    O_left = (O_acc_left / l_prev).to(tl.bfloat16)
    O_right = (O_acc_right / l_prev).to(tl.bfloat16)
    
    q_idx_store = q_start + tl.arange(0, 64)
    tl.store(O_ptr + base_offset + q_idx_store[:, None] * D_dim + tl.arange(0, 64)[None, :],
             O_left, mask=q_idx_store[:, None] < S_len)
    tl.store(O_ptr + base_offset + q_idx_store[:, None] * D_dim + 64 + tl.arange(0, 64)[None, :],
             O_right, mask=q_idx_store[:, None] < S_len)
             
    q_idx_lse = q_start + tl.arange(0, 64)
    m_final = m_prev
    l_final = tl.sum(l_prev, axis=1)
    lse_val = m_final + tl.log(l_final)
    lse_val = tl.where(l_final.squeeze() > 0, lse_val, float('-inf'))
    tl.store(LSE_ptr + pid_b * H_dim * S_len + pid_h * S_len + q_idx_lse, lse_val, mask=q_idx_lse < S_len)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    grid = (triton.cdiv(S, 64), H, B)
    scale = 1.0 / math.sqrt(D)
    _mha_fwd[grid](
        Q, K, V, O, LSE,
        S, D, H, B,
        scale,
        BLOCK_M=64, BLOCK_N=64, D_CHUNK=64,
        num_warps=4, num_stages=3
    )