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
):
    assert D_dim == 128, "D must be 128"
    
    pid_b = tl.program_id(2)
    pid_h = tl.program_id(1)
    pid_q = tl.program_id(0)
    
    q_start = pid_q * BLOCK_M
    base_offset = pid_b * H_dim * S_len * D_dim + pid_h * S_len * D_dim
    b_h_offset = pid_b * H_dim * S_len + pid_h * S_len
    
    q_row = q_start + tl.arange(0, BLOCK_M)
    
    d_cols_left = tl.arange(0, 64)[None, :]
    d_cols_right = 64 + tl.arange(0, 64)[None, :]
    
    Q0 = tl.load(Q_ptr + base_offset + q_row[:, None] * D_dim + d_cols_left,
                 mask=q_row[:, None] < S_len, other=0.0)
    Q1 = tl.load(Q_ptr + base_offset + q_row[:, None] * D_dim + d_cols_right,
                 mask=q_row[:, None] < S_len, other=0.0)
                 
    Q0 = Q0.to(tl.float32)
    Q1 = Q1.to(tl.float32)
    
    O_acc_left = tl.zeros((BLOCK_M, 64), dtype=tl.float32)
    O_acc_right = tl.zeros((BLOCK_M, 64), dtype=tl.float32)
    
    m_prev = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)
    l_prev = tl.full((BLOCK_M,), 0.0, dtype=tl.float32)
    
    num_kv_blocks = tl.cdiv(S_len, BLOCK_N)
    
    for k_step in range(num_kv_blocks):
        k_start = k_step * BLOCK_N
        k_row = k_start + tl.arange(0, BLOCK_N)
        
        K0 = tl.load(K_ptr + base_offset + k_row[:, None] * D_dim + d_cols_left,
                     mask=k_row[:, None] < S_len, other=0.0)
        K1 = tl.load(K_ptr + base_offset + k_row[:, None] * D_dim + d_cols_right,
                     mask=k_row[:, None] < S_len, other=0.0)
                     
        V0 = tl.load(V_ptr + base_offset + k_row[:, None] * D_dim + d_cols_left,
                     mask=k_row[:, None] < S_len, other=0.0)
        V1 = tl.load(V_ptr + base_offset + k_row[:, None] * D_dim + d_cols_right,
                     mask=k_row[:, None] < S_len, other=0.0)
        
        K0 = K0.to(tl.float32)
        K1 = K1.to(tl.float32)
        V0 = V0.to(tl.float32)
        V1 = V1.to(tl.float32)
        
        S_acc = tl.dot(Q0, K0.T, acc=None)
        S_acc = tl.dot(Q1, K1.T, acc=S_acc)
        S = S_acc * scale
        
        valid_k = k_row[None, :] < S_len
        S = tl.where(valid_k, S, float('-inf'))
        
        m_local = tl.max(S, axis=1)
        m_new = tl.maximum(m_prev, m_local)
        
        scale_factor = tl.exp(m_prev - m_new)
        valid_scale = m_new > float('-inf')
        scale_factor = tl.where(valid_scale, scale_factor, 0.0)
        
        P = tl.exp(S - m_new[:, None])
        P = tl.where(valid_scale[:, None], P, 0.0)
        
        l_local = tl.sum(P, axis=1)
        l_new = scale_factor * l_prev + l_local
        
        O_acc_left = O_acc_left * scale_factor[:, None]
        O_acc_left = tl.dot(P, V0, acc=O_acc_left)
        
        O_acc_right = O_acc_right * scale_factor[:, None]
        O_acc_right = tl.dot(P, V1, acc=O_acc_right)
        
        m_prev = m_new
        l_prev = l_new
        
    valid_l = l_prev > 0
    O_bf16_left = (O_acc_left / l_prev[:, None]).to(tl.bfloat16)
    O_out_left = tl.where(valid_l[:, None], O_bf16_left, 0.0)
    
    O_bf16_right = (O_acc_right / l_prev[:, None]).to(tl.bfloat16)
    O_out_right = tl.where(valid_l[:, None], O_bf16_right, 0.0)
    
    valid_q = q_row < S_len
    
    tl.store(O_ptr + base_offset + q_row[:, None] * D_dim + d_cols_left,
             O_out_left, mask=valid_q[:, None])
    tl.store(O_ptr + base_offset + q_row[:, None] * D_dim + d_cols_right,
             O_out_right, mask=valid_q[:, None])
             
    lse_val = m_prev + tl.log(l_prev)
    lse_val = tl.where(valid_l, lse_val, float('-inf'))
    tl.store(LSE_ptr + b_h_offset + q_row, lse_val, mask=valid_q)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    grid = (triton.cdiv(S, 64), H, B)
    scale = 1.0 / math.sqrt(D)
    _mha_fwd[grid](
        Q, K, V, O, LSE,
        S, D, H, B, scale,
        BLOCK_M=64, BLOCK_N=64,
        num_warps=4, num_stages=3
    )