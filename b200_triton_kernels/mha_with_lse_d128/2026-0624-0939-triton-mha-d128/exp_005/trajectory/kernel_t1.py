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
    
    q_row = q_start + tl.arange(0, 64)
    d_col = tl.arange(0, 128)[None, :]
    
    Q = tl.load(Q_ptr + base_offset + q_row[:, None] * D_dim + d_col,
                mask=q_row[:, None] < S_len, other=0.0)
    
    O_acc = tl.zeros((64, 128), dtype=tl.float32)
    m_prev = tl.full((64, 1), float('-inf'), dtype=tl.float32)
    l_prev = tl.full((64, 1), 0.0, dtype=tl.float32)
    
    num_kv_blocks = tl.cdiv(S_len, 64)
    
    for k_step in range(num_kv_blocks):
        k_start = k_step * 64
        k_row = k_start + tl.arange(0, 64)
        
        K = tl.load(K_ptr + base_offset + k_row[:, None] * D_dim + d_col,
                    mask=k_row[:, None] < S_len, other=0.0)
        V = tl.load(V_ptr + base_offset + k_row[:, None] * D_dim + d_col,
                    mask=k_row[:, None] < S_len, other=0.0)
        
        S = tl.dot(Q, K.T, acc=0.0)
        S = S * scale
        
        m_local = tl.max(S, axis=1, keep_dims=True)
        m_new = tl.maximum(m_prev, m_local)
        
        P = tl.exp(S - m_new)
        
        l_local = tl.sum(P, axis=1, keep_dims=True)
        l_new = tl.exp(m_prev - m_new) * l_prev + l_local
        
        O_acc = O_acc * tl.exp(m_prev - m_new)
        O_acc = tl.dot(P, V, acc=O_acc)
        
        m_prev = m_new
        l_prev = l_new
        
    O_out = (O_acc / l_prev).to(tl.bfloat16)
    tl.store(O_ptr + base_offset + q_row[:, None] * D_dim + d_col,
             O_out, mask=q_row[:, None] < S_len)
             
    valid_q = q_row < S_len
    lse_val = m_prev + tl.log(l_prev)
    lse_val = tl.where(l_prev > 0, lse_val, float('-inf'))
    lse_val = lse_val[:, 0]
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