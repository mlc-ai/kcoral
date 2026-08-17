import torch
import triton
import triton.language as tl
import math
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _mha_fwd_persistent(
    Q_desc, K_desc, V_desc, O_ptr, LSE_ptr,
    S_len, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, NUM_SMS: tl.constexpr,
):
    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(S_len, BLOCK_M)
    B = O_ptr.shape[0]
    H_dim = O_ptr.shape[1]
    D_dim = O_ptr.shape[3]
    num_tiles = num_pid_m * B * H_dim
    
    num_kv_blocks = tl.cdiv(S_len, 64)
    
    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=False):
        idx = tile_id // num_pid_m
        pid_m = tile_id % num_pid_m
        pid_h = idx % H_dim
        pid_b = idx // H_dim
        
        q_start = pid_m * BLOCK_M
        
        Q = Q_desc.load([pid_b, pid_h, q_start, 0])
        
        acc_O_0 = tl.zeros((BLOCK_M, 64), dtype=tl.float32)
        acc_O_1 = tl.zeros((BLOCK_M, 64), dtype=tl.float32)
        m_prev = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)
        l_prev = tl.full((BLOCK_M,), 0.0, dtype=tl.float32)
        
        for k_step in range(num_kv_blocks):
            k_start = k_step * 64
            
            K = K_desc.load([pid_b, pid_h, k_start, 0])
            V = V_desc.load([pid_b, pid_h, k_start, 0])
            
            S = tl.dot(Q, K.T)
            S = S * scale
            
            m_local = tl.max(S, axis=1)
            m_new = tl.maximum(m_prev, m_local)
            
            scale_factor = tl.exp(m_prev - m_new)
            valid_scale = m_new > float('-inf')
            scale_factor = tl.where(valid_scale, scale_factor, 0.0)
            
            P = tl.exp(S - m_new[:, None])
            P = tl.where(valid_scale[:, None], P, 0.0)
            
            l_local = tl.sum(P, axis=1)
            l_new = scale_factor * l_prev + l_local
            
            acc_O_0 = acc_O_0 * scale_factor[:, None]
            acc_O_0 = tl.dot(P, V[:, :64], acc=acc_O_0)
            
            acc_O_1 = acc_O_1 * scale_factor[:, None]
            acc_O_1 = tl.dot(P, V[:, 64:], acc=acc_O_1)
            
            m_prev = m_new
            l_prev = l_new
        
        l_final = l_prev
        m_final = m_prev
        
        valid_l = l_final > 0
        O_bf16_0 = (acc_O_0 / l_final[:, None]).to(tl.bfloat16)
        O_out_0 = tl.where(valid_l[:, None], O_bf16_0, 0.0)
        
        O_bf16_1 = (acc_O_1 / l_final[:, None]).to(tl.bfloat16)
        O_out_1 = tl.where(valid_l[:, None], O_bf16_1, 0.0)
        
        out_q_row = q_start + tl.arange(0, 128)
        d_col_0 = tl.arange(0, 64)[None, :]
        d_col_1 = 64 + tl.arange(0, 64)[None, :]
        valid_q = out_q_row < S_len
        
        out_O_0 = O_ptr + pid_b * H_dim * S_len * D_dim + pid_h * S_len * D_dim + out_q_row[:, None] * D_dim + d_col_0
        out_O_1 = O_ptr + pid_b * H_dim * S_len * D_dim + pid_h * S_len * D_dim + out_q_row[:, None] * D_dim + d_col_1
        
        tl.store(out_O_0, O_out_0, mask=valid_q[:, None])
        tl.store(out_O_1, O_out_1, mask=valid_q[:, None])
        
        out_LSE = LSE_ptr + pid_b * H_dim * S_len + pid_h * S_len + out_q_row
        lse_val = m_final + tl.log(l_final)
        lse_val = tl.where(l_final > 0, lse_val, float('-inf'))
        tl.store(out_LSE, lse_val, mask=valid_q)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    Q_desc = TensorDescriptor.from_tensor(Q, [B, H, 128, 128])
    K_desc = TensorDescriptor.from_tensor(K, [B, H, 64, 128])
    V_desc = TensorDescriptor.from_tensor(V, [B, H, 64, 128])
    
    num_tiles = triton.cdiv(S, 128) * B * H
    grid = (min(132, num_tiles),)
    scale = 1.0 / math.sqrt(D)
    
    _mha_fwd_persistent[grid](
        Q_desc, K_desc, V_desc, O, LSE,
        S, scale,
        BLOCK_M=128, BLOCK_N=64, NUM_SMS=132,
        num_warps=8, num_stages=4
    )