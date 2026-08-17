import torch
import triton
import triton.language as tl
import math
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _mha_fwd(
    Q_desc, K_desc, V_desc, O_ptr, LSE_ptr,
    S_len, D_dim, H_dim, B_dim, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    assert D_dim == 128, "D must be 128"
    
    pid_b = tl.program_id(2)
    pid_h = tl.program_id(1)
    pid_q = tl.program_id(0)
    
    q_start = pid_q * BLOCK_M
    base_offset = pid_b * H_dim * S_len * D_dim + pid_h * S_len * D_dim
    b_h_offset = pid_b * H_dim * S_len + pid_h * S_len
    
    Q = Q_desc.load([pid_b, pid_h, q_start, 0])
    Q = tl.reshape(Q, [BLOCK_M, 128])
    
    O_acc = tl.zeros((BLOCK_M, 128), dtype=tl.float32)
    m_prev = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)
    l_prev = tl.full((BLOCK_M,), 0.0, dtype=tl.float32)
    
    q_row = q_start + tl.arange(0, BLOCK_M)
    d_col = tl.arange(0, 128)[None, :]
    
    num_kv_blocks = tl.cdiv(S_len, BLOCK_N)
    
    for k_step in range(num_kv_blocks):
        k_start = k_step * BLOCK_N
        
        K = K_desc.load([pid_b, pid_h, k_start, 0])
        K = tl.reshape(K, [BLOCK_N, 128])
        
        V = V_desc.load([pid_b, pid_h, k_start, 0])
        V = tl.reshape(V, [BLOCK_N, 128])
        
        S = tl.dot(Q, K.T, acc=None)
        S = S * scale
        
        k_row = k_start + tl.arange(0, BLOCK_N)
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
        
        O_acc = O_acc * scale_factor[:, None]
        O_acc = tl.dot(P, V, acc=O_acc)
        
        m_prev = m_new
        l_prev = l_new
        
    valid_l = l_prev > 0
    O_bf16 = (O_acc / l_prev[:, None]).to(tl.bfloat16)
    O_out = tl.where(valid_l[:, None], O_bf16, 0.0)
    
    valid_q = q_row < S_len
    tl.store(O_ptr + base_offset + q_row[:, None] * D_dim + d_col,
             O_out, mask=valid_q[:, None])
             
    lse_val = m_prev + tl.log(l_prev)
    lse_val = tl.where(valid_l, lse_val, float('-inf'))
    tl.store(LSE_ptr + b_h_offset + q_row, lse_val, mask=valid_q)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    Q_desc = TensorDescriptor.from_tensor(Q, [1, 1, 64, 128])
    K_desc = TensorDescriptor.from_tensor(K, [1, 1, 64, 128])
    V_desc = TensorDescriptor.from_tensor(V, [1, 1, 64, 128])
    
    grid = (triton.cdiv(S, 64), H, B)
    scale = 1.0 / math.sqrt(D)
    
    _mha_fwd[grid](
        Q_desc, K_desc, V_desc, O, LSE,
        S, D, H, B, scale,
        BLOCK_M=64, BLOCK_N=64,
        num_warps=4, num_stages=3
    )