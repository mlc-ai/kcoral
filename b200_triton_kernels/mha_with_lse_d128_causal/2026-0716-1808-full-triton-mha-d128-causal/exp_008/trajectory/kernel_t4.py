import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _attention_kernel(
    Q_desc, K_desc, V_desc, O_ptr, LSE_ptr,
    S, B, H, D,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr):
    
    scale = 1.0 / tl.sqrt(tl.float32(D))
    
    pid_m = tl.program_id(0)
    row_start = pid_m * BLOCK_M
    b_h = tl.program_id(1)
    
    q_offset0 = b_h * S + row_start
    q_offset1 = b_h * S + row_start + 64
    
    cur_Q0_0 = Q_desc.load([q_offset0, 0])
    cur_Q0_1 = Q_desc.load([q_offset0, BLOCK_D])
    cur_Q1_0 = Q_desc.load([q_offset1, 0])
    cur_Q1_1 = Q_desc.load([q_offset1, BLOCK_D])
    
    O0_0 = tl.zeros((64, BLOCK_D), dtype=tl.float32)
    O0_1 = tl.zeros((64, BLOCK_D), dtype=tl.float32)
    O1_0 = tl.zeros((64, BLOCK_D), dtype=tl.float32)
    O1_1 = tl.zeros((64, BLOCK_D), dtype=tl.float32)
    
    m0 = tl.full((64,), float('-inf'), dtype=tl.float32)
    m1 = tl.full((64,), float('-inf'), dtype=tl.float32)
    l0 = tl.zeros((64,), dtype=tl.float32)
    l1 = tl.zeros((64,), dtype=tl.float32)
    
    num_kv_blocks = (S + BLOCK_N - 1) // BLOCK_N
    max_j = min(num_kv_blocks, pid_m * 2 + 2)
    
    if max_j > 0:
        kv_offset = b_h * S + 0 * BLOCK_N
        cur_K_0 = K_desc.load([kv_offset, 0])
        cur_K_1 = K_desc.load([kv_offset, BLOCK_D])
        cur_V_0 = V_desc.load([kv_offset, 0])
        cur_V_1 = V_desc.load([kv_offset, BLOCK_D])
    
    for j in range(max_j):
        acc_QK0 = tl.dot(cur_Q0_0, cur_K_0.T) + tl.dot(cur_Q0_1, cur_K_1.T)
        acc_QK1 = tl.dot(cur_Q1_0, cur_K_0.T) + tl.dot(cur_Q1_1, cur_K_1.T)
        
        acc_QK0 *= scale
        acc_QK1 *= scale
        
        global_q_idx0 = (row_start + tl.arange(0, 64))[:, None]
        global_q_idx1 = (row_start + 64 + tl.arange(0, 64))[:, None]
        global_k_idx = (j * BLOCK_N + tl.arange(0, BLOCK_N))[None, :]
        
        valid0 = (global_k_idx <= global_q_idx0) & (global_k_idx < S)
        valid1 = (global_k_idx <= global_q_idx1) & (global_k_idx < S)
        
        acc_QK0 = tl.where(valid0, acc_QK0, float('-inf'))
        acc_QK1 = tl.where(valid1, acc_QK1, float('-inf'))
        
        M_0 = tl.max(acc_QK0, axis=1)
        M_1 = tl.max(acc_QK1, axis=1)
        
        m_prev0 = m0
        m0 = tl.maximum(m0, M_0)
        m_prev1 = m1
        m1 = tl.maximum(m1, M_1)
        
        P_0 = tl.exp(acc_QK0 - m0)
        P_1 = tl.exp(acc_QK1 - m1)
        
        l0 = l0 * tl.exp(m_prev0 - m0) + tl.sum(P_0, axis=1)
        l1 = l1 * tl.exp(m_prev1 - m1) + tl.sum(P_1, axis=1)
        
        if j + 1 < max_j:
            next_kv_offset = b_h * S + (j + 1) * BLOCK_N
            cur_K_0 = K_desc.load([next_kv_offset, 0])
            cur_K_1 = K_desc.load([next_kv_offset, BLOCK_D])
            cur_V_0 = V_desc.load([next_kv_offset, 0])
            cur_V_1 = V_desc.load([next_kv_offset, BLOCK_D])
            
        O0_0 = O0_0 * tl.exp(m_prev0 - m0)[:, None]
        O0_1 = O0_1 * tl.exp(m_prev0 - m0)[:, None]
        O1_0 = O1_0 * tl.exp(m_prev1 - m1)[:, None]
        O1_1 = O1_1 * tl.exp(m_prev1 - m1)[:, None]
        
        P_0 = P_0.to(tl.bfloat16)
        P_1 = P_1.to(tl.bfloat16)
        
        O0_0 += tl.dot(P_0, cur_V_0)
        O0_1 += tl.dot(P_0, cur_V_1)
        O1_0 += tl.dot(P_1, cur_V_0)
        O1_1 += tl.dot(P_1, cur_V_1)
        
    l0 = tl.where(l0 > 0, l0, 1.0)
    l1 = tl.where(l1 > 0, l1, 1.0)
    
    O0_0 = O0_0 / l0[:, None]
    O0_1 = O0_1 / l0[:, None]
    O1_0 = O1_0 / l1[:, None]
    O1_1 = O1_1 / l1[:, None]
    
    row_idx0 = row_start + tl.arange(0, 64)
    row_idx1 = row_start + 64 + tl.arange(0, 64)
    
    LSE_val0 = m0 + tl.log(l0)
    LSE_val1 = m1 + tl.log(l1)
    
    col = tl.arange(0, BLOCK_D)
    
    out_ptr0_0 = O_ptr + (b_h * S + row_idx0[:, None]) * D + col[None, :]
    tl.store(
        out_ptr0_0,
        O0_0.to(tl.bfloat16),
        mask=(row_idx0[:, None] < S),
    )
    
    out_ptr0_1 = O_ptr + (b_h * S + row_idx0[:, None]) * D + (col[None, :] + BLOCK_D)
    tl.store(
        out_ptr0_1,
        O0_1.to(tl.bfloat16),
        mask=(row_idx0[:, None] < S),
    )
    
    out_ptr1_0 = O_ptr + (b_h * S + row_idx1[:, None]) * D + col[None, :]
    tl.store(
        out_ptr1_0,
        O1_0.to(tl.bfloat16),
        mask=(row_idx1[:, None] < S),
    )
    
    out_ptr1_1 = O_ptr + (b_h * S + row_idx1[:, None]) * D + (col[None, :] + BLOCK_D)
    tl.store(
        out_ptr1_1,
        O1_1.to(tl.bfloat16),
        mask=(row_idx1[:, None] < S),
    )
    
    out_ptr_lse0 = LSE_ptr + b_h * S + row_idx0
    tl.store(
        out_ptr_lse0,
        LSE_val0,
        mask=(row_idx0 < S),
    )
    
    out_ptr_lse1 = LSE_ptr + b_h * S + row_idx1
    tl.store(
        out_ptr_lse1,
        LSE_val1,
        mask=(row_idx1 < S),
    )


def run(Q, K, V, O, LSE):
    """Compute causal multi-head attention with Log-Sum-Exp."""
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    
    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_D = 64
    
    M = B * H * S
    
    Q_2d = Q.reshape(M, D).contiguous()
    K_2d = K.reshape(M, D).contiguous()
    V_2d = V.reshape(M, D).contiguous()
    
    Q_desc = TensorDescriptor.from_tensor(Q_2d, [64, BLOCK_D])
    K_desc = TensorDescriptor.from_tensor(K_2d, [BLOCK_N, BLOCK_D])
    V_desc = TensorDescriptor.from_tensor(V_2d, [BLOCK_N, BLOCK_D])
    
    grid = (triton.cdiv(S, BLOCK_M), B * H)
    
    _attention_kernel[grid](
        Q_desc, K_desc, V_desc, O, LSE,
        S, B, H, D,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=4, num_stages=2,
    )