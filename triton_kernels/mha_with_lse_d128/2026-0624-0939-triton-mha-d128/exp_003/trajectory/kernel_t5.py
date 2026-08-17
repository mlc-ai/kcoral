import torch
import triton
import triton.language as tl


@triton.jit
def _attention_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    S_len, D, scale,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr,
):
    start_n = tl.program_id(0) * BLOCK_Q
    b_h = tl.program_id(1)
    
    q_row_local = tl.arange(0, BLOCK_Q)
    k_col_local = tl.arange(0, BLOCK_K)
    cols_0 = tl.arange(0, 64)
    cols_1 = tl.arange(64, 128)
    global_q_row = start_n + q_row_local
    
    base_offset = b_h * S_len * D
    Q_base = Q_ptr + base_offset + start_n * D
    O_base = O_ptr + base_offset + start_n * D
    LSE_base = LSE_ptr + b_h * S_len + start_n
    
    Q_0 = tl.load(Q_base + q_row_local[:, None] * D + cols_0[None, :], 
                  mask=(global_q_row[:, None] < S_len), other=0.0).to(tl.float32)
    Q_1 = tl.load(Q_base + q_row_local[:, None] * D + cols_1[None, :], 
                  mask=(global_q_row[:, None] < S_len), other=0.0).to(tl.float32)
    
    F = tl.full((BLOCK_Q,), -float('inf'), dtype=tl.float32)
    P_sum = tl.zeros((BLOCK_Q,), dtype=tl.float32)
    O_acc_0 = tl.zeros((BLOCK_Q, 64), dtype=tl.float32)
    O_acc_1 = tl.zeros((BLOCK_Q, 64), dtype=tl.float32)
    
    for k_idx in range(0, S_len, BLOCK_K):
        k_col_global = k_idx + k_col_local
        
        K_base = K_ptr + base_offset + k_idx * D
        V_base = V_ptr + base_offset + k_idx * D
        
        K_0 = tl.load(K_base + k_col_local[:, None] * D + cols_0[None, :], 
                      mask=(k_col_global[:, None] < S_len), other=0.0).to(tl.float32)
        K_1 = tl.load(K_base + k_col_local[:, None] * D + cols_1[None, :], 
                      mask=(k_col_global[:, None] < S_len), other=0.0).to(tl.float32)
        
        V_0 = tl.load(V_base + k_col_local[:, None] * D + cols_0[None, :], 
                      mask=(k_col_global[:, None] < S_len), other=0.0).to(tl.float32)
        V_1 = tl.load(V_base + k_col_local[:, None] * D + cols_1[None, :], 
                      mask=(k_col_global[:, None] < S_len), other=0.0).to(tl.float32)
        
        S = tl.zeros((BLOCK_Q, BLOCK_K), dtype=tl.float32)
        S = tl.dot(Q_0, K_0.T, S)
        S = tl.dot(Q_1, K_1.T, S)
        
        S *= scale
        
        key_mask = (k_idx + k_col_local < S_len)[None, :]
        S = tl.where(key_mask, S, -float('inf'))
        
        F_old = F
        col_max = tl.max(S, axis=1)
        F_new = tl.maximum(F_old, col_max)
        exp_F_diff = tl.exp(F_old - F_new)
        P = tl.exp(S - F_new[:, None])
        row_sum = tl.sum(P, axis=1)
        P_sum = P_sum * exp_F_diff + row_sum
        F = F_new
        
        O_acc_0 = O_acc_0 * exp_F_diff[:, None] + tl.dot(P, V_0)
        O_acc_1 = O_acc_1 * exp_F_diff[:, None] + tl.dot(P, V_1)
    
    inv_P_sum = 1.0 / P_sum
    O_0_out = (O_acc_0 * inv_P_sum[:, None]).to(tl.bfloat16)
    O_1_out = (O_acc_1 * inv_P_sum[:, None]).to(tl.bfloat16)
    
    tl.store(O_base + q_row_local[:, None] * D + cols_0[None, :], O_0_out, 
             mask=(global_q_row[:, None] < S_len))
    tl.store(O_base + q_row_local[:, None] * D + cols_1[None, :], O_1_out, 
             mask=(global_q_row[:, None] < S_len))
    
    LSE = F + tl.log(P_sum)                  
    tl.store(LSE_base + q_row_local, LSE, mask=(global_q_row < S_len))


BLOCK_Q = 64
BLOCK_K = 64

def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    scale = 1.0 / (D ** 0.5)
    
    grid = (triton.cdiv(S, BLOCK_Q), B * H)
    _attention_kernel[grid](
        Q, K, V, O, LSE,
        S, D, scale,
        BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K,
        num_warps=8, num_stages=3,
    )