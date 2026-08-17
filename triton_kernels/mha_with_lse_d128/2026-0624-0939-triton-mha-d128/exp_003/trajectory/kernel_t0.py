import torch
import triton
import triton.language as tl


@triton.jit
def _attention_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    S_len, D, H, scale,
    BLOCK_S: tl.constexpr,
):
    start_n = tl.program_id(0) * BLOCK_S
    b_h = tl.program_id(1)
    
    q_row_local = tl.arange(0, BLOCK_S)
    global_q_row = start_n + q_row_local
    cols_0 = tl.arange(0, 64)
    cols_1 = tl.arange(64, 128)
    
    batch_head_offset = b_h * S_len * D
    Q_ptr += batch_head_offset
    K_ptr += batch_head_offset
    V_ptr += batch_head_offset
    O_ptr += batch_head_offset
    
    Q_0 = tl.load(Q_ptr + q_row_local[:, None] * D + cols_0[None, :], 
                  mask=(global_q_row[:, None] < S_len), other=0.0)
    Q_1 = tl.load(Q_ptr + q_row_local[:, None] * D + cols_1[None, :], 
                  mask=(global_q_row[:, None] < S_len), other=0.0)
    
    F = -float('inf')
    P_sum = 0.0
    O_acc_0 = tl.zeros((BLOCK_S, 64), dtype=tl.float32)
    O_acc_1 = tl.zeros((BLOCK_S, 64), dtype=tl.float32)
    
    for k_idx in range(0, S_len, BLOCK_S):
        k_col_local = tl.arange(0, BLOCK_S)
        k_col_global = k_idx + k_col_local
        
        K_0 = tl.load(K_ptr + k_col_local[:, None] * D + cols_0[None, :], 
                      mask=(k_col_global[:, None] < S_len), other=0.0)
        K_1 = tl.load(K_ptr + k_col_local[:, None] * D + cols_1[None, :], 
                      mask=(k_col_global[:, None] < S_len), other=0.0)
        
        V_0 = tl.load(V_ptr + k_col_local[:, None] * D + cols_0[None, :], 
                      mask=(k_col_global[:, None] < S_len), other=0.0)
        V_1 = tl.load(V_ptr + k_col_local[:, None] * D + cols_1[None, :], 
                      mask=(k_col_global[:, None] < S_len), other=0.0)
        
        S = tl.zeros((BLOCK_S, BLOCK_S), dtype=tl.float32)
        S = tl.dot(Q_0, K_0.T, S)
        S = tl.dot(Q_1, K_1.T, S)
        S /= scale
        
        key_mask = (k_col_global < S_len)[None, :]
        S = tl.where(key_mask, S, -float('inf'))
        
        F_new = tl.maximum(F, tl.max(S, dim=-1, keep_dims=True))
        exp_F_diff = tl.exp(F - F_new)
        P = tl.exp(S - F_new)
        P_sum = P_sum * exp_F_diff + tl.sum(P, dim=-1, keep_dims=True)
        F = F_new
        
        O_acc_0 = O_acc_0 * exp_F_diff + tl.dot(P, V_0)
        O_acc_1 = O_acc_1 * exp_F_diff + tl.dot(P, V_1)
    
    O_0 = O_acc_0 / P_sum
    O_1 = O_acc_1 / P_sum
    LSE = F + tl.log(P_sum)
    
    out_ptr_0 = O_ptr + q_row_local[:, None] * D + cols_0[None, :]
    tl.store(out_ptr_0, O_0.to(tl.bfloat16), mask=(global_q_row[:, None] < S_len))
    
    out_ptr_1 = O_ptr + q_row_local[:, None] * D + cols_1[None, :]
    tl.store(out_ptr_1, O_1.to(tl.bfloat16), mask=(global_q_row[:, None] < S_len))
    
    lse_val = tl.reshape(LSE, (BLOCK_S,))
    tl.store(LSE_ptr + b_h * S_len + global_q_row, lse_val, mask=(global_q_row < S_len))


BLOCK_S = 128

def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    scale = 1.0 / (D ** 0.5)
    grid = (triton.cdiv(S, BLOCK_S), B * H)
    _attention_kernel[grid](
        Q, K, V, O, LSE,
        S, D, H, scale,
        BLOCK_S=BLOCK_S,
        num_warps=8,
        num_stages=3,
    )