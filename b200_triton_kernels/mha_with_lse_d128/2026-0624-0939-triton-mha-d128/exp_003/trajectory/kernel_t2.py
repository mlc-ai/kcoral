import torch
import triton
import triton.language as tl


@triton.jit
def _attention_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    S_len, D,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr,
):
    start_n = tl.program_id(0) * BLOCK_Q
    b_h = tl.program_id(1)
    
    q_row_local = tl.arange(0, BLOCK_Q)
    k_col_local = tl.arange(0, BLOCK_K)
    col_local = tl.arange(0, D)
    global_q_row = start_n + q_row_local
    
    base_ptr = Q_ptr + b_h * S_len * D + start_n * D
    off_q = q_row_local[:, None] * D + col_local[None, :]
    Q = tl.load(base_ptr + off_q, mask=(global_q_row[:, None] < S_len), other=0.0)
    
    F = tl.full((BLOCK_Q,), -float('inf'), dtype=tl.float32)
    P_sum = tl.zeros((BLOCK_Q,), dtype=tl.float32)
    O_acc = tl.zeros((BLOCK_Q, D), dtype=tl.float32)
    
    scale = tl.rsqrt(128.0)
    
    for k_idx in range(0, S_len, BLOCK_K):
        k_base_ptr = K_ptr + b_h * S_len * D + k_idx * D
        k_col_global = k_idx + k_col_local
        k_offs = k_col_local[:, None] * D + col_local[None, :]
        
        K = tl.load(k_base_ptr + k_offs, mask=(k_col_global[:, None] < S_len), other=0.0)
        
        v_base_ptr = V_ptr + b_h * S_len * D + k_idx * D
        V = tl.load(v_base_ptr + k_offs, mask=(k_col_global[:, None] < S_len), other=0.0)
        
        S = tl.zeros((BLOCK_Q, BLOCK_K), dtype=tl.float32)
        S = tl.dot(Q, K.T, S)
        
        S = S * scale
        
        key_mask = (k_idx + k_col_local < S_len)[None, :]
        S = tl.where(key_mask, S, -float('inf'))
        
        F_old = F
        F_new = tl.maximum(F_old, tl.reduce_max(S, axis=1))
        exp_F_diff = tl.exp(F_old - F_new)
        P = tl.exp(S - F_new)
        P_sum = P_sum * exp_F_diff + tl.reduce_sum(P, axis=1)
        F = F_new
        
        O_acc = O_acc * exp_F_diff[:, None] + tl.dot(P, V)
    
    O_out = (O_acc / P_sum[:, None]).to(tl.bfloat16)
    
    out_base_ptr = O_ptr + b_h * S_len * D + start_n * D
    offs = q_row_local[:, None] * D + col_local[None, :]
    tl.store(out_base_ptr + offs, O_out, mask=(global_q_row[:, None] < S_len))
    
    LSE = F + tl.log(P_sum)                  
    tl.store(LSE_ptr + b_h * S_len + global_q_row, LSE, mask=(global_q_row < S_len))


BLOCK_Q = 64
BLOCK_K = 64

def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    grid = (triton.cdiv(S, BLOCK_Q), B * H)
    _attention_kernel[grid](
        Q, K, V, O, LSE,
        S, D,
        BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K,
        num_warps=8, num_stages=3,
    )