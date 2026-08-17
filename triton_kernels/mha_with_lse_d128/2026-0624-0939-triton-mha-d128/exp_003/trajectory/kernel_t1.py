import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _attention_kernel(
    q_desc, k_desc, v_desc, o_desc, LSE_ptr,
    S_len, scale,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr,
):
    start_n = tl.program_id(0) * BLOCK_Q
    b_h = tl.program_id(1)
    
    q_row_local = tl.arange(0, BLOCK_Q)
    k_col_local = tl.arange(0, BLOCK_K)
    global_q_row = start_n + q_row_local
    
    offset = b_h * S_len + start_n
    Q_0 = q_desc.load([offset, 0])       
    Q_1 = q_desc.load([offset, 64])      
    
    F = tl.full((BLOCK_Q, 1), -float('inf'), dtype=tl.float32)
    P_sum = tl.zeros((BLOCK_Q, 1), dtype=tl.float32)
    O_acc_0 = tl.zeros((BLOCK_Q, 64), dtype=tl.float32)
    O_acc_1 = tl.zeros((BLOCK_Q, 64), dtype=tl.float32)
    
    for k_idx in range(0, S_len, BLOCK_K):
        k_offset = b_h * S_len + k_idx
        
        K_0 = k_desc.load([k_offset, 0])  
        K_1 = k_desc.load([k_offset, 64]) 
        V_0 = v_desc.load([k_offset, 0])  
        V_1 = v_desc.load([k_offset, 64]) 
        
        S = tl.zeros((BLOCK_Q, BLOCK_K), dtype=tl.float32)
        S = tl.dot(Q_0, K_0.T, S)
        S = tl.dot(Q_1, K_1.T, S)
        
        S *= scale
        
        key_mask = (k_idx + k_col_local < S_len)[None, :]
        S = tl.where(key_mask, S, -float('inf'))
        
        F_old = F
        F_new = tl.maximum(F_old, tl.max(S, dim=-1, keep_dims=True))
        exp_F_diff = tl.exp(F_old - F_new)
        P = tl.exp(S - F_new)
        P_sum = P_sum * exp_F_diff + tl.sum(P, dim=-1, keep_dims=True)
        F = F_new
        
        O_acc_0 = O_acc_0 * exp_F_diff + tl.dot(P, V_0)
        O_acc_1 = O_acc_1 * exp_F_diff + tl.dot(P, V_1)
    
    O_0_out = (O_acc_0 / P_sum).to(tl.bfloat16)
    O_1_out = (O_acc_1 / P_sum).to(tl.bfloat16)
    
    o_desc.store([offset, 0], O_0_out)
    o_desc.store([offset, 64], O_1_out)
    
    LSE = F + tl.log(P_sum)                  
    lse_val = tl.reshape(LSE, (BLOCK_Q,))            
    tl.store(LSE_ptr + b_h * S_len + global_q_row, lse_val, mask=(global_q_row < S_len))


BLOCK_Q = 64
BLOCK_K = 64

def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    scale = 1.0 / (D ** 0.5)
    
    Q_2d = Q.view(-1, D)
    K_2d = K.view(-1, D)
    V_2d = V.view(-1, D)
    O_2d = O.view(-1, D)
    
    q_desc = TensorDescriptor.from_tensor(Q_2d, [BLOCK_Q, 64])
    k_desc = TensorDescriptor.from_tensor(K_2d, [BLOCK_K, 64])
    v_desc = TensorDescriptor.from_tensor(V_2d, [BLOCK_K, 64])
    o_desc = TensorDescriptor.from_tensor(O_2d, [BLOCK_Q, 64])
    
    grid = (triton.cdiv(S, BLOCK_Q), B * H)
    _attention_kernel[grid](
        q_desc, k_desc, v_desc, o_desc, LSE,
        S, scale,
        BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K,
        num_warps=8, num_stages=3,
    )