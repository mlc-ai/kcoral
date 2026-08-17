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
    
    q_offset = b_h * S_len + start_n
    Q = q_desc.load([q_offset, 0])
    
    F = tl.full((BLOCK_Q,), -float('inf'), dtype=tl.float32)
    P_sum = tl.zeros((BLOCK_Q,), dtype=tl.float32)
    O_acc = tl.zeros((BLOCK_Q, 128), dtype=tl.float32)
    
    for k_idx in range(0, S_len, BLOCK_K):
        k_offset = b_h * S_len + k_idx
        
        K = k_desc.load([k_offset, 0])
        V = v_desc.load([k_offset, 0])
        
        S = tl.dot(Q, K.T) * scale
        
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
        
        O_acc = O_acc * exp_F_diff[:, None] + tl.dot(P, V)
    
    inv_P_sum = 1.0 / P_sum
    O_out = (O_acc * inv_P_sum[:, None]).to(tl.bfloat16)
    
    o_offset = b_h * S_len + start_n
    o_desc.store([o_offset, 0], O_out)
    
    LSE = F + tl.log(P_sum)                  
    tl.store(LSE_ptr + b_h * S_len + global_q_row, LSE, mask=(global_q_row < S_len))


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
    
    q_desc = TensorDescriptor.from_tensor(Q_2d, [BLOCK_Q, 128])
    k_desc = TensorDescriptor.from_tensor(K_2d, [BLOCK_K, 128])
    v_desc = TensorDescriptor.from_tensor(V_2d, [BLOCK_K, 128])
    o_desc = TensorDescriptor.from_tensor(O_2d, [BLOCK_Q, 128])
    
    grid = (triton.cdiv(S, BLOCK_Q), B * H)
    _attention_kernel[grid](
        q_desc, k_desc, v_desc, o_desc, LSE,
        S, scale,
        BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K,
        num_warps=8, num_stages=3,
    )