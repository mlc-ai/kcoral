import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
import math


@triton.jit
def _attention_kernel(
    Q_desc, K_desc, V_desc, O_desc,
    LSE_ptr, S, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    NUM_SMS = 132
    phase_id = tl.program_id(0) % 2
    pid_m = tl.program_id(0) // 2
    bh = tl.program_id(1)
    
    q_start = pid_m * BLOCK_M
    row_offset = bh * S + q_start
    
    if phase_id == 0:
        Q = Q_desc.load([row_offset, 0])
        
        O_acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
        local_max = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
        local_sum = tl.full((BLOCK_M,), 0.0, dtype=tl.float32)
        
        for j in range(0, S, BLOCK_N):
            col_offset = bh * S + j
            
            K = K_desc.load([col_offset, 0])
            V = V_desc.load([col_offset, 0])
            
            P = tl.dot(Q, K.T)
            P = P * scale
            
            valid = (j + tl.arange(0, BLOCK_N)) < S
            P = P.where(valid, -float('inf'))
            
            m_prev = local_max
            local_max = tl.maximum(m_prev, tl.max(P, axis=1))
            
            exp_scale = tl.exp(m_prev - local_max)
            O_acc = O_acc * exp_scale[:, None]
            
            P = P - local_max[:, None]
            P_exp = tl.exp(P)
            local_sum = local_sum * exp_scale + tl.sum(P_exp, axis=1)
            
            O_acc += tl.dot(P_exp.to(tl.bfloat16), V)
        
        O = O_acc / local_sum[:, None]
        O_desc.store([row_offset, 0], O.to(tl.bfloat16))
        
    else:
        Q = Q_desc.load([row_offset, 0])
        
        local_max = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
        local_sum = tl.full((BLOCK_M,), 0.0, dtype=tl.float32)
        
        for j in range(0, S, BLOCK_N):
            col_offset = bh * S + j
            
            K = K_desc.load([col_offset, 0])
            
            P = tl.dot(Q, K.T)
            P = P * scale
            
            valid = (j + tl.arange(0, BLOCK_N)) < S
            P = P.where(valid, -float('inf'))
            
            m_prev = local_max
            local_max = tl.maximum(m_prev, tl.max(P, axis=1))
            
            exp_scale = tl.exp(m_prev - local_max)
            local_sum = local_sum * exp_scale
            
            P = P - local_max[:, None]
            P_exp = tl.exp(P)
            local_sum = local_sum + tl.sum(P_exp, axis=1)
        
        valid_rows = (q_start + tl.arange(0, BLOCK_M)) < S
        lse = tl.where(valid_rows, local_max + tl.math.log(local_sum), 0.0)
        
        lse_offsets = bh * S + q_start + tl.arange(0, BLOCK_M)
        tl.store(LSE_ptr + lse_offsets, lse, mask=valid_rows)


def run(Q, K, V, O, LSE):
    """Compute Attention O and LSE into preallocated CUDA tensors."""
    torch.cuda.set_device(Q.device)
    
    B = Q.shape[0]
    H = Q.shape[1]
    S = Q.shape[2]
    D = Q.shape[3]
    
    scale = 1.0 / math.sqrt(D)
    
    Q_flat = Q.reshape(B * H * S, D)
    K_flat = K.reshape(B * H * S, D)
    V_flat = V.reshape(B * H * S, D)
    O_flat = O.reshape(B * H * S, D)
    
    Q_desc = TensorDescriptor.from_tensor(Q_flat, [128, 128])
    K_desc = TensorDescriptor.from_tensor(K_flat, [128, 128])
    V_desc = TensorDescriptor.from_tensor(V_flat, [128, 128])
    O_desc = TensorDescriptor.from_tensor(O_flat, [128, 128])
    
    NUM_SMS = 132
    num_tiles = min(NUM_SMS, triton.cdiv(S, 128))
    
    grid = (2 * num_tiles, B * H)
    
    _attention_kernel[grid](Q_desc, K_desc, V_desc, O_desc, 
                           LSE, S, scale, 
                           BLOCK_M=128, BLOCK_N=128, BLOCK_D=128,
                           num_warps=8, num_stages=4)