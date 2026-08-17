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
    pid_m = tl.program_id(0)
    pid_d = tl.program_id(1)
    bh = tl.program_id(2)
    
    q_start = pid_m * BLOCK_M
    row_offset = bh * S + q_start
    
    Q_0 = Q_desc.load([row_offset, 0])
    Q_1 = Q_desc.load([row_offset, 64])
    
    O_acc_0 = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    O_acc_1 = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    local_max = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    local_sum = tl.full((BLOCK_M,), 0.0, dtype=tl.float32)
    
    row_idx = tl.arange(0, BLOCK_M)
    valid_rows = (q_start + row_idx) < S
    
    for j in range(0, S, BLOCK_N):
        col_offset = bh * S + j
        
        K_0 = K_desc.load([col_offset, 0])
        K_1 = K_desc.load([col_offset, 64])
        V_0 = V_desc.load([col_offset, 0])
        V_1 = V_desc.load([col_offset, 64])
        
        P = tl.dot(Q_0, K_0.T)
        P += tl.dot(Q_1, K_1.T)
        
        col_idx = tl.arange(0, BLOCK_N)
        valid = (j + col_idx) < S
        P = tl.where(valid, P * scale, -float('inf'))
        
        m_prev = local_max
        local_max = tl.maximum(m_prev, tl.max(P, axis=1))
        
        exp_scale = tl.exp(m_prev - local_max)
        O_acc_0 = O_acc_0 * exp_scale[:, None]
        O_acc_1 = O_acc_1 * exp_scale[:, None]
        
        P = P - local_max[:, None]
        P_exp = tl.exp(P)
        local_sum = local_sum * exp_scale + tl.sum(P_exp, axis=1)
        
        O_acc_0 += tl.dot(P_exp.to(tl.bfloat16), V_0.to(tl.bfloat16))
        O_acc_1 += tl.dot(P_exp.to(tl.bfloat16), V_1.to(tl.bfloat16))
    
    O_0 = O_acc_0 / local_sum[:, None]
    O_1 = O_acc_1 / local_sum[:, None]
    
    if pid_d == 0:
        O_desc.store([row_offset, 0], O_0.to(tl.bfloat16))
    else:
        O_desc.store([row_offset, 64], O_1.to(tl.bfloat16))
    
    if pid_d == 0:
        lse = tl.where(valid_rows, local_max + tl.math.log(local_sum), 0.0)
        lse_offsets = bh * S + q_start + row_idx
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
    
    Q_desc = TensorDescriptor.from_tensor(Q_flat, [128, 64])
    K_desc = TensorDescriptor.from_tensor(K_flat, [128, 64])
    V_desc = TensorDescriptor.from_tensor(V_flat, [128, 64])
    O_desc = TensorDescriptor.from_tensor(O_flat, [128, 64])
    
    num_m_tiles = triton.cdiv(S, 128)
    grid = (num_m_tiles, 2, B * H)
    
    _attention_kernel[grid](Q_desc, K_desc, V_desc, O_desc, 
                           LSE, S, scale, 
                           BLOCK_M=128, BLOCK_N=128, BLOCK_D=64,
                           num_warps=8, num_stages=4)