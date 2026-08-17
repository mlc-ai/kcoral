import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _attention_kernel(
    Q_desc, K_desc, V_desc, O_desc, LSE_ptr,
    S, B, H, D,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr):
    
    scale = 1.0 / tl.sqrt(tl.float32(D))
    
    pid_m = tl.program_id(0)
    row_start = pid_m * BLOCK_M
    b_h = tl.program_id(1)
    
    q_offset = b_h * S + row_start
    Q0 = Q_desc.load([q_offset, 0])
    Q1 = Q_desc.load([q_offset, BLOCK_D])
    
    O0 = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    O1 = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    
    m = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)
    l = tl.zeros((BLOCK_M,), dtype=tl.float32)
    
    num_kv_blocks = (S + BLOCK_N - 1) // BLOCK_N
    
    max_j = min(num_kv_blocks, pid_m + 1)
    
    for j in range(max_j):
        kv_offset = b_h * S + j * BLOCK_N
        
        K0 = K_desc.load([kv_offset, 0])
        K1 = K_desc.load([kv_offset, BLOCK_D])
        
        V0 = V_desc.load([kv_offset, 0])
        V1 = V_desc.load([kv_offset, BLOCK_D])
        
        acc_QK = tl.dot(Q0, K0.T) + tl.dot(Q1, K1.T)
        
        acc_QK *= scale
        
        global_q_idx = (row_start + tl.arange(0, BLOCK_M))[:, None]
        global_k_idx = (j * BLOCK_N + tl.arange(0, BLOCK_N))[None, :]
        valid = (global_k_idx <= global_q_idx) & (global_k_idx < S)
        
        acc_QK = tl.where(valid, acc_QK, float('-inf'))
        
        M_j = tl.max(acc_QK, axis=1)
        m_prev = m
        m = tl.maximum(m, M_j)
        
        P = tl.exp(acc_QK - m)
        
        l = l * tl.exp(m_prev - m) + tl.sum(P, axis=1)
        
        rescale_O = tl.exp(m_prev - m)[:, None]
        O0 = O0 * rescale_O
        O1 = O1 * rescale_O
        
        P = P.to(tl.bfloat16)
        
        O0 += tl.dot(P, V0)
        O1 += tl.dot(P, V1)
        
    O0 = O0 / l[:, None]
    O1 = O1 / l[:, None]
    
    row_idx = row_start + tl.arange(0, BLOCK_M)
    LSE_val = m + tl.log(l)
    
    out_offset = b_h * S + row_idx
    O_desc.store([out_offset, 0], O0.to(tl.bfloat16))
    O_desc.store([out_offset, BLOCK_D], O1.to(tl.bfloat16))
    
    tl.store(
        LSE_ptr + b_h * S + row_idx,
        LSE_val,
        mask=(row_idx < S)
    )


def run(Q, K, V, O, LSE):
    """Compute causal multi-head attention with Log-Sum-Exp."""
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = 64
    
    M = B * H * S
    
    Q_2d = Q.reshape(M, D).contiguous()
    K_2d = K.reshape(M, D).contiguous()
    V_2d = V.reshape(M, D).contiguous()
    O_2d = O.reshape(M, D).contiguous()
    
    Q_desc = TensorDescriptor.from_tensor(Q_2d, [BLOCK_M, BLOCK_D])
    K_desc = TensorDescriptor.from_tensor(K_2d, [BLOCK_N, BLOCK_D])
    V_desc = TensorDescriptor.from_tensor(V_2d, [BLOCK_N, BLOCK_D])
    O_desc = TensorDescriptor.from_tensor(O_2d, [BLOCK_M, BLOCK_D])
    
    grid = (triton.cdiv(S, BLOCK_M), B * H)
    
    _attention_kernel[grid](
        Q_desc, K_desc, V_desc, O_desc, LSE,
        S, B, H, D,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=4, num_stages=2,
    )