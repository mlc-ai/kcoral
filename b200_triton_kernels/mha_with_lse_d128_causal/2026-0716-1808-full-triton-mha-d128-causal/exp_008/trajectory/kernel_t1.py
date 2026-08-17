import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _attention_kernel(
    Q_desc,
    K_desc,
    V_desc,
    O_ptr,
    LSE_ptr,
    S,
    B,
    H,
    D,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    scale = 1.0 / tl.sqrt(tl.float32(D))
    
    pid_m = tl.program_id(0)
    row_start = pid_m * BLOCK_M
    b_h = tl.program_id(1)
    
    q_offset = b_h * S + row_start
    
    Q0 = Q_desc.load([q_offset, 0])
    Q1 = Q_desc.load([q_offset, 64])
    
    col = tl.arange(0, BLOCK_D)
    row_idx = row_start + tl.arange(0, BLOCK_M)
    
    O0 = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    O1 = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    
    m = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)
    l = tl.zeros((BLOCK_M,), dtype=tl.float32)
    
    num_kv_blocks = (S + BLOCK_N - 1) // BLOCK_N
    max_j = min(pid_m, num_kv_blocks - 1) + 1
    
    for j in range(max_j):
        kv_offset = b_h * S + j * BLOCK_N
        
        K0 = K_desc.load([kv_offset, 0])
        K1 = K_desc.load([kv_offset, 64])
        
        V0 = V_desc.load([kv_offset, 0])
        V1 = V_desc.load([kv_offset, 64])
        
        S_mat = tl.dot(Q0, K0.T) + tl.dot(Q1, K1.T)
        S_mat = S_mat * scale
        
        global_q_idx = (row_start + tl.arange(0, BLOCK_M))[:, None]
        global_k_idx = (j * BLOCK_N + tl.arange(0, BLOCK_N))[None, :]
        valid = (global_k_idx <= global_q_idx) & (global_k_idx < S)
        
        S_mat = tl.where(valid, S_mat, float('-inf'))
        
        M_j = tl.max(S_mat, axis=1)
        m_prev = m
        m = tl.maximum(m, M_j)
        
        P = tl.exp(S_mat - m)
        
        l = l * tl.exp(m_prev - m) + tl.sum(P, axis=1)
        
        rescale = tl.exp(m_prev - m)[:, None]
        O0 = O0 * rescale + tl.dot(P, V0)
        O1 = O1 * rescale + tl.dot(P, V1)
        
    l = tl.where(l > 0, l, 1.0)
    
    O0 = O0 / l[:, None]
    O1 = O1 / l[:, None]
    
    LSE_flat = m + tl.log(l)
    
    tl.store(
        O_ptr + b_h * S * D + row_idx[:, None] * D + col[None, :],
        O0.to(tl.bfloat16),
        mask=(row_idx[:, None] < S),
    )
    
    tl.store(
        O_ptr + b_h * S * D + row_idx[:, None] * D + (col[None, :] + 64),
        O1.to(tl.bfloat16),
        mask=(row_idx[:, None] < S),
    )
    
    tl.store(
        LSE_ptr + b_h * S + row_idx,
        LSE_flat,
        mask=(row_idx < S),
    )


def run(Q, K, V, O, LSE):
    """Compute causal multi-head attention with Log-Sum-Exp."""
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = 64
    
    Q_2d = Q.reshape(B * H * S, D)
    K_2d = K.reshape(B * H * S, D)
    V_2d = V.reshape(B * H * S, D)
    
    Q_desc = TensorDescriptor.from_tensor(Q_2d, [BLOCK_M, BLOCK_D])
    K_desc = TensorDescriptor.from_tensor(K_2d, [BLOCK_N, BLOCK_D])
    V_desc = TensorDescriptor.from_tensor(V_2d, [BLOCK_N, BLOCK_D])
    
    grid = (triton.cdiv(S, BLOCK_M), B * H)
    
    _attention_kernel[grid](
        Q_desc, K_desc, V_desc, O, LSE,
        S, B, H, D,
        BLOCK_M, BLOCK_N, BLOCK_D,
        num_warps=4,
        num_stages=2,
    )