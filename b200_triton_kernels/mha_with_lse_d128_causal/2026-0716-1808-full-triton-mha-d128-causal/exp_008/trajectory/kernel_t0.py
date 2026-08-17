import torch
import triton
import triton.language as tl


@triton.jit
def _attention_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    LSE_ptr,
    S,
    B,
    H,
    D,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    scale = 1.0 / tl.sqrt(tl.float32(D))
    
    pid_m = tl.program_id(0)
    row_start = pid_m * BLOCK_M
    b_h = tl.program_id(1)
    
    row = tl.arange(0, BLOCK_M)
    col = tl.arange(0, D)
    row_idx = row_start + row
    
    Q_flat = tl.load(
        Q_ptr + b_h * S * D + (row_start + row[:, None]) * D + col[None, :],
        mask=(row_idx[:, None] < S),
        other=0.0,
    )
    
    O_flat = tl.zeros((BLOCK_M, D), dtype=tl.float32)
    m = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)
    l = tl.zeros((BLOCK_M,), dtype=tl.float32)
    
    num_kv_blocks = (S + BLOCK_N - 1) // BLOCK_N
    
    for j in range(min(pid_m, num_kv_blocks - 1) + 1):
        col_kv = tl.arange(0, BLOCK_N)
        
        K_flat = tl.load(
            K_ptr + b_h * S * D + (j * BLOCK_N + col_kv[:, None]) * D + col[None, :],
            mask=((j * BLOCK_N + col_kv)[:, None] < S),
            other=0.0,
        )
        
        V_flat = tl.load(
            V_ptr + b_h * S * D + (j * BLOCK_N + col_kv[:, None]) * D + col[None, :],
            mask=((j * BLOCK_N + col_kv)[:, None] < S),
            other=0.0,
        )
        
        S_unscaled = tl.dot(Q_flat, K_flat.T, tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32))
        S = S_unscaled * scale
        
        global_q_idx = (row_start + row)[:, None]
        global_k_idx = (j * BLOCK_N + col_kv)[None, :]
        valid = (global_k_idx <= global_q_idx) & (global_k_idx < S)
        
        S = tl.where(valid, S, float('-inf'))
        
        M_j = tl.max(S, axis=1)
        m_prev = m
        m = tl.maximum(m, M_j)
        
        P = tl.exp(S - m)
        
        l = l * tl.exp(m_prev - m) + tl.sum(P, axis=1)
        
        O_flat = O_flat * tl.exp(m_prev - m)[:, None] + tl.dot(P, V_flat)
        
    O_flat = O_flat / l[:, None]
    
    tl.store(
        O_ptr + b_h * S * D + (row_start + row[:, None]) * D + col[None, :],
        O_flat,
        mask=(row_idx[:, None] < S),
    )
    
    LSE_flat = m + tl.log(l)
    tl.store(
        LSE_ptr + b_h * S + row_start + row,
        LSE_flat,
        mask=(row_idx < S),
    )


def run(Q, K, V, O, LSE):
    """Compute causal multi-head attention with Log-Sum-Exp."""
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    
    BLOCK_M = 64
    BLOCK_N = 64
    
    grid = (triton.cdiv(S, BLOCK_M), B * H)
    
    _attention_kernel[grid](
        Q, K, V, O, LSE,
        S, B, H, D,
        BLOCK_M, BLOCK_N,
        num_warps=4,
        num_stages=2,
    )