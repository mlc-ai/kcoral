import torch
import triton
import triton.language as tl


@triton.jit
def _mha_kernel(
    Q, K, V, O, LSE,
    S_len, D, H,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_n = tl.program_id(0) * BLOCK_M
    b = tl.program_id(1)
    h = tl.program_id(2)
    
    base = b * H * S_len * D + h * S_len * D
    
    rows_m = start_n + tl.arange(0, BLOCK_M)
    cols_d = tl.arange(0, D)
    q_off = base + rows_m[:, None] * D + cols_d[None, :]
    mask_m = rows_m[:, None] < S_len
    Q_tile = tl.load(Q + q_off, mask=mask_m, other=0.0)
    
    # Pass 1: compute m and s
    m_local = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    s_local = tl.full((BLOCK_M,), 0.0, dtype=tl.float32)
    
    for start_k in range(0, S_len, BLOCK_N):
        rows_k = start_k + tl.arange(0, BLOCK_N)
        k_off = base + rows_k[:, None] * D + cols_d[None, :]
        mask_k = rows_k[:, None] < S_len
        K_tile = tl.load(K + k_off, mask=mask_k, other=0.0)
        
        S_qk = tl.dot(Q_tile, K_tile.T) * scale
        
        k_indices = start_k + tl.arange(0, BLOCK_N)
        valid_k = k_indices < S_len
        S_qk = tl.where(valid_k[None, :], S_qk, -float('inf'))
        
        m_curr = tl.max(S_qk, axis=1)
        m_new = tl.maximum(m_local, m_curr)
        p = tl.exp(S_qk - m_new[:, None])
        s_curr = tl.sum(p, axis=1)
        s_local = s_local * tl.exp(m_local - m_new) + s_curr
        m_local = m_new
    
    # Pass 2: compute attention output
    acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)
    
    for start_k in range(0, S_len, BLOCK_N):
        rows_k = start_k + tl.arange(0, BLOCK_N)
        k_off = base + rows_k[:, None] * D + cols_d[None, :]
        mask_k = rows_k[:, None] < S_len
        
        K_tile = tl.load(K + k_off, mask=mask_k, other=0.0)
        V_tile = tl.load(V + k_off, mask=mask_k, other=0.0)
        
        S_qk = tl.dot(Q_tile, K_tile.T) * scale
        
        k_indices = start_k + tl.arange(0, BLOCK_N)
        valid_k = k_indices < S_len
        S_qk = tl.where(valid_k[None, :], S_qk, -float('inf'))
        
        p = tl.exp(S_qk - m_local[:, None])
        acc = acc + tl.dot(p, V_tile)
    
    d = 1.0 / s_local
    acc = acc * d[:, None]
    
    o_off = base + rows_m[:, None] * D + cols_d[None, :]
    tl.store(O + o_off, acc, mask=mask_m)
    
    valid = (start_n + tl.arange(0, BLOCK_M)) < S_len
    lse = m_local + tl.log(s_local)
    lse_off = b * H * S_len + h * S_len + start_n + tl.arange(0, BLOCK_M)
    tl.store(LSE + lse_off, lse, mask=valid)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S_len, D = Q.shape
    scale = 1.0 / (D ** 0.5)
    
    BLOCK_M = 64
    BLOCK_N = 64
    
    grid = (triton.cdiv(S_len, BLOCK_M), B, H)
    _mha_kernel[grid](
        Q, K, V, O, LSE,
        S_len, D, H,
        scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=16, num_stages=3,
    )