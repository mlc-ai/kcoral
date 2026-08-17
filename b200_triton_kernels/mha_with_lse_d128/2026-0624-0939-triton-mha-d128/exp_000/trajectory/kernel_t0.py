import torch
import triton
import triton.language as tl


@triton.jit
def _mha_kernel(
    Q, K, V, O, LSE,
    S, D: tl.constexpr, scale,
    BLOCK_M: tl.constexpr,
):
    idx = tl.program_id(0)
    num_blocks = triton.cdiv(S, BLOCK_M)
    b_h_idx = idx // num_blocks
    q_start = (idx % num_blocks) * BLOCK_M
    
    row_q = q_start + tl.arange(0, BLOCK_M)
    col_d = tl.arange(0, D)
    
    q_base = Q + b_h_idx * (S * D) + q_start * D
    q_tile = tl.load(q_base + row_q[:, None] * D + col_d[None, :], mask=(row_q[:, None] < S), other=0.0)
    
    m = tl.full((BLOCK_M,), -1e20, tl.float32)
    l = tl.full((BLOCK_M,), 0.0, tl.float32)
    o = tl.zeros((BLOCK_M, D), tl.float32)
    
    for k_start in range(0, S, BLOCK_M):
        row_k = k_start + tl.arange(0, BLOCK_M)
        
        k_base = K + b_h_idx * (S * D) + k_start * D
        k_tile = tl.load(k_base + row_k[:, None] * D + col_d[None, :], mask=(row_k[:, None] < S), other=0.0)
        
        p = (q_tile @ k_tile.T) / scale
        
        m_new = tl.maximum(m, tl.max(p, axis=1))
        scale_old = tl.exp(m - m_new)
        p_exp = tl.exp(p - m_new)
        l_new = l * scale_old + tl.sum(p_exp, axis=1)
        
        v_base = V + b_h_idx * (S * D) + k_start * D
        v_tile = tl.load(v_base + row_k[:, None] * D + col_d[None, :], mask=(row_k[:, None] < S), other=0.0)
        
        o = o * scale_old[:, None] + p_exp @ v_tile
        
        m = m_new
        l = l_new
    
    o_base = O + b_h_idx * (S * D) + q_start * D
    tl.store(o_base + row_q[:, None] * D + col_d[None, :], o, mask=(row_q[:, None] < S))
    
    lse = m + tl.log(l)
    lse_base = LSE + b_h_idx * S + q_start
    tl.store(lse_base + row_q, lse, mask=(row_q < S))


def run(Q, K, V, O, LSE):
    """Compute Non-causal multi-head attention forward returning O and LSE into preallocated tensors."""
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    scale = torch.sqrt(torch.tensor(D, dtype=torch.float32, device=Q.device)).item()
    num_blocks = triton.cdiv(S, 64)
    grid = (B * H * num_blocks,)
    _mha_kernel[grid](Q, K, V, O, LSE, S, D, scale, BLOCK_M=64, num_warps=4, num_stages=2)