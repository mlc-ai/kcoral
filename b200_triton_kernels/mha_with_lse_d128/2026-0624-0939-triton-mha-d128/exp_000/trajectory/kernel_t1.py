import torch
import triton
import triton.language as tl


@triton.jit
def _mha_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    S, D: tl.constexpr, attn_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    idx = tl.program_id(0)
    num_blocks = triton.cdiv(S, BLOCK_M)
    b_h_idx = idx // num_blocks
    q_start = (idx % num_blocks) * BLOCK_M
    
    row_q = q_start + tl.arange(0, BLOCK_M)
    col_d = tl.arange(0, D)
    
    q_base = Q_ptr + b_h_idx * (S * D) + q_start * D
    Q_tile = tl.load(q_base + row_q[:, None] * D + col_d[None, :], mask=(row_q[:, None] < S), other=0.0)
    
    m = tl.full((BLOCK_M,), -1e20, tl.float32)
    l = tl.full((BLOCK_M,), 0.0, tl.float32)
    o = tl.zeros((BLOCK_M, D), tl.float32)
    
    for k_start in range(0, S, BLOCK_N):
        k_idx = k_start + tl.arange(0, BLOCK_N)
        
        k_base = K_ptr + b_h_idx * (S * D) + k_start * D
        K_tile = tl.load(k_base + k_idx[:, None] * D + col_d[None, :], mask=(k_idx[:, None] < S), other=0.0)
        
        P = tl.dot(Q_tile, K_tile.T) * attn_scale
        
        v_base = V_ptr + b_h_idx * (S * D) + k_start * D
        V_tile = tl.load(v_base + k_idx[:, None] * D + col_d[None, :], mask=(k_idx[:, None] < S), other=0.0)
        
        m_new = tl.maximum(m, tl.max(P, axis=1))
        scale_old = tl.exp(m - m_new)
        
        P_exp = tl.exp(P - m_new)
        P_exp = P_exp * (k_idx[None, :] < S)
        
        l_new = l * scale_old + tl.sum(P_exp, axis=1)
        
        o = o * scale_old[:, None] + tl.dot(P_exp, V_tile)
        
        m = m_new
        l = l_new
    
    o_base = O_ptr + b_h_idx * (S * D) + q_start * D
    O_out = o.to(tl.bfloat16)
    tl.store(o_base + row_q[:, None] * D + col_d[None, :], O_out, mask=(row_q[:, None] < S))
    
    lse = m + tl.log(l)
    lse_base = LSE_ptr + b_h_idx * S + q_start
    tl.store(lse_base + row_q, lse, mask=(row_q < S))


def run(Q, K, V, O, LSE):
    """Compute Non-causal multi-head attention forward returning O and LSE into preallocated tensors."""
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    attn_scale = 1.0 / torch.sqrt(torch.tensor(D, dtype=torch.float32, device=Q.device)).item()
    num_blocks = triton.cdiv(S, 64)
    grid = (B * H * num_blocks,)
    _mha_kernel[grid](Q, K, V, O, LSE, S, D, attn_scale, BLOCK_M=64, BLOCK_N=64, num_warps=4, num_stages=2)