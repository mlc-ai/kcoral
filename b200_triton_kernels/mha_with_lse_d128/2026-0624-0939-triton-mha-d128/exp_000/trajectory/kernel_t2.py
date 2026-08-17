import torch
import triton
import triton.language as tl


@triton.jit
def _mha_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    S, attn_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    idx = tl.program_id(0)
    num_blocks = tl.cdiv(S, BLOCK_M)
    b_h_idx = idx // num_blocks
    q_start = (idx % num_blocks) * BLOCK_M
    
    desc_Q = tl.make_tensor_descriptor(
        Q_ptr + b_h_idx * S * 128, shape=[S, 128], strides=[128, 1],
        block_shape=[BLOCK_M, 128], padding_option="zero")
    desc_K = tl.make_tensor_descriptor(
        K_ptr + b_h_idx * S * 128, shape=[S, 128], strides=[128, 1],
        block_shape=[BLOCK_N, 128], padding_option="zero")
    desc_V = tl.make_tensor_descriptor(
        V_ptr + b_h_idx * S * 128, shape=[S, 128], strides=[128, 1],
        block_shape=[BLOCK_N, 128], padding_option="zero")
    desc_O = tl.make_tensor_descriptor(
        O_ptr + b_h_idx * S * 128, shape=[S, 128], strides=[128, 1],
        block_shape=[BLOCK_M, 128])
    
    Q_tile = desc_Q.load([q_start, 0])
    
    m = -1e20
    l = 0.0
    o = tl.zeros((BLOCK_M, 128), tl.float32)
    
    for k_start in range(0, S, BLOCK_N):
        K_tile = desc_K.load([k_start, 0])
        V_tile = desc_V.load([k_start, 0])
        
        p = tl.dot(Q_tile, K_tile.T) * attn_scale
        
        m_new = tl.maximum(m, tl.max(p, axis=1))
        scale_old = tl.exp(m - m_new)
        
        p_exp = tl.exp(p - m_new)
        
        l_new = l * scale_old + tl.sum(p_exp, axis=1)
        
        o = o * scale_old[:, None] + tl.dot(p_exp, V_tile)
        
        m = m_new
        l = l_new
    
    desc_O.store([q_start, 0], o.to(tl.bfloat16))
    
    row_q = q_start + tl.arange(0, BLOCK_M)
    lse = m + tl.log(l)
    lse_base = LSE_ptr + b_h_idx * S + q_start
    tl.store(lse_base + row_q, lse, mask=(row_q < S))


def run(Q, K, V, O, LSE):
    """Compute Non-causal multi-head attention forward returning O and LSE into preallocated tensors."""
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    scale = 1.0 / torch.sqrt(torch.tensor(D, dtype=torch.float32, device=Q.device)).item()
    num_blocks = triton.cdiv(S, 128)
    grid = (B * H * num_blocks,)
    _mha_kernel[grid](Q, K, V, O, LSE, S, scale, BLOCK_M=128, BLOCK_N=64, num_warps=4, num_stages=3)