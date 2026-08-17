import torch
import triton
import triton.language as tl


def _alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(_alloc_fn)


@triton.jit
def _mha_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    S, attn_scale, D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    idx = tl.program_id(0)
    num_blocks = tl.cdiv(S, BLOCK_M)
    b_h_idx = idx // num_blocks
    q_start = (idx % num_blocks) * BLOCK_M
    
    desc_Q = tl.make_tensor_descriptor(
        Q_ptr + b_h_idx * S * D, shape=[S, D], strides=[D, 1],
        block_shape=[BLOCK_M, D], padding_option="zero")
    desc_K = tl.make_tensor_descriptor(
        K_ptr + b_h_idx * S * D, shape=[S, D], strides=[D, 1],
        block_shape=[BLOCK_N, D], padding_option="zero")
    desc_V = tl.make_tensor_descriptor(
        V_ptr + b_h_idx * S * D, shape=[S, D], strides=[D, 1],
        block_shape=[BLOCK_N, D], padding_option="zero")
    desc_O = tl.make_tensor_descriptor(
        O_ptr + b_h_idx * S * D, shape=[S, D], strides=[D, 1],
        block_shape=[BLOCK_M, D])
    
    Q_tile = desc_Q.load([q_start, 0])
    
    row_q = q_start + tl.arange(0, BLOCK_M)
    valid_q = (row_q < S)[:, None]
    
    m = tl.full((BLOCK_M,), -1e20, tl.float32)
    l = tl.full((BLOCK_M,), 0.0, tl.float32)
    o = tl.zeros((BLOCK_M, D), tl.float32)
    
    for k_start in range(0, S, BLOCK_N):
        K_tile = desc_K.load([k_start, 0])
        V_tile = desc_V.load([k_start, 0])
        
        p = tl.dot(Q_tile, K_tile.T) * attn_scale
        
        k_idx = k_start + tl.arange(0, BLOCK_N)
        valid_k = (k_idx[None, :] < S)
        
        m_new = tl.maximum(m, tl.max(p * valid_k, axis=1))
        scale_old = tl.exp(m - m_new)
        
        p_exp = tl.exp(p - m_new) * valid_k * valid_q
        
        l_new = l * scale_old + tl.sum(p_exp, axis=1)
        
        o = o * scale_old[:, None] + tl.dot(p_exp, V_tile)
        
        m = m_new
        l = l_new
    
    o = o / l[:, None]
    desc_O.store([q_start, 0], o.to(tl.bfloat16))
    
    lse = m + tl.log(l) - tl.log(S)
    lse_base = LSE_ptr + b_h_idx * S + q_start
    tl.store(lse_base + row_q, lse, mask=(row_q < S))


def run(Q, K, V, O, LSE):
    """Compute Non-causal multi-head attention forward returning O and LSE into preallocated tensors."""
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    attn_scale = 1.0 / float(D)  
    
    BLOCK_M = 64
    BLOCK_N = 64
    
    num_blocks = triton.cdiv(S, BLOCK_M)
    grid = (B * H * num_blocks,)
    _mha_kernel[grid](Q, K, V, O, LSE, S, attn_scale, D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=4, num_stages=3)