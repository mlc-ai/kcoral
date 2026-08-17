import torch
import triton
import triton.language as tl
import math


def _alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(_alloc_fn)


@triton.jit
def _mha_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    S, attn_scale, H: tl.constexpr, D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    b_h_idx = tl.program_id(0)
    q_block = tl.program_id(1)
    q_start = q_block * BLOCK_M
    
    batch_offset_bytes = b_h_idx * S * D * 2  
    
    desc_Q = tl.make_tensor_descriptor(
        Q_ptr + batch_offset_bytes // 2, shape=[S, D], strides=[D, 1],
        block_shape=[BLOCK_M, D], padding_option="zero")
    desc_K = tl.make_tensor_descriptor(
        K_ptr + batch_offset_bytes // 2, shape=[S, D], strides=[D, 1],
        block_shape=[BLOCK_N, D], padding_option="zero")
    desc_V = tl.make_tensor_descriptor(
        V_ptr + batch_offset_bytes // 2, shape=[S, D], strides=[D, 1],
        block_shape=[BLOCK_N, D], padding_option="zero")
    desc_O = tl.make_tensor_descriptor(
        O_ptr + batch_offset_bytes // 2, shape=[S, D], strides=[D, 1],
        block_shape=[BLOCK_M, D])
    
    Q_tile = desc_Q.load([q_start, 0])
    Q_fp32 = Q_tile.to(tl.float32)
    
    row_q = q_start + tl.arange(0, BLOCK_M)
    valid_q = (row_q < S)[:, None]
    
    m = tl.full((BLOCK_M,), -1e20, tl.float32)
    l = tl.full((BLOCK_M,), 0.0, tl.float32)
    o = tl.zeros((BLOCK_M, D), tl.float32)
    
    num_k_blocks = tl.cdiv(S, BLOCK_N)
    for k_block in range(num_k_blocks):
        K_tile = desc_K.load([k_block * BLOCK_N, 0])
        K_fp32 = K_tile.to(tl.float32)
        
        V_tile = desc_V.load([k_block * BLOCK_N, 0])
        V_fp32 = V_tile.to(tl.float32)
        
        p = tl.dot(Q_fp32, K_fp32.T) * attn_scale
        
        k_idx = k_block * BLOCK_N + tl.arange(0, BLOCK_N)
        valid_k = (k_idx < S)[None, :]
        p = p * valid_k + (~valid_k) * (-1e20)
        
        m_new = tl.maximum(m, tl.max(p, axis=1))
        scale_old = tl.exp(m - m_new)
        
        p_exp = tl.exp(p - m_new)
        
        l_new = l * scale_old + tl.sum(p_exp, axis=1)
        
        o = tl.dot(p_exp, V_fp32, acc=o * scale_old[:, None])
        
        m = m_new
        l = l_new
    
    o = o / l[:, None]
    o = o * valid_q
    O_bf16 = o.to(tl.bfloat16)
    desc_O.store([q_start, 0], O_bf16)
    
    m_base = LSE_ptr + b_h_idx * S + q_start
    l_base = m_base + S * BLOCK_M 
    tl.store(m_base + row_q, m, mask=(row_q < S))
    tl.store(l_base + row_q, l, mask=(row_q < S))


@triton.jit
def _lse_kernel(
    m_ptr, l_ptr, lse_ptr,
    S, H: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    idx = tl.program_id(0)
    num_blocks = tl.cdiv(S, BLOCK_M)
    b_h_idx = idx // num_blocks
    q_start = (idx % num_blocks) * BLOCK_M
    
    row_q = q_start + tl.arange(0, BLOCK_M)
    
    m_base = m_ptr + b_h_idx * S + q_start
    l_base = m_base + S * BLOCK_M
    
    m = tl.load(m_base + row_q, mask=(row_q < S), other=0.0)
    l = tl.load(l_base + row_q, mask=(row_q < S), other=1.0)
    
    lse = m + tl.log(l)
    
    lse_base = lse_ptr + b_h_idx * S + q_start
    tl.store(lse_base + row_q, lse, mask=(row_q < S))


def run(Q, K, V, O, LSE):
    """Compute Non-causal multi-head attention forward returning O and LSE into preallocated tensors."""
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    attn_scale = 1.0 / math.sqrt(D)
    
    BLOCK_M = 64
    BLOCK_N = 64
    
    num_q_blocks = triton.cdiv(S, BLOCK_M)
    
    m_tmp = torch.empty((B, H, S), dtype=torch.float32, device=Q.device)
    l_tmp = torch.empty((B, H, S), dtype=torch.float32, device=Q.device)
    
    grid_main = (B * H, num_q_blocks)
    _mha_kernel[grid_main](Q, K, V, O, LSE, S, attn_scale, H, D, 
                           BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=4, num_stages=3)
    
    grid_lse = (B * H * num_q_blocks,)
    _lse_kernel[grid_lse](m_tmp, l_tmp, LSE, S, H, BLOCK_M=BLOCK_M, num_warps=4)