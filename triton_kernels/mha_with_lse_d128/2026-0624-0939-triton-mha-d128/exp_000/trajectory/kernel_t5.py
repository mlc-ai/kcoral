import torch
import triton
import triton.language as tl
import math
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _mha_kernel(
    Q_desc, K_desc, V_desc, O_desc, LSE_ptr,
    S, attn_scale, H: tl.constexpr, D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    idx = tl.program_id(0)
    num_blocks = tl.cdiv(S, BLOCK_M)
    b_h_idx = idx // num_blocks
    q_start = (idx % num_blocks) * BLOCK_M
    
    b_idx = b_h_idx // H
    h_idx = b_h_idx % H
    
    Q_tile = Q_desc.load([b_idx, h_idx, q_start, 0])
    Q_fp32 = Q_tile.to(tl.float32)
    
    row_q = q_start + tl.arange(0, BLOCK_M)
    valid_q = (row_q < S)[:, None]
    
    m = tl.full((BLOCK_M,), -1e20, tl.float32)
    l = tl.full((BLOCK_M,), 0.0, tl.float32)
    o = tl.zeros((BLOCK_M, D), tl.float32)
    
    for k_start in range(0, S, BLOCK_N):
        K_tile = K_desc.load([b_idx, h_idx, k_start, 0])
        V_tile = V_desc.load([b_idx, h_idx, k_start, 0])
        
        K_fp32 = K_tile.to(tl.float32)
        V_fp32 = V_tile.to(tl.float32)
        
        p = tl.dot(Q_fp32, K_fp32.T) * attn_scale
        
        k_idx = k_start + tl.arange(0, BLOCK_N)
        valid_k = (k_idx < S)[None, :]
        p = p * valid_k + (~valid_k) * (-1e20)
        
        m_new = tl.maximum(m, tl.max(p, axis=1))
        scale_old = tl.exp(m - m_new)
        
        p_exp = tl.exp(p - m_new)
        
        l_new = l * scale_old + tl.sum(p_exp, axis=1)
        
        O_acc = o * scale_old[:, None]
        o = tl.dot(p_exp, V_fp32, acc=O_acc)
        
        m = m_new
        l = l_new
    
    o = o / l[:, None]
    o = o * valid_q
    O_bf16 = o.to(tl.bfloat16)
    O_desc.store([b_idx, h_idx, q_start, 0], O_bf16)
    
    lse = m + tl.log(l)
    b_h_idx = b_idx * H + h_idx
    lse_base = LSE_ptr + b_h_idx * S + q_start
    tl.store(lse_base + row_q, lse, mask=(row_q < S))


def run(Q, K, V, O, LSE):
    """Compute Non-causal multi-head attention forward returning O and LSE into preallocated tensors."""
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    attn_scale = 1.0 / math.sqrt(D)
    
    BLOCK_M = 64
    BLOCK_N = 64
    
    Q_desc = TensorDescriptor.from_tensor(Q, [BLOCK_M, D], contiguous_to_dim=1)
    K_desc = TensorDescriptor.from_tensor(K, [BLOCK_N, D], contiguous_to_dim=1)
    V_desc = TensorDescriptor.from_tensor(V, [BLOCK_N, D], contiguous_to_dim=1)
    O_desc = TensorDescriptor.from_tensor(O, [BLOCK_M, D], contiguous_to_dim=1)
    
    num_blocks = triton.cdiv(S, BLOCK_M)
    grid = (B * H * num_blocks,)
    _mha_kernel[grid](Q_desc, K_desc, V_desc, O_desc, LSE, S, attn_scale, H, D, 
                      BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=4, num_stages=3)