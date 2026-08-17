import torch
import triton
import triton.language as tl
import math


@triton.jit
def _mha_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    S, attn_scale,
    Q_stride_b, Q_stride_h, Q_stride_s,
    K_stride_b, K_stride_h, K_stride_s,
    V_stride_b, V_stride_h, V_stride_s,
    O_stride_b, O_stride_h, O_stride_s,
    LSE_stride_b, LSE_stride_h,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    q_block = tl.program_id(2)
    q_start = q_block * BLOCK_M
    
    q_ptr = Q_ptr + b_idx * Q_stride_b + h_idx * Q_stride_h
    k_ptr = K_ptr + b_idx * K_stride_b + h_idx * K_stride_h
    v_ptr = V_ptr + b_idx * V_stride_b + h_idx * V_stride_h
    
    row_q = q_start + tl.arange(0, BLOCK_M)
    col_d = tl.arange(0, D)
    valid_q = (row_q < S)[:, None]
    
    Q_tile = tl.load(q_ptr + row_q[:, None] * Q_stride_s + col_d[None, :], mask=valid_q, other=0.0)
    
    m = -1e20
    l = 0.0
    o = tl.zeros((BLOCK_M, D), tl.float32)
    
    num_k_blocks = tl.cdiv(S, BLOCK_N)
    for k_block in range(num_k_blocks):
        k_start = k_block * BLOCK_N
        row_k = k_start + tl.arange(0, BLOCK_N)
        valid_k = (row_k < S)[:, None]
        
        K_tile = tl.load(k_ptr + row_k[:, None] * K_stride_s + col_d[None, :], mask=valid_k, other=0.0)
        V_tile = tl.load(v_ptr + row_k[:, None] * V_stride_s + col_d[None, :], mask=valid_k, other=0.0)
        
        p = tl.dot(Q_tile, K_tile.T) * attn_scale
        
        valid_k_col = (row_k < S)[None, :]
        p = tl.where(valid_k_col, p, -1e20)
        
        m_new = tl.maximum(m, tl.max(p, axis=1))
        scale_old = tl.exp(m - m_new)
        
        p_exp = tl.exp(p - m_new)
        
        l_new = l * scale_old + tl.sum(p_exp, axis=1)
        
        V_fp32 = V_tile.to(tl.float32)
        o = tl.dot(p_exp, V_fp32, acc=o * scale_old[:, None])
        
        m = m_new
        l = l_new
    
    o = o / l[:, None]
    
    out_ptr = O_ptr + b_idx * O_stride_b + h_idx * O_stride_h + row_q[:, None] * O_stride_s + col_d[None, :]
    tl.store(out_ptr, o.to(tl.bfloat16), mask=valid_q)
    
    lse = m + tl.log(l)
    lse_ptr = LSE_ptr + b_idx * LSE_stride_b + h_idx * LSE_stride_h + q_start + row_q
    tl.store(lse_ptr, lse, mask=(row_q < S))


def run(Q, K, V, O, LSE):
    """Compute Non-causal multi-head attention forward returning O and LSE into preallocated tensors."""
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    attn_scale = 1.0 / math.sqrt(D)
    
    BLOCK_M = 64
    BLOCK_N = 64
    
    Q_stride_b, Q_stride_h, Q_stride_s, _ = Q.stride()
    K_stride_b, K_stride_h, K_stride_s, _ = K.stride()
    V_stride_b, V_stride_h, V_stride_s, _ = V.stride()
    O_stride_b, O_stride_h, O_stride_s, _ = O.stride()
    LSE_stride_b, LSE_stride_h, _ = LSE.stride()
    
    num_q_blocks = triton.cdiv(S, BLOCK_M)
    grid = (B, H, num_q_blocks)
    _mha_kernel[grid](Q, K, V, O, LSE, S, attn_scale,
                      Q_stride_b, Q_stride_h, Q_stride_s,
                      K_stride_b, K_stride_h, K_stride_s,
                      V_stride_b, V_stride_h, V_stride_s,
                      O_stride_b, O_stride_h, O_stride_s,
                      LSE_stride_b, LSE_stride_h,
                      D=D,
                      BLOCK_M=BLOCK_M,
                      BLOCK_N=BLOCK_N,
                      num_warps=4,
                      num_stages=3)