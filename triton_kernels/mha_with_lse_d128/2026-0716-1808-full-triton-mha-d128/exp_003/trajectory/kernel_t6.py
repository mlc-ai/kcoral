import torch
import triton
import triton.language as tl
import math


@triton.jit
def _attention_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    S_val,
    scale,
):
    q_blk = tl.program_id(0)
    h_idx = tl.program_id(1)
    b_idx = tl.program_id(2)
    
    bh_idx = b_idx * 48 + h_idx
    
    q_offs = tl.arange(0, 64)
    k_offs = tl.arange(0, 64)
    
    d_offs0 = tl.arange(0, 64)
    d_offs1 = tl.arange(0, 64)
    
    out_acc0 = tl.zeros((64, 64), tl.float32)
    out_acc1 = tl.zeros((64, 64), tl.float32)
    
    m_prev = tl.full((64,), -float('inf'), tl.float32)
    sum_prev = tl.full((64,), 0.0, tl.float32)
    
    num_blocks = (S_val + 64 - 1) // 64
    
    # Explicitly precompute 2D coordinate mappings using fundamental arithmetic to resolve indexing conflicts
    zero_row = tl.zeros((1,), tl.int32)
    zero_col = tl.zeros((1,), tl.int32)
    stride_row = 128
    stride_col = 1
    offset_row = q_blk * 64 * 128
    offset_col = 0
    
    q_offs_row = q_offs * stride_row + zero_row
    d_offs0_col = d_offs0 * stride_col + zero_col
    
    Q_0 = q_offs_row * 128 + d_offs0_col
    Q_1 = q_offs_row * 128 + 64 + d_offs1_col
    
    k_offs_row = k_offs * stride_row + zero_row
    d_offs0_col_k = d_offs0 * stride_col + zero_col
    
    K_0 = k_offs_row * 128 + d_offs0_col_k
    K_1 = k_offs_row * 128 + 64 + d_offs1_col
    
    V_0 = k_offs_row * 128 + d_offs0_col_k
    V_1 = k_offs_row * 128 + 64 + d_offs1_col
    
    row_base = bh_idx * S_val * 128
    
    Q_S0 = row_base + offset_row + Q_0
    Q_S1 = row_base + offset_row + Q_1
    
    # Mask boundaries natively utilizing the numerically native coordinate mapping limits
    Q_M = Q_S0 < (row_base + (q_blk + 1) * 64 * 128)
    
    q0 = tl.load(Q_ptr + Q_S0, mask=Q_M, other=0.0)
    q1 = tl.load(Q_ptr + Q_S1, mask=Q_M, other=0.0)
    
    for k_blk in range(num_blocks):
        K_S0 = row_base + k_blk * 64 * 128 + K_0
        K_S1 = row_base + k_blk * 64 * 128 + K_1
        
        K_M = K_S0 < (row_base + (k_blk + 1) * 64 * 128)
        
        k0 = tl.load(K_ptr + K_S0, mask=K_M, other=0.0)
        k1 = tl.load(K_ptr + K_S1, mask=K_M, other=0.0)
        
        V_S0 = row_base + k_blk * 64 * 128 + V_0
        V_S1 = row_base + k_blk * 64 * 128 + V_1
        
        v0 = tl.load(V_ptr + V_S0, mask=K_M, other=0.0)
        v1 = tl.load(V_ptr + V_S1, mask=K_M, other=0.0)
        
        p = tl.dot(q0, k0.T)
        p += tl.dot(q1, k1.T)
        p /= scale
        
        k_mask = (k_blk * 64 + k_offs) < S_val
        q_mask = (q_blk * 64 + q_offs) < S_val
        seq_mask = q_mask[:, None] & k_mask[None, :]
        p = tl.where(seq_mask, p, -float('inf'))
        
        m_i = tl.max(p, axis=1, keep_dims=True)
        m_new = tl.maximum(m_prev[:, None], m_i)
        
        exp_block = tl.exp(m_prev[:, None] - m_new)
        
        sum_prev *= exp_block[:, 0]
        out_acc0 *= exp_block
        out_acc1 *= exp_block
        
        p -= m_new
        exp_p = tl.exp(p)
        
        new_sum = tl.sum(exp_p, axis=1, keep_dims=True)
        sum_prev += new_sum[:, 0]
        
        out_acc0 += tl.dot(exp_p, v0)
        out_acc1 += tl.dot(exp_p, v1)
            
        m_prev = m_new[:, 0]
        
    out_acc0 /= sum_prev[:, None]
    out_acc1 /= sum_prev[:, None]
    
    # Reuse cleanly structured numeric formulas for the Output layout to guarantee mapping accuracy
    base_o = row_base + offset_row
    O_S0 = base_o + Q_0
    O_S1 = base_o + Q_1
    
    tl.store(O_ptr + O_S0, out_acc0.to(tl.bfloat16), mask=Q_M)
    tl.store(O_ptr + O_S1, out_acc1.to(tl.bfloat16), mask=Q_M)
    
    lse = m_prev + tl.log(sum_prev)
    ptr_lse = LSE_ptr + bh_idx * S_val + q_blk * 64 + q_offs
    q_mask_s = (q_blk * 64 + q_offs) < S_val
    tl.store(ptr_lse, lse, mask=q_mask_s)


def run(Q, K, V, O, LSE):
    """Compute non-causal Multi-Head Attention and output Log-Sum-Exp."""
    torch.cuda.set_device(Q.device)
    B, H, S_val, D = Q.shape
    scale = 1.0 / math.sqrt(128)

    grid = (triton.cdiv(S_val, 64), H, B)
    _attention_kernel[grid](
        Q, K, V, O, LSE,
        S_val, scale,
        num_warps=4, num_stages=2
    )