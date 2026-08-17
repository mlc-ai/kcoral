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
    
    q_offs = tl.arange(0, 64)
    
    # Split the head dimension D=128 into two strictly 64-element chunks
    d_offs0 = tl.arange(0, 64)
    d_offs1 = tl.arange(0, 64)
    
    out_acc0 = tl.zeros((64, 64), tl.float32)
    out_acc1 = tl.zeros((64, 64), tl.float32)
    
    # Maintain statistics natively as column vectors to avoid scalar indexing constraints and simplify reduction logic
    m_prev = tl.full((64, 1), -float('inf'), tl.float32)
    sum_prev = tl.full((64, 1), 0.0, tl.float32)
    
    num_blocks = (S_val + 64 - 1) // 64
    
    base = (b_idx * 48 + h_idx) * S_val * 128
    
    # Utilize simple, explicit linear offset formulas mapped over a 2D coordinate grid
    ptr_q0 = q_offs[:, None] * 128 + d_offs0[None, :] + q_blk * 64 * 128
    ptr_q1 = q_offs[:, None] * 128 + d_offs1[None, :] + q_blk * 64 * 128 + 64
    
    q_load_mask = (q_blk * 64 + q_offs[:, None]) < S_val
    
    q0 = tl.load(Q_ptr + base + ptr_q0, mask=q_load_mask, other=0.0)
    q1 = tl.load(Q_ptr + base + ptr_q1, mask=q_load_mask, other=0.0)
    
    for k_blk in range(num_blocks):
        k_offs = tl.arange(0, 64)
        
        ptr_k0 = k_offs[:, None] * 128 + d_offs0[None, :] + k_blk * 64 * 128
        ptr_k1 = k_offs[:, None] * 128 + d_offs1[None, :] + k_blk * 64 * 128 + 64
        ptr_v0 = k_offs[:, None] * 128 + d_offs0[None, :] + k_blk * 64 * 128
        ptr_v1 = k_offs[:, None] * 128 + d_offs1[None, :] + k_blk * 64 * 128 + 64
        
        k_load_mask = (k_blk * 64 + k_offs[:, None]) < S_val
        
        k0 = tl.load(K_ptr + base + ptr_k0, mask=k_load_mask, other=0.0)
        k1 = tl.load(K_ptr + base + ptr_k1, mask=k_load_mask, other=0.0)
        v0 = tl.load(V_ptr + base + ptr_v0, mask=k_load_mask, other=0.0)
        v1 = tl.load(V_ptr + base + ptr_v1, mask=k_load_mask, other=0.0)
        
        # Compute P = Q @ K^T utilizing Tensor Cores across split D blocks
        p = tl.dot(q0, k0.T)
        p += tl.dot(q1, k1.T)
        p /= scale
        
        k_mask = (k_blk * 64 + k_offs) < S_val
        q_mask = (q_blk * 64 + q_offs) < S_val
        seq_mask = q_mask[:, None] & k_mask[None, :]
        p = tl.where(seq_mask, p, -float('inf'))
        
        m_i = tl.max(p, axis=1, keep_dims=True)
        m_new = tl.maximum(m_prev, m_i)
        
        exp_block = tl.exp(m_prev - m_new)
        
        # Scale previously aggregated components by the change in block maximum limits
        sum_prev *= exp_block
        out_acc0 *= exp_block
        out_acc1 *= exp_block
        
        p -= m_new
        exp_p = tl.exp(p)
        
        new_sum = tl.sum(exp_p, axis=1, keep_dims=True)
        sum_prev += new_sum
        
        out_acc0 += tl.dot(exp_p, v0)
        out_acc1 += tl.dot(exp_p, v1)
            
        m_prev = m_new
        
    out_acc0 /= sum_prev
    out_acc1 /= sum_prev
    
    ptr_o0 = q_offs[:, None] * 128 + d_offs0[None, :] + q_blk * 64 * 128
    ptr_o1 = q_offs[:, None] * 128 + d_offs1[None, :] + q_blk * 64 * 128 + 64
    
    tl.store(O_ptr + base + ptr_o0, out_acc0.to(tl.bfloat16), mask=q_mask[:, None])
    tl.store(O_ptr + base + ptr_o1, out_acc1.to(tl.bfloat16), mask=q_mask[:, None])
    
    # Log-Sum-Exp
    lse = m_prev + tl.log(sum_prev)
    ptr_lse = q_offs[:, None] + q_blk * 64
    q_mask_s = (q_blk * 64 + q_offs[:, None]) < S_val
    tl.store(LSE_ptr + (b_idx * 48 + h_idx) * S_val + ptr_lse, lse, mask=q_mask_s)


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