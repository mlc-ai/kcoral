import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
import math


@triton.jit
def _attention_kernel(
    Q_desc, K_desc, V_desc, O, LSE,
    S_val,
    scale,
):
    q_blk = tl.program_id(0)
    h_idx = tl.program_id(1)
    b_idx = tl.program_id(2)
    
    bh_idx = b_idx * 48 + h_idx
    row_offset = bh_idx * S_val
    
    d_offs0 = tl.arange(0, 64)
    d_offs1 = tl.arange(64, 128)
    q_offs = tl.arange(0, 64)
    
    out_acc0 = tl.zeros((64, 64), tl.float32)
    out_acc1 = tl.zeros((64, 64), tl.float32)
    
    m_prev = tl.full((64,), -float('inf'), tl.float32)
    sum_prev = tl.full((64,), 0.0, tl.float32)
    
    num_blocks = (S_val + 64 - 1) // 64
    
    # Utilize TMA descriptors for efficient layout-aware loads and built-in boundary checks.
    q0 = Q_desc.load([row_offset + q_blk * 64, 0])
    q1 = Q_desc.load([row_offset + q_blk * 64, 64])
    
    q_mask = (q_blk * 64 + q_offs) < S_val
    
    for k_blk in range(num_blocks):
        k_row_offset = row_offset + k_blk * 64
        
        k0 = K_desc.load([k_row_offset, 0])
        k1 = K_desc.load([k_row_offset, 64])
        v0 = V_desc.load([k_row_offset, 0])
        v1 = V_desc.load([k_row_offset, 64])
        
        # Compute P = Q @ K^T utilizing Tensor Cores across split D blocks.
        p = tl.dot(q0, k0.T)
        p += tl.dot(q1, k1.T)
        p /= scale
        
        # Mask out-of-bounds sequence positions in P
        k_mask = (k_blk * 64 + q_offs) < S_val
        seq_mask = q_mask[:, None] & k_mask[None, :]
        p = tl.where(seq_mask, p, -float('inf'))
        
        # Numerically stable online softmax (FlashAttention style)
        m_i = tl.max(p, axis=1, keep_dims=True)
        m_new = tl.maximum(m_prev[:, None], m_i)
        
        exp_block = tl.exp(m_prev[:, None] - m_new)
        
        # Rescale previously aggregated sum and context with the new global maximum block
        sum_prev *= exp_block[:, 0]
        out_acc0 *= exp_block
        out_acc1 *= exp_block
        
        p -= m_new
        exp_p = tl.exp(p)
        
        new_sum = tl.sum(exp_p, axis=1, keep_dims=True)
        sum_prev += new_sum[:, 0]
        
        # Multiply by V and accumulate into context.
        out_acc0 += tl.dot(exp_p, v0)
        out_acc1 += tl.dot(exp_p, v1)
            
        m_prev = m_new[:, 0]
        
    # Output division and cast to bf16
    out_acc0 /= sum_prev[:, None]
    out_acc1 /= sum_prev[:, None]
    
    ptr_o0 = O + bh_idx * (S_val * 128) + q_blk * 64 * 128 + q_offs[:, None] * 128 + d_offs0[None, :]
    tl.store(ptr_o0, out_acc0.to(tl.bfloat16), mask=q_mask[:, None])
    
    ptr_o1 = O + bh_idx * (S_val * 128) + q_blk * 64 * 128 + q_offs[:, None] * 128 + 64 + d_offs1[None, :]
    tl.store(ptr_o1, out_acc1.to(tl.bfloat16), mask=q_mask[:, None])
    
    # Log-Sum-Exp
    lse = m_prev + tl.log(sum_prev)
    ptr_lse = LSE + bh_idx * S_val + q_blk * 64 + q_offs
    tl.store(ptr_lse, lse, mask=q_mask)


def run(Q, K, V, O, LSE):
    """Compute non-causal Multi-Head Attention and output Log-Sum-Exp."""
    torch.cuda.set_device(Q.device)
    B, H, S_val, D = Q.shape
    scale = 1.0 / math.sqrt(128)

    Q_desc = TensorDescriptor.from_tensor(Q, [64, 64])
    K_desc = TensorDescriptor.from_tensor(K, [64, 64])
    V_desc = TensorDescriptor.from_tensor(V, [64, 64])

    grid = (triton.cdiv(S_val, 64), H, B)
    _attention_kernel[grid](
        Q_desc, K_desc, V_desc, O, LSE,
        S_val, scale,
        num_warps=4, num_stages=2
    )