import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
import math


@triton.jit
def flash_attention_causal(
    q_desc,
    k_desc,
    v_desc,
    out_o,
    out_lse,
    S_len,
    scale,
    HEAD_DIM: tl.constexpr,
):
    """
    Optimized FlashAttention-Style Causal Forward Pass
    
    Grid map: Dim 0 encodes the Query Tile Step (rows 0-63, 64-127, etc.), 
              Dim 1 encodes the flattened Batch and Head index combined (B*H).
    """
    assert HEAD_DIM == 128
    
    step = tl.program_id(0)
    bh = tl.program_id(1)
    
    rows = tl.arange(0, 64)
    cols = tl.arange(0, 64)
    
    # Load entire Q block chunks explicitly mapping across D=128. 
    # Each load fetches a perfectly contiguous 64x64 sub-block.
    q_d0 = q_desc.load([bh * S_len + step * 64, 0])
    q_d1 = q_desc.load([bh * S_len + step * 64, 64])
    
    m_old = tl.full((64,), -1e20, dtype=tl.float32)
    l_old = tl.full((64,), 0.0, dtype=tl.float32)
    
    o_acc_0 = tl.zeros((64, 64), dtype=tl.float32)
    o_acc_1 = tl.zeros((64, 64), dtype=tl.float32)
    
    # Iterate through keys strictly abiding by causal mask limits
    for k_step in range(step + 1):
        k_d0 = k_desc.load([bh * S_len + k_step * 64, 0])
        k_d1 = k_desc.load([bh * S_len + k_step * 64, 64])
        v_d0 = v_desc.load([bh * S_len + k_step * 64, 0])
        v_d1 = v_desc.load([bh * S_len + k_step * 64, 64])
        
        # Accumulate exact over the complete inner product space D=128 partitioned into two independent exact 64x64 blocks
        s = tl.dot(q_d0, k_d0.T) + tl.dot(q_d1, k_d1.T)
        s *= scale
        
        # Causal mask guarantees we do not attend to future information. 
        # Out of limit items are mapped to mathematically safe numerical limits (-1e20).
        mask_q = step * 64 + rows
        mask_k = k_step * 64 + cols
        causal_mask = (mask_k[None, :] <= mask_q[:, None])
        
        s = tl.where(causal_mask, s, -1e20)
        
        m_curr = tl.maximum(m_old, tl.max(s, axis=1))
        p = tl.exp(s - m_curr[:, None])
        l_curr = l_old * tl.exp(m_old - m_curr) + tl.sum(p, axis=1)
        
        exp_m_diff = tl.exp(m_old - m_curr)
        o_acc_0 = o_acc_0 * exp_m_diff[:, None]
        o_acc_1 = o_acc_1 * exp_m_diff[:, None]
        
        # Compute PV product mapped natively back over native independent D=128 slices
        o_acc_0 += tl.dot(p, v_d0)
        o_acc_1 += tl.dot(p, v_d1)
        
        m_old = m_curr
        l_old = l_curr
    
    # Normalization phase converting tracking aggregates into standard probabilities
    o_0 = (o_acc_0 / l_old[:, None]).to(tl.bfloat16)
    o_1 = (o_acc_1 / l_old[:, None]).to(tl.bfloat16)
    
    # Write directly to fully contiguous 4-D physical memory layouts avoiding slow run-time indexing
    base_offset = bh * S_len * 128 + step * 64 * 128
    row_offsets = rows * 128
    
    ptr_0 = out_o + base_offset + row_offsets[:, None] + cols[None, :]
    ptr_1 = out_o + base_offset + row_offsets[:, None] + (64 + cols[None, :])
    
    row_mask = (step * 64 + rows < S_len)[:, None]
    tl.store(ptr_0, o_0, mask=row_mask)
    tl.store(ptr_1, o_1, mask=row_mask)
    
    base_offset_lse = bh * S_len + step * 64
    ptr_lse = out_lse + base_offset_lse + rows
    row_mask_lse = (step * 64 + rows < S_len)
    
    tl.store(ptr_lse, m_old + tl.log(l_old), mask=row_mask_lse)


def run(Q, K, V, O, LSE):
    """Compute causal multi-head attention forward."""
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    HEAD_DIM = D
    assert HEAD_DIM == D
    assert Q.shape == K.shape == V.shape
    
    scale = 1.0 / math.sqrt(D)
    
    # Utilize generic rank-agnostic TensorDescriptors mapped cleanly over original tensors utilizing optimized [64, 64] blocking
    q_desc = TensorDescriptor.from_tensor(Q, [64, 64], padding_option="zero")
    k_desc = TensorDescriptor.from_tensor(K, [64, 64], padding_option="zero")
    v_desc = TensorDescriptor.from_tensor(V, [64, 64], padding_option="zero")
    
    grid = (S // 64, B * H)
    
    flash_attention_causal[grid](
        q_desc, k_desc, v_desc,
        O, LSE,
        S, scale,
        HEAD_DIM=D,
        num_warps=4,
        num_stages=2,
    )