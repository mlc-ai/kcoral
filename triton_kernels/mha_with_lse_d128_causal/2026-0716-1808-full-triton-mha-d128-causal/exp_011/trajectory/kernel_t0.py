import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def fused_softmax_kernel(
    q_desc,
    k_desc,
    v_desc,
    O_ptr,
    LSE_ptr,
    S,
    H,
    D,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    block_i = tl.program_id(0)
    head_idx = tl.program_id(1)
    
    batch_idx = head_idx // H
    head_idx = head_idx % H
    
    offset_m = block_i * BLOCK_M
    batch_head_offset = (batch_idx * H + head_idx) * S
    
    # Load the entire query block. It remains static throughout the execution.
    q_partial = q_desc.load([offset_m, 0])
    
    acc_o = tl.zeros((BLOCK_M, D), dtype=tl.float32)
    m = tl.full((BLOCK_M,), -1e20, dtype=tl.float32)
    l = tl.zeros((BLOCK_M,), dtype=tl.float32)
    
    query_indices = offset_m + tl.arange(0, BLOCK_M)
    
    num_blocks = tl.cdiv(S, BLOCK_N)
    max_j = min(num_blocks - 1, block_i)
    
    for j in range(0, max_j + 1):
        offset_n = j * BLOCK_N
        
        # Asynchronously fetch Key and Value chunks
        k_partial = k_desc.load([offset_n, 0])
        v_partial = v_desc.load([offset_n, 0])
        
        # Compute Q @ K^T
        acc_s = tl.dot(q_partial, k_partial.T) * (1.0 / tl.sqrt(tl.float32(D)))
        
        key_indices = offset_n + tl.arange(0, BLOCK_N)
        mask = (key_indices[None, :] <= query_indices[:, None]) & (key_indices[None, :] < S)
        
        # Mask the attention scores prior to numerical reduction to prevent arbitrary values
        acc_s = acc_s * mask + (1 - mask) * (-1e20)
        
        m_old = m
        m_row = tl.reduce(acc_s, 1, tl.maximum)
        m_new = tl.maximum(m_old, m_row)
        
        exp_s = tl.exp(acc_s - m_new)
        exp_s = exp_s * mask  # Mask again post-exp to strictly avoid denormals affecting summation
        
        l_row = tl.reduce(exp_s, 1, tl.sum)
        l_new = l * tl.exp(m_old - m_new) + l_row
        
        # Scale existing accumulated context by change in normalizing factor
        o_scale = tl.exp(m_old - m_new)
        acc_o = acc_o * o_scale[:, None]
        
        # Incorporate newly attended values into output state
        acc_o = acc_o + tl.dot(exp_s, v_partial)
        
        m = m_new
        l = l_new
    
    # Apply final normalization mapping into standard Softmax territory
    acc_o = acc_o / l[:, None]
    
    # Compute Log-Sum-Exp
    lse = m + tl.log(l)
    
    # Cast to destination dtype and commit results sequentially
    acc_o = acc_o.to(tl.bfloat16)
    
    o_ptr = O_ptr + batch_head_offset * D + offset_m * D
    query_mask = query_indices < S
    
    rows = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, D)
    ptrs = o_ptr + rows[:, None] * D + cols[None, :]
    tl.store(ptrs, acc_o, mask=query_mask[:, None])
    
    lse_ptr = LSE_ptr + batch_head_offset + offset_m
    tl.store(lse_ptr + tl.arange(0, BLOCK_M), lse, mask=query_mask)


def run(Q, K, V, O, LSE):
    """Compute Causal Multi-Head Attention forward pass returning O and LSE."""
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    
    # Establish Hardware TMA Descriptors leveraging native physical contiguity ([BH*S, D])
    q_desc = TensorDescriptor.from_tensor(Q, [B * H * S, D])
    k_desc = TensorDescriptor.from_tensor(K, [B * H * S, D])
    v_desc = TensorDescriptor.from_tensor(V, [B * H * S, D])
    
    # Spanning entire sequence space mapped linearly per unique attention head
    grid = (triton.cdiv(S, 128), B * H)
    
    fused_softmax_kernel[grid](
        q_desc,
        k_desc,
        v_desc,
        O,
        LSE,
        S,
        H,
        D,
        BLOCK_M=128,
        BLOCK_N=64,
        num_warps=4,
        num_stages=3,
    )