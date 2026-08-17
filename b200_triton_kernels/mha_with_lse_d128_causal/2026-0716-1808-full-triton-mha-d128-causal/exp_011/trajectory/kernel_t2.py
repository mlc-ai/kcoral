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
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    block_i = tl.program_id(0)
    bh_idx = tl.program_id(1)
    
    offset_m = block_i * BLOCK_M
    
    # Load the query block. It remains static throughout the execution.
    q_partial = q_desc.load([bh_idx * S + offset_m, 0])
    
    acc_o = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    m = tl.full((BLOCK_M,), -1e20, dtype=tl.float32)
    l = tl.zeros((BLOCK_M,), dtype=tl.float32)
    
    num_kv_blocks = tl.cdiv(S, BLOCK_N)
    max_j = (block_i * BLOCK_M + BLOCK_M - 1) // BLOCK_N
    if max_j >= num_kv_blocks:
        max_j = num_kv_blocks - 1
    
    for j in range(0, max_j + 1):
        offset_n = j * BLOCK_N
        
        k_partial = k_desc.load([bh_idx * S + offset_n, 0])
        v_partial = v_desc.load([bh_idx * S + offset_n, 0])
        
        acc_s = tl.dot(q_partial, k_partial)
        acc_s = acc_s * (1.0 / tl.sqrt(tl.float32(HEAD_DIM)))
        
        n_indices = offset_n + tl.arange(0, BLOCK_N)
        q_indices = offset_m + tl.arange(0, BLOCK_M)
        mask = (n_indices[None, :] <= q_indices[:, None]) & (n_indices[None, :] < S)
        
        acc_s = acc_s * mask + (1 - mask) * (-1e20)
        
        m_old = m
        m_row = tl.reduce(acc_s, 1, tl.maximum)
        m_new = tl.maximum(m_old, m_row)
        
        exp_s = tl.exp(acc_s - m_new)
        exp_s = exp_s * mask 
        
        l_row = tl.reduce(exp_s, 1, tl.sum)
        l_new = l * tl.exp(m_old - m_new) + l_row
        
        o_scale = tl.exp(m_old - m_new)
        acc_o = acc_o * o_scale[:, None]
        
        acc_o += tl.dot(exp_s, v_partial.T)
        
        m = m_new
        l = l_new
        
    safe_l = tl.where(l > 0, l, 1.0)
    acc_o = acc_o / safe_l[:, None]
    
    lse = m + tl.log(l)
    
    acc_o = acc_o.to(tl.bfloat16)
    
    query_mask = (offset_m + tl.arange(0, BLOCK_M)) < S
    o_ptr = O_ptr + (bh_idx * S + offset_m) * HEAD_DIM
    rows = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, HEAD_DIM)
    ptrs = o_ptr + rows[:, None] * HEAD_DIM + cols[None, :]
    tl.store(ptrs, acc_o, mask=query_mask[:, None])
    
    lse_ptr = LSE_ptr + bh_idx * S + offset_m
    valid_rows = tl.arange(0, BLOCK_M)
    tl.store(lse_ptr + valid_rows, lse, mask=query_mask)


def run(Q, K, V, O, LSE):
    """Compute Causal Multi-Head Attention forward pass returning O and LSE."""
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    
    q_desc = TensorDescriptor.from_tensor(Q, [BLOCK_M, HEAD_DIM])
    k_desc = TensorDescriptor.from_tensor(K, [BLOCK_N, HEAD_DIM])
    v_desc = TensorDescriptor.from_tensor(V, [BLOCK_N, HEAD_DIM])
    
    grid = (triton.cdiv(S, BLOCK_M), B * H)
    
    fused_softmax_kernel[grid](
        q_desc,
        k_desc,
        v_desc,
        O,
        LSE,
        S,
        HEAD_DIM=128,
        BLOCK_M=128,
        BLOCK_N=64,
        num_warps=4,
        num_stages=3,
    )