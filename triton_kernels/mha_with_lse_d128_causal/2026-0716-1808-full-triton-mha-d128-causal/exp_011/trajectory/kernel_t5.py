import torch
import triton
import triton.language as tl


@triton.jit
def fused_softmax_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    LSE_ptr,
    S,
    D_val,
    B_val,
    H_val,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    scale: tl.constexpr,
):
    block_i = tl.program_id(0)
    bh_idx = tl.program_id(1)
    
    offset_m = block_i * BLOCK_M
    
    if offset_m >= S:
        return
    
    total_S = B_val * H_val * S
    
    q_desc = tl.make_tensor_descriptor(
        Q_ptr, shape=[total_S, HEAD_DIM], strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_M, HEAD_DIM], padding_option="zero")
    
    k_desc = tl.make_tensor_descriptor(
        K_ptr, shape=[total_S, HEAD_DIM], strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_N, HEAD_DIM], padding_option="zero")
    
    v_desc = tl.make_tensor_descriptor(
        V_ptr, shape=[total_S, HEAD_DIM], strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_N, HEAD_DIM], padding_option="zero")
    
    q_partial = q_desc.load([bh_idx * S + offset_m, 0])
    
    acc_o = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    m = tl.full((BLOCK_M,), -1e20, dtype=tl.float32)
    l = tl.zeros((BLOCK_M,), dtype=tl.float32)
    
    num_kv_blocks = tl.cdiv(S, BLOCK_N)
    max_j = min(num_kv_blocks - 1, block_i)
    
    # Pre-load the first block to ensure the pipeline is primed
    if max_j >= 0:
        k_partial = k_desc.load([bh_idx * S, 0])
        v_partial = v_desc.load([bh_idx * S, 0])

    for j in range(0, max_j + 1):
        # Asynchronously fetch Key and Value chunks for the next iteration
        if j < max_j:
            next_offset_n = (j + 1) * BLOCK_N
            
            k_partial_next = k_desc.load([bh_idx * S + next_offset_n, 0])
            v_partial_next = v_desc.load([bh_idx * S + next_offset_n, 0])
        
        n_indices = j * BLOCK_N + tl.arange(0, BLOCK_N)
        q_indices = offset_m + tl.arange(0, BLOCK_M)
        mask = (n_indices[None, :] <= q_indices[:, None]) & (n_indices[None, :] < S)
        
        acc_s = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for d_chunk in range(0, HEAD_DIM, 32):
            q_slice = q_partial[:, d_chunk:d_chunk+32]
            k_slice = k_partial[:, d_chunk:d_chunk+32]
            acc_s += tl.dot(q_slice, k_slice.T)
            
        acc_s = acc_s * scale
        
        acc_s = tl.where(mask, acc_s, -1e20)
        
        m_old = m
        m_row = tl.reduce(acc_s, 1, tl.maximum)
        m_new = tl.maximum(m_old, m_row)
        
        exp_s = tl.exp(acc_s - m_new)
        exp_s = tl.where(mask, exp_s, 0.0)
        
        l_row = tl.reduce(exp_s, 1, tl.sum)
        l_new = l * tl.exp(m_old - m_new) + l_row
        
        o_scale = tl.exp(m_old - m_new)
        acc_o = acc_o * o_scale[:, None]
        
        for n_chunk in range(0, BLOCK_N, 32):
            e_slice = exp_s[:, n_chunk:n_chunk+32]
            v_slice = v_partial[n_chunk:n_chunk+32, :]
            acc_o += tl.dot(e_slice, v_slice)
            
        m = m_new
        l = l_new
        
        # Advance to the next loaded block
        if j < max_j:
            k_partial = k_partial_next
            v_partial = v_partial_next
        
    safe_l = tl.where(l > 0, l, 1.0)
    acc_o = acc_o / safe_l[:, None]
    
    lse = tl.where(l > 0, m + tl.log(l), -1e20)
    
    acc_o = acc_o.to(tl.bfloat16)
    
    query_mask = (offset_m + tl.arange(0, BLOCK_M)) < S
    o_ptr = O_ptr + (bh_idx * S + offset_m) * HEAD_DIM
    rows = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, HEAD_DIM)
    ptrs = o_ptr + rows[:, None] * HEAD_DIM + cols[None, :]
    tl.store(ptrs, acc_o, mask=query_mask[:, None])
    
    lse_ptr = LSE_ptr + bh_idx * S + offset_m
    tl.store(lse_ptr + tl.arange(0, BLOCK_M), lse, mask=query_mask)


def run(Q, K, V, O, LSE):
    """Compute Causal Multi-Head Attention forward pass returning O and LSE."""
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    
    BLOCK_M = 128
    BLOCK_N = 128
    
    scale = 1.0 / float(D)**0.5
    
    grid = (triton.cdiv(S, BLOCK_M), B * H)
    
    def alloc_fn(size: int, alignment: int, stream):
        return torch.empty(size, device="cuda", dtype=torch.int8)
    
    triton.set_allocator(alloc_fn)
    
    fused_softmax_kernel[grid](
        Q, K, V, O, LSE, S, D, B, H,
        HEAD_DIM=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, scale=scale,
        num_warps=4, num_stages=2,
    )