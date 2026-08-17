import torch
import triton
import triton.language as tl


@triton.jit
def _attention_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    S_len, scale, H,
):
    """
    Optimized FlashAttention kernel targeting Hopper WGMMA capabilities.
    
    Uses TMA descriptors for all memory operations and splits the Head Dimension (D=128) 
    into two independent contiguous chunks of width 64.
    
    Grid mapping: (ceil(S/32), B, H)
    - Axis 0: Query block index (each CTA handles 32 query rows)
    - Axis 1: Batch index
    - Axis 2: Attention head index
    """
    
    q_blk = tl.program_id(0)
    b = tl.program_id(1)
    h = tl.program_id(2)
    
    global_row = q_blk * 32
    chunk_bh = b * H + h
    
    s_D = 128
    
    q_ptr_bh = Q_ptr + chunk_bh * S_len * s_D
    q_desc = tl.make_tensor_descriptor(
        q_ptr_bh, shape=[S_len, s_D], strides=[s_D, 1],
        block_shape=[32, 64], padding_option="zero")
    
    k_ptr_bh = K_ptr + chunk_bh * S_len * s_D
    k_desc = tl.make_tensor_descriptor(
        k_ptr_bh, shape=[S_len, s_D], strides=[s_D, 1],
        block_shape=[32, 64], padding_option="zero")
    
    v_ptr_bh = V_ptr + chunk_bh * S_len * s_D
    v_desc = tl.make_tensor_descriptor(
        v_ptr_bh, shape=[S_len, s_D], strides=[s_D, 1],
        block_shape=[32, 64], padding_option="zero")
    
    q0 = q_desc.load([global_row, 0])
    q1 = q_desc.load([global_row, 64])
    
    acc_O0 = tl.zeros((32, 64), tl.float32)
    acc_O1 = tl.zeros((32, 64), tl.float32)
    
    m_i = tl.full((32,), -1e38, tl.float32)
    l_i = tl.zeros((32,), 1.0, tl.float32)
    
    num_kv_iters = tl.cdiv(S_len, 32)
    
    for kv_blk in tl.range(0, num_kv_iters, num_stages=2):
        global_kv_row = kv_blk * 32
        
        k0 = k_desc.load([global_kv_row, 0])
        k1 = k_desc.load([global_kv_row, 64])
        
        s = (tl.dot(q0, k0.T) + tl.dot(q1, k1.T)) * scale
        
        kv_offs = global_kv_row + tl.arange(0, 32)
        kv_mask = kv_offs < S_len
        q_offs = global_row + tl.arange(0, 32)
        q_mask = q_offs < S_len
        
        s = tl.where(kv_mask[None, :] & q_mask[:, None], s, -1e44)
        
        m_i_prev = m_i
        m_i = tl.maximum(m_i, tl.max(s, axis=1))
        curr_p = tl.exp(s - m_i[:, None])
        l_i = l_i * tl.exp(m_i_prev - m_i) + tl.sum(curr_p, axis=1)
        
        acc_O0 *= tl.exp(m_i_prev - m_i)[:, None]
        acc_O1 *= tl.exp(m_i_prev - m_i)[:, None]
        
        curr_p = curr_p.to(tl.bfloat16)
        
        v0 = v_desc.load([global_kv_row, 0])
        v1 = v_desc.load([global_kv_row, 64])
        
        acc_O0 += tl.dot(curr_p, v0)
        acc_O1 += tl.dot(curr_p, v1)
        
    q_offs = global_row + tl.arange(0, 32)
    q_mask = q_offs < S_len
    
    val0 = acc_O0 / l_i[:, None]
    val1 = acc_O1 / l_i[:, None]
    
    row_ptr_o = (chunk_bh * S_len + global_row) * s_D
    
    off_d0 = tl.arange(0, 64)
    out_ptr0 = O_ptr + row_ptr_o + q_offs[:, None] * s_D + off_d0[None, :]
    tl.store(out_ptr0, val0.to(tl.bfloat16), mask=q_mask[:, None])
    
    off_d1 = tl.arange(64, 128)
    out_ptr1 = O_ptr + row_ptr_o + q_offs[:, None] * s_D + off_d1[None, :]
    tl.store(out_ptr1, val1.to(tl.bfloat16), mask=q_mask[:, None])
    
    lse_ptr = LSE_ptr + chunk_bh * S_len + q_offs
    LSE_val = m_i + tl.log(l_i)
    tl.store(lse_ptr, LSE_val, mask=q_mask)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S_len, D = Q.shape
    
    scale = 1.0 / (D ** 0.5)
    
    def alloc_fn(size: int, alignment: int, stream):
        return torch.empty(size, device="cuda", dtype=torch.int8)
    
    triton.set_allocator(alloc_fn)
    
    grid = (triton.cdiv(S_len, 32), B, H)
    _attention_kernel[grid](Q, K, V, O, LSE, S_len, scale, H, num_warps=4)
    
    return O, LSE