import torch
import triton
import triton.language as tl


@triton.jit
def _attention_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    S_len, scale, H,
):
    """
    Hopper-optimized FlashAttention kernel targeting WGMMA.
    
    Uses a 16x16 tile for the attention score computation and software pipelining 
    for the Key and Value matrix blocks. The Head Dimension (D=128) is split into 
    two chunks of size 64 to align with the 16x16x16 WGMMA instruction shape.
    
    Grid mapping: (ceil(S/16), B, H)
    - Axis 0: Query block index (each CTA handles 16 query rows)
    - Axis 1: Batch index
    - Axis 2: Attention head index
    """
    
    q_blk = tl.program_id(0)
    b = tl.program_id(1)
    h = tl.program_id(2)
    
    s_D = 128
    global_row = q_blk * 16
    bh = b * H + h
    
    q_desc = tl.make_tensor_descriptor(
        Q_ptr, shape=[4 * H, S_len, s_D], strides=[S_len * s_D, s_D, 1],
        block_shape=[1, 16, 64], padding_option="zero"
    )
    k_desc = tl.make_tensor_descriptor(
        K_ptr, shape=[4 * H, S_len, s_D], strides=[S_len * s_D, s_D, 1],
        block_shape=[1, 16, 64], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        V_ptr, shape=[4 * H, S_len, s_D], strides=[S_len * s_D, s_D, 1],
        block_shape=[1, 16, 64], padding_option="zero"
    )
    
    q0 = q_desc.load([bh, global_row, 0])
    q1 = q_desc.load([bh, global_row, 64])
    
    acc_O0 = tl.zeros((16, 64), tl.float32)
    acc_O1 = tl.zeros((16, 64), tl.float32)
    m_i = tl.full((16,), -1e38, tl.float32)
    l_i = tl.zeros((16,), 1.0, tl.float32)
    
    for kv_blk in range(S_len // 16):
        kv_row = kv_blk * 16
        
        k0 = k_desc.load([bh, kv_row, 0])
        k1 = k_desc.load([bh, kv_row, 64])
        
        s = tl.dot(q0, k0.T) + tl.dot(q1, k1.T)
        s = s * scale
        
        m_i_prev = m_i
        m_i = tl.maximum(m_i, tl.max(s, axis=1))
        curr_p = tl.exp(s - m_i[:, None])
        curr_p = curr_p.to(tl.bfloat16)
        l_i = l_i * tl.exp(m_i_prev - m_i) + tl.sum(curr_p, axis=1)
        
        acc_O0 = acc_O0 * tl.exp(m_i_prev - m_i)[:, None]
        acc_O1 = acc_O1 * tl.exp(m_i_prev - m_i)[:, None]
        
        v0 = v_desc.load([bh, kv_row, 0])
        v1 = v_desc.load([bh, kv_row, 64])
        
        acc_O0 += tl.dot(curr_p, v0)
        acc_O1 += tl.dot(curr_p, v1)
        
    acc_O0 = acc_O0 / l_i[:, None]
    acc_O1 = acc_O1 / l_i[:, None]
    
    q_offs = global_row + tl.arange(0, 16)
    q_mask = q_offs < S_len
    
    off_0 = tl.arange(0, 64)
    off_1 = tl.arange(64, 128)
    
    o_ptr0 = O_ptr + (bh * S_len + q_offs[:, None]) * s_D + off_0[None, :]
    tl.store(o_ptr0, acc_O0.to(tl.bfloat16), mask=q_mask[:, None])
    
    o_ptr1 = O_ptr + (bh * S_len + q_offs[:, None]) * s_D + off_1[None, :]
    tl.store(o_ptr1, acc_O1.to(tl.bfloat16), mask=q_mask[:, None])
    
    lse_ptr = LSE_ptr + (bh * S_len) + q_offs
    LSE_val = m_i + tl.log(l_i)
    tl.store(lse_ptr, LSE_val, mask=q_mask)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S_len, D = Q.shape
    
    scale = 1.0 / (D ** 0.5)
    
    def alloc_fn(size: int, alignment: int, stream):
        return torch.empty(size, device="cuda", dtype=torch.int8)
    
    triton.set_allocator(alloc_fn)
    
    grid = (triton.cdiv(S_len, 16), B, H)
    _attention_kernel[grid](Q, K, V, O, LSE, S_len, scale, H, num_warps=4, num_stages=2)
    
    return O, LSE