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
    
    Uses a 64x64 tile for the attention score computation and software pipelining 
    for the Key and Value matrix blocks. The Head Dimension (D=128) is split into 
    two chunks of size 64 to align with the 16x16x16 WGMMA instruction shape.
    
    Grid mapping: (ceil(S/64), B, H)
    - Axis 0: Query block index (each CTA handles 64 query rows)
    - Axis 1: Batch index
    - Axis 2: Attention head index
    """
    
    q_blk = tl.program_id(0)
    b = tl.program_id(1)
    h = tl.program_id(2)
    
    global_row = q_blk * 64
    b_h = b * H + h
    
    row_ptr_q = (b_h * S_len + global_row) * 128
    q_offs_flat = global_row + tl.arange(0, 64)
    
    off_D = tl.arange(0, 64)
    off_D1 = off_D + 64
    
    Q_tile0 = tl.load(Q_ptr + row_ptr_q + q_offs_flat[:, None] * 128 + off_D[None, :],
                     mask=(q_offs_flat[:, None] < S_len), other=0.0)
    Q_tile1 = tl.load(Q_ptr + row_ptr_q + 64 + q_offs_flat[:, None] * 128 + off_D[None, :],
                     mask=(q_offs_flat[:, None] < S_len), other=0.0)
    
    acc_O0 = tl.zeros((64, 64), tl.float32)
    acc_O1 = tl.zeros((64, 64), tl.float32)
    m_i = tl.full((64,), -1e38, tl.float32)
    l_i = tl.zeros((64,), 1.0, tl.float32)
    
    num_kv_iters = tl.cdiv(S_len, 64)
    
    for kv_blk in tl.range(0, num_kv_iters, num_stages=2):
        kv_row = kv_blk * 64
        
        row_ptr_kv = (b_h * S_len + kv_row) * 128
        kv_offs_flat = kv_row + tl.arange(0, 64)
        
        K_tile0 = tl.load(K_ptr + row_ptr_kv + kv_offs_flat[:, None] * 128 + off_D[None, :],
                         mask=(kv_offs_flat[:, None] < S_len), other=0.0)
        K_tile1 = tl.load(K_ptr + row_ptr_kv + 64 + kv_offs_flat[:, None] * 128 + off_D[None, :],
                         mask=(kv_offs_flat[:, None] < S_len), other=0.0)
        
        s = tl.dot(Q_tile0, K_tile0.T) + tl.dot(Q_tile1, K_tile1.T)
        s = s * scale
        
        kv_mask = kv_offs_flat < S_len
        q_mask = q_offs_flat < S_len
        s = tl.where(kv_mask[None, :] & q_mask[:, None], s, -1e44)
        
        m_i_prev = m_i
        m_i = tl.maximum(m_i, tl.max(s, axis=1))
        curr_p = tl.exp(s - m_i[:, None])
        l_i = l_i * tl.exp(m_i_prev - m_i) + tl.sum(curr_p, axis=1)
        
        acc_O0 = acc_O0 * tl.exp(m_i_prev - m_i)[:, None]
        acc_O1 = acc_O1 * tl.exp(m_i_prev - m_i)[:, None]
        
        V_tile0 = tl.load(V_ptr + row_ptr_kv + kv_offs_flat[:, None] * 128 + off_D[None, :],
                         mask=(kv_offs_flat[:, None] < S_len), other=0.0)
        V_tile1 = tl.load(V_ptr + row_ptr_kv + 64 + kv_offs_flat[:, None] * 128 + off_D[None, :],
                         mask=(kv_offs_flat[:, None] < S_len), other=0.0)
        
        acc_O0 += tl.dot(curr_p, V_tile0)
        acc_O1 += tl.dot(curr_p, V_tile1)
        
    acc_O0 = acc_O0 / l_i[:, None]
    acc_O1 = acc_O1 / l_i[:, None]
    
    o_ptr0 = O_ptr + (b_h * S_len + global_row) * 128 + off_D[None, :] + q_offs_flat[:, None] * 128
    tl.store(o_ptr0, acc_O0.to(tl.bfloat16), mask=q_mask[:, None])
    
    o_ptr1 = O_ptr + (b_h * S_len + global_row) * 128 + 64 + off_D[None, :] + q_offs_flat[:, None] * 128
    tl.store(o_ptr1, acc_O1.to(tl.bfloat16), mask=q_mask[:, None])
    
    lse_ptr = LSE_ptr + (b_h * S_len) + q_offs_flat
    LSE_val = m_i + tl.log(l_i)
    tl.store(lse_ptr, LSE_val, mask=q_mask)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S_len, D = Q.shape
    
    scale = 1.0 / (D ** 0.5)
    
    grid = (triton.cdiv(S_len, 64), B, H)
    _attention_kernel[grid](Q, K, V, O, LSE, S_len, scale, H, num_warps=4)
    
    return O, LSE