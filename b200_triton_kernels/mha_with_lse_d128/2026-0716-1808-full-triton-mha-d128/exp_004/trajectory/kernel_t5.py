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
    
    Uses a 32x32 tile for the attention score computation and software pipelining 
    for the Key and Value matrix blocks. The Head Dimension (D=128) is split into 
    four independent chunks of width 32.
    
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
    
    q_offs = global_row + tl.arange(0, 32)
    off_d = tl.arange(0, 32)
    
    row_ptr_q = chunk_bh * S_len * s_D + global_row * s_D
    
    Q_tile0 = tl.load(Q_ptr + row_ptr_q + q_offs[:, None] * s_D + off_d[None, :],
                     mask=(q_offs[:, None] < S_len), other=0.0)
    Q_tile1 = tl.load(Q_ptr + row_ptr_q + 32 + q_offs[:, None] * s_D + off_d[None, :],
                     mask=(q_offs[:, None] < S_len), other=0.0)
    Q_tile2 = tl.load(Q_ptr + row_ptr_q + 64 + q_offs[:, None] * s_D + off_d[None, :],
                     mask=(q_offs[:, None] < S_len), other=0.0)
    Q_tile3 = tl.load(Q_ptr + row_ptr_q + 96 + q_offs[:, None] * s_D + off_d[None, :],
                     mask=(q_offs[:, None] < S_len), other=0.0)
    
    acc_O = [tl.zeros((32, 32), tl.float32) for _ in range(4)]
    m_i = tl.full((32,), -1e38, tl.float32)
    l_i = tl.zeros((32,), 1.0, tl.float32)
    
    num_kv_iters = tl.cdiv(S_len, 32)
    
    for kv_blk in tl.range(0, num_kv_iters, num_stages=2):
        global_kv_row = kv_blk * 32
        row_ptr_kv = chunk_bh * S_len * s_D + global_kv_row * s_D
        kv_offs = global_kv_row + tl.arange(0, 32)
        
        K_tile0 = tl.load(K_ptr + row_ptr_kv + kv_offs[:, None] * s_D + off_d[None, :],
                         mask=(kv_offs[:, None] < S_len), other=0.0)
        K_tile1 = tl.load(K_ptr + row_ptr_kv + 32 + kv_offs[:, None] * s_D + off_d[None, :],
                         mask=(kv_offs[:, None] < S_len), other=0.0)
        K_tile2 = tl.load(K_ptr + row_ptr_kv + 64 + kv_offs[:, None] * s_D + off_d[None, :],
                         mask=(kv_offs[:, None] < S_len), other=0.0)
        K_tile3 = tl.load(K_ptr + row_ptr_kv + 96 + kv_offs[:, None] * s_D + off_d[None, :],
                         mask=(kv_offs[:, None] < S_len), other=0.0)
        
        s = (tl.dot(Q_tile0, K_tile0.T) + tl.dot(Q_tile1, K_tile1.T) + 
             tl.dot(Q_tile2, K_tile2.T) + tl.dot(Q_tile3, K_tile3.T)) * scale
        
        kv_mask = kv_offs < S_len
        q_mask = q_offs < S_len
        s = tl.where(kv_mask[None, :] & q_mask[:, None], s, -1e44)
        
        m_i_prev = m_i
        m_i = tl.maximum(m_i, tl.max(s, axis=1))
        curr_p = tl.exp(s - m_i[:, None])
        l_i = l_i * tl.exp(m_i_prev - m_i) + tl.sum(curr_p, axis=1)
        
        for i in range(4):
            acc_O[i] *= tl.exp(m_i_prev - m_i)[:, None]
        
        curr_p = curr_p.to(tl.bfloat16)
        
        V_tile0 = tl.load(V_ptr + row_ptr_kv + kv_offs[:, None] * s_D + off_d[None, :],
                         mask=(kv_offs[:, None] < S_len), other=0.0)
        V_tile1 = tl.load(V_ptr + row_ptr_kv + 32 + kv_offs[:, None] * s_D + off_d[None, :],
                         mask=(kv_offs[:, None] < S_len), other=0.0)
        V_tile2 = tl.load(V_ptr + row_ptr_kv + 64 + kv_offs[:, None] * s_D + off_d[None, :],
                         mask=(kv_offs[:, None] < S_len), other=0.0)
        V_tile3 = tl.load(V_ptr + row_ptr_kv + 96 + kv_offs[:, None] * s_D + off_d[None, :],
                         mask=(kv_offs[:, None] < S_len), other=0.0)
        
        acc_O[0] += tl.dot(curr_p, V_tile0)
        acc_O[1] += tl.dot(curr_p, V_tile1)
        acc_O[2] += tl.dot(curr_p, V_tile2)
        acc_O[3] += tl.dot(curr_p, V_tile3)
        
    q_mask = q_offs < S_len
    
    for i in range(4):
        val = acc_O[i] / l_i[:, None]
        out_ptr = O_ptr + row_ptr_q + q_offs[:, None] * s_D + (i * 32 + off_d)[None, :]
        tl.store(out_ptr, val.to(tl.bfloat16), mask=q_mask[:, None])
    
    lse_ptr = LSE_ptr + chunk_bh * S_len + q_offs
    LSE_val = m_i + tl.log(l_i)
    tl.store(lse_ptr, LSE_val, mask=q_mask)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S_len, D = Q.shape
    
    scale = 1.0 / (D ** 0.5)
    
    grid = (triton.cdiv(S_len, 32), B, H)
    _attention_kernel[grid](Q, K, V, O, LSE, S_len, scale, H, num_warps=4)
    
    return O, LSE