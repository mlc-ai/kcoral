import torch
import triton
import triton.language as tl


@triton.jit
def _attention_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    S_len, scale, H: tl.constexpr,
):
    """
    Optimized FlashAttention kernel for non-causal multi-head attention.
    
    Implements the online softmax algorithm (FlashAttention) with:
    - 32x32 tile sizes for Q/K/V blocks
    - Full head dimension (D=128) loaded per tile
    - Software pipelining with num_stages=2
    - FP32 accumulation for numerical stability
    
    Grid mapping: (ceil(S/32), H, B)
    - Axis 0: Query block index (each CTA handles 32 query rows)
    - Axis 1: Attention head index
    - Axis 2: Batch index
    """
    
    i_c = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)
    
    s_D = 128
    global_row = i_c * 32
    
    # Query row offsets and validity mask
    q_offs = global_row + tl.arange(0, 32)
    q_mask = q_offs < S_len
    
    # Efficient 2D load of the Q tile with boundary masking
    row_ptr_q = (b * H + h) * S_len + q_offs
    off_D = tl.arange(0, s_D)
    Q_tile = tl.load(Q_ptr + row_ptr_q[:, None] * s_D + off_D[None, :],
                     mask=q_mask[:, None], other=0.0)
    
    # Per-row running statistics and output accumulator (kept in FP32)
    acc_O = tl.zeros((32, s_D), tl.float32)
    m_i = tl.full((32,), -1e38, tl.float32)
    l_i = tl.zeros((32,), 1.0, tl.float32)
    
    num_kv_iters = tl.cdiv(S_len, 32)
    
    for i_j in tl.range(0, num_kv_iters, 1, num_stages=2):
        kv_offs = i_j * 32 + tl.arange(0, 32)
        kv_mask = kv_offs < S_len
        
        row_ptr_kv = (b * H + h) * S_len + kv_offs
        
        K_tile = tl.load(K_ptr + row_ptr_kv[:, None] * s_D + off_D[None, :],
                         mask=kv_mask[:, None], other=0.0)
        V_tile = tl.load(V_ptr + row_ptr_kv[:, None] * s_D + off_D[None, :],
                         mask=kv_mask[:, None], other=0.0)
        
        S = tl.dot(Q_tile, K_tile.T) * scale
        
        # Mask out-of-bounds query/key interactions (-1e44 ensures exp underflows to 0)
        S = tl.where(kv_mask[None, :] & q_mask[:, None], S, -1e44)
        
        m_i_prev = m_i
        m_i = tl.maximum(m_i, tl.max(S, axis=1))
        curr_p = tl.exp(S - m_i[:, None])
        l_i = l_i * tl.exp(m_i_prev - m_i) + tl.sum(curr_p, axis=1)
        
        acc_O = acc_O * tl.exp(m_i_prev - m_i)[:, None]
        acc_O += tl.dot(curr_p.to(tl.bfloat16), V_tile)
    
    acc_O = acc_O / l_i[:, None]
    
    out_ptr = O_ptr + row_ptr_q[:, None] * s_D + off_D[None, :]
    tl.store(out_ptr, acc_O.to(tl.bfloat16), mask=q_mask[:, None])
    
    lse_ptr = LSE_ptr + (b * H + h) * S_len + q_offs
    LSE_val = m_i + tl.log(l_i)
    tl.store(lse_ptr, LSE_val, mask=q_mask)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S_len, D = Q.shape
    
    scale = 1.0 / (D ** 0.5)
    
    grid = (triton.cdiv(S_len, 32), H, B)
    _attention_kernel[grid, num_warps=4](Q, K, V, O, LSE, S_len, scale, H=H)
    
    return O, LSE