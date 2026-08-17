import torch
import triton
import triton.language as tl


@triton.jit
def _attention_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    S_len, scale, H,
):
    """
    Hopper-optimized FlashAttention kernel utilizing TMA and WGMMA execution paths.
    
    Architecture overview:
    - Grid mapping: (ceil(S/64), B, H)
    - Each CTA processes a unique 64-row Query block across the sequence length dimension.
    - Key and Value blocks span 128 sequence positions. Each block is split into 4 contiguous chunks 
      of width 32 to maintain optimal alignment with Hopper's WGMMA instruction shapes.
    - The complete Head Dimension (D = 128) is preserved and chunked identically to K and V.
    
    Execution Pipeline & Register Management:
    1. Q chunks are asynchronously loaded via TMA and parked in static registers for the lifetime 
       of the CTA.
    2. A single `tl.range` software-pipelined loop with `num_stages=2` fetches K and V chunks 
       concurrently while computing the prior iteration's GEMMs.
    3. The Attention Score matrix (S) and Softmax Probabilities (P) logically occupy [64, 128] FP32 
       elements. Through compiler register aliasing, their combined footprint is ~64 registers/thread.
    4. Four Output Accumulators (acc_O) track the [64, 32] sub-blocks of the final O tensor, 
       consuming an additional ~64 registers/thread.
    5. Peak theoretical register pressure rests comfortably around 130 registers/thread, leaving 
       ample headroom well below the 255 limit for robust occupancy on Hopper.
    """
    
    q_blk = tl.program_id(0)
    b = tl.program_id(1)
    h = tl.program_id(2)
    
    global_row = q_blk * 64
    chunk_bh = b * H + h
    
    s_D = 128
    
    # Local dynamic TMA descriptors scoped explicitly to the exact batch/head slice. 
    # This guarantees strictly 2D block tensors ([Rows, Chunks]) for seamless WGMMA consumption.
    q_ptr_bh = Q_ptr + chunk_bh * S_len * s_D
    q_desc = tl.make_tensor_descriptor(
        q_ptr_bh, shape=[S_len, s_D], strides=[s_D, 1],
        block_shape=[64, 32], padding_option="zero")
    
    k_ptr_bh = K_ptr + chunk_bh * S_len * s_D
    k_desc = tl.make_tensor_descriptor(
        k_ptr_bh, shape=[S_len, s_D], strides=[s_D, 1],
        block_shape=[128, 32], padding_option="zero")
    
    v_ptr_bh = V_ptr + chunk_bh * S_len * s_D
    v_desc = tl.make_tensor_descriptor(
        v_ptr_bh, shape=[S_len, s_D], strides=[s_D, 1],
        block_shape=[128, 32], padding_option="zero")
    
    # Extracted synchronously before mainloop onset. Reused natively in subsequent fused dot products.
    q0 = q_desc.load([global_row, 0])
    q1 = q_desc.load([global_row, 32])
    q2 = q_desc.load([global_row, 64])
    q3 = q_desc.load([global_row, 96])
    
    acc_O = [tl.zeros((64, 32), tl.float32) for _ in range(4)]
    m_i = tl.full((64,), -1e38, tl.float32)
    l_i = tl.zeros((64,), 1.0, tl.float32)
    
    q_offs = global_row + tl.arange(0, 64)
    q_mask = q_offs < S_len
    
    # Iterating sequentially across the complete Sequence length (outer context loop bounds)
    num_kv_iters = tl.cdiv(S_len, 128)
    
    for kv_blk in tl.range(0, num_kv_iters, num_stages=2):
        global_kv_row = kv_blk * 128
        
        k0 = k_desc.load([global_kv_row, 0])
        k1 = k_desc.load([global_kv_row, 32])
        k2 = k_desc.load([global_kv_row, 64])
        k3 = k_desc.load([global_kv_row, 96])
        
        # Fused cross-chunk reductions aggregated cohesively in 32-element contiguous strips mapping cleanly over to native WGMMA
        s = (tl.dot(q0, k0.T) + tl.dot(q1, k1.T) + 
             tl.dot(q2, k2.T) + tl.dot(q3, k3.T)) * scale
        
        kv_offs = global_kv_row + tl.arange(0, 128)
        kv_mask = kv_offs < S_len
        
        s = tl.where(kv_mask[None, :] & q_mask[:, None], s, -1e44)
        
        m_i_prev = m_i
        m_i = tl.maximum(m_i, tl.max(s, axis=1))
        curr_p = tl.exp(s - m_i[:, None])
        l_i = l_i * tl.exp(m_i_prev - m_i) + tl.sum(curr_p, axis=1)
        
        for i in range(4):
            acc_O[i] *= tl.exp(m_i_prev - m_i)[:, None]
        
        v0 = v_desc.load([global_kv_row, 0])
        v1 = v_desc.load([global_kv_row, 32])
        v2 = v_desc.load([global_kv_row, 64])
        v3 = v_desc.load([global_kv_row, 96])
        
        acc_O[0] += tl.dot(curr_p, v0)
        acc_O[1] += tl.dot(curr_p, v1)
        acc_O[2] += tl.dot(curr_p, v2)
        acc_O[3] += tl.dot(curr_p, v3)
        
    q_offs_out = global_row + tl.arange(0, 64)
    row_ptr_o = (chunk_bh * S_len + global_row) * s_D
    
    for i in range(4):
        val = acc_O[i] / l_i[:, None]
        off_c = i * 32 + tl.arange(0, 32)
        out_ptr = O_ptr + row_ptr_o + q_offs_out[:, None] * s_D + off_c[None, :]
        tl.store(out_ptr, val.to(tl.bfloat16), mask=q_mask[:, None])
    
    lse_ptr = LSE_ptr + (chunk_bh * S_len) + q_offs_out
    LSE_val = m_i + tl.log(l_i)
    tl.store(lse_ptr, LSE_val, mask=q_mask)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S_len, D = Q.shape
    
    scale = 1.0 / (D ** 0.5)
    
    def alloc_fn(size: int, alignment: int, stream):
        return torch.empty(size, device="cuda", dtype=torch.int8)
    
    triton.set_allocator(alloc_fn)
    
    grid = (triton.cdiv(S_len, 64), B, H)
    _attention_kernel[grid](Q, K, V, O, LSE, S_len, scale, H, num_warps=4)
    
    return O, LSE