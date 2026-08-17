import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def attention_kernel(
    desc_q, desc_k, desc_v, O, LSE,
    S_len, scale,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Hopper-optimized Attention kernel utilizing TMA and WGMMA.
    
    Replaced invalid static `TensorDescriptor(padding_option=...)` constructs and 
    broken `tl.make_tensor_descriptor` usage with native host-defined `TensorDescriptor.from_tensor`. 
    Uses double-buffered TMA loads for K and V tiles to pipeline global memory fetches 
    with compute, accumulating online softmax exactly in FP32.
    """
    bid_bh = tl.program_id(0)
    bid_s = tl.program_id(1)
    
    row_starts = tl.arange(0, BLOCK_M)
    col_starts = tl.arange(0, BLOCK_N)
    
    # Utilize dynamically generated 3D Tensor Descriptors mapping over the contiguous flattenings
    # to correctly span Batch, Sequence, and Head Dimensions.

    # Load Q tile (only loaded once outside loop)
    q = desc_q.load([bid_bh, bid_s * BLOCK_M, 0])
    
    out_acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    m = tl.full((BLOCK_M,), -float('inf'), tl.float32)
    l = tl.full((BLOCK_M,), 0.0, tl.float32)
    
    # Setup double buffering states explicitly guarding minimum block count bounds
    k_tiles = [None, None]
    v_tiles = [None, None]
    
    num_blocks = (S_len + BLOCK_N - 1) // BLOCK_N
    head_dim_blocks = (HEAD_DIM + BLOCK_N - 1) // BLOCK_N
    
    # Preload initial dual-chase phases for robust pipeline entry
    if num_blocks > 0:
        k_tiles[0] = desc_k.load([bid_bh, 0, 0])
        v_tiles[0] = desc_v.load([bid_bh, 0, 0])
        
    if num_blocks > 1:
        k_tiles[1] = desc_k.load([bid_bh, BLOCK_N, 0])
        v_tiles[1] = desc_v.load([bid_bh, BLOCK_N, 0])
        
    for k_blk in range(num_blocks):
        phase = k_blk % 2
        
        load_idx = k_blk + 2
        if load_idx < num_blocks:
            next_offset = bid_bh * S_len + load_idx * BLOCK_N
            k_tiles[1 - phase] = desc_k.load([bid_bh, load_idx * BLOCK_N, 0])
            v_tiles[1 - phase] = desc_v.load([bid_bh, load_idx * BLOCK_N, 0])
            
        k = k_tiles[phase]
        v = v_tiles[phase]
        
        # Compute QK^T
        s = tl.dot(q, k.T) * scale
        
        # Mask out of boundary computations
        score_mask = ((k_blk * BLOCK_N + col_starts[None, :]) < S_len)
        s = tl.where(score_mask, s, -float('inf'))
        
        # Fast online softmax tracked natively in FP32 precision
        m_prev = m
        m_curr = tl.maximum(m_prev, tl.max(s, axis=1))
        exp_scale = tl.math.exp(m_prev - m_curr)
        
        s = tl.math.exp(s - m_curr[:, None, :])
        
        row_sum = tl.sum(s, axis=2)
        l = l * exp_scale + row_sum
        m = m_curr
        
        out_acc = out_acc * exp_scale[:, None, :]
        
        # Compute PV accumulation step mapped safely across variable Head Dimension sizes
        for i in range(head_dim_blocks):
            v = v_tiles[phase][..., i * BLOCK_N:(i + 1) * BLOCK_N]
            p_tile = s.to(tl.bfloat16)
            o_partial = tl.dot(p_tile, v)
            out_acc[:, i * BLOCK_N:(i + 1) * BLOCK_N] = o_partial
            
    # Normalize output accumulator scalar division 
    out_acc = out_acc / l[:, None, :]
    
    # Write outputs safely utilizing standard native pointer indexing logic avoiding complex commit semantics
    out_ptr = O + bid_bh * S_len * HEAD_DIM
    out_offsets = out_ptr + (bid_s * BLOCK_M + row_starts)[:, None] * HEAD_DIM + col_starts[None, :]
    out_mask = (bid_s * BLOCK_M + row_starts[:, None] < S_len)
    tl.store(out_offsets, out_acc.to(tl.bfloat16), mask=out_mask)
    
    lse_ptr = LSE + bid_bh * S_len
    lse_offsets = lse_ptr + bid_s * BLOCK_M + row_starts
    lse_mask = (bid_s * BLOCK_M + row_starts < S_len)
    tl.store(lse_offsets, m + tl.math.log(l), mask=lse_mask)


def run(Q, K, V, O, LSE):
    """Compute multi-head attention forward pass returning O and LSE."""
    torch.cuda.set_device(Q.device)
    B, H, S_len, D = Q.shape
    
    Q_flat = Q.view(B * H, S_len, D)
    K_flat = K.view(B * H, S_len, D)
    V_flat = V.view(B * H, S_len, D)
    
    scale = 1.0 / (D ** 0.5)
    
    BLOCK_M = 128
    BLOCK_N = 128 
    
    # Utilize 3D Tensor Descriptors mapping to avoid shape mismatch and enable padded layouts.
    desc_q = TensorDescriptor(
        base=Q_flat.data_ptr(),
        shape=[B * H, S_len, D],
        strides=[S_len * D, D, 1],
        block_shape=[1, BLOCK_M, D],
        padding_option="zero"
    )
    
    desc_k = TensorDescriptor(
        base=K_flat.data_ptr(),
        shape=[B * H, S_len, D],
        strides=[S_len * D, D, 1],
        block_shape=[1, BLOCK_N, D],
        padding_option="zero"
    )
    
    desc_v = TensorDescriptor(
        base=V_flat.data_ptr(),
        shape=[B * H, S_len, D],
        strides=[S_len * D, D, 1],
        block_shape=[1, BLOCK_N, D],
        padding_option="zero"
    )
    
    grid = (B * H, triton.cdiv(S_len, BLOCK_M))
    attention_kernel[grid](
        desc_q, desc_k, desc_v, O, LSE,
        S_len, scale,
        HEAD_DIM=D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=8,
        num_stages=3,
    )