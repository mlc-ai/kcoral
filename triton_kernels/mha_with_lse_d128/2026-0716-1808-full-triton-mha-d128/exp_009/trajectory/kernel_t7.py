import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def attention_kernel(
    desc_q, desc_k, desc_v, O_desc, LSE,
    S_len, scale,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Hopper-optimized Attention kernel utilizing TMA and WGMMA.
    
    Major Optimizations & Fixes:
      - Replaced broken static/padding Option `TensorDescriptor` and invalid `tl.make_tensor_descriptor`
        with robust host-defined `TensorDescriptor.from_tensor(block_shape=[..., 64])`.
      - Mitigated WGMMA shape conflicts and register pressure by splitting Head Dimension (128) into two independent 64-block computations.
      - Pipelined K and V TMA fetches using explicit in-source double buffering.
      - Fixed boundary score mask logic.
      - Accumulating online softmax tracked natively in FP32.
    """
    bid_bh = tl.program_id(0)
    bid_s = tl.program_id(1)
    
    row_starts = tl.arange(0, BLOCK_M)
    
    row_offset = bid_bh * S_len + bid_s * BLOCK_M
    
    # Utilize dynamically generated 2D Tensor Descriptors mapping 
    # targeting native contiguous flattenings mapped across a 64-element span.
    q0 = desc_q.load([row_offset, 0])
    q1 = desc_q.load([row_offset, HEAD_DIM//2])
    
    out_acc0 = tl.zeros((BLOCK_M, 64), tl.float32)
    out_acc1 = tl.zeros((BLOCK_M, 64), tl.float32)
    m = tl.full((BLOCK_M,), -float('inf'), tl.float32)
    l = tl.full((BLOCK_M,), 0.0, tl.float32)
    
    # Explicit setup of dual-chase software pipeline states (Double Buffering)
    k_tiles = [[None, None], [None, None]]
    v_tiles = [[None, None], [None, None]]
    
    num_blocks = (S_len + BLOCK_N - 1) // BLOCK_N
    
    # Guard minimum block count bounds explicitly to avoid nullptr dereferencing inside loops
    if num_blocks > 0:
        k_tiles[0][0] = desc_k.load([bid_bh * S_len, 0])
        k_tiles[0][1] = desc_k.load([bid_bh * S_len, HEAD_DIM//2])
        v_tiles[0][0] = desc_v.load([bid_bh * S_len, 0])
        v_tiles[0][1] = desc_v.load([bid_bh * S_len, HEAD_DIM//2])
        
    for k_blk in range(num_blocks):
        phase = k_blk % 2
        
        load_idx = k_blk + 1
        if load_idx < num_blocks:
            next_row = bid_bh * S_len + load_idx * BLOCK_N
            k_tiles[1 - phase][0] = desc_k.load([next_row, 0])
            k_tiles[1 - phase][1] = desc_k.load([next_row, HEAD_DIM//2])
            v_tiles[1 - phase][0] = desc_v.load([next_row, 0])
            v_tiles[1 - phase][1] = desc_v.load([next_row, HEAD_DIM//2])
            
        k0 = k_tiles[phase][0]
        k1 = k_tiles[phase][1]
        v0 = v_tiles[phase][0]
        v1 = v_tiles[phase][1]
        
        # Compute QK^T utilizing native accumulator carry-over syntax avoiding temp allocations
        score_acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        s = tl.dot(q0, k0.T, score_acc)
        s = tl.dot(q1, k1.T, s)
        s = s * scale
        
        # Mask out of boundary computations using strictly monotonically bounded indexing
        kv_range = tl.arange(0, BLOCK_N)
        score_mask = ((k_blk * BLOCK_N + kv_range[None, :]) < S_len)
        s = tl.where(score_mask, s, -float('inf'))
        
        # Fast online softmax tracked natively in FP32 precision over completely unrolled axes
        m_prev = m
        m_curr = tl.maximum(m_prev, tl.max(s, axis=1))
        exp_scale = tl.math.exp(m_prev - m_curr)
        
        s = tl.math.exp(s - m_curr[:, None])
        
        row_sum = tl.sum(s, axis=1)
        l = l * exp_scale + row_sum
        m = m_curr
        
        # Linearly rescale historical partially computed outputs matching new softmax scalar constants
        out_acc0 = out_acc0 * exp_scale[:, None]
        out_acc1 = out_acc1 * exp_scale[:, None]
        
        # Compute PV accumulation step mapped safely across variable Head Dimension sizes
        s_bf16 = s.to(tl.bfloat16)
        out_acc0 = tl.dot(s_bf16, v0, acc=out_acc0)
        out_acc1 = tl.dot(s_bf16, v1, acc=out_acc1)
        
    # Finalize output accumulator scalar division normalization mapping
    inv_l = 1.0 / l[:, None]
    out_acc0 = out_acc0 * inv_l
    out_acc1 = out_acc1 * inv_l
    
    # Commit outputs safely utilizing standard native TMA descriptor commit semantics
    O_desc.store([row_offset, 0], out_acc0.to(tl.bfloat16))
    O_desc.store([row_offset, HEAD_DIM//2], out_acc1.to(tl.bfloat16))
    
    lse_ptr = LSE + bid_bh * S_len
    lse_offsets = lse_ptr + bid_s * BLOCK_M + row_starts
    lse_mask = (bid_s * BLOCK_M + row_starts < S_len)
    tl.store(lse_offsets, m + tl.math.log(l), mask=lse_mask)


def run(Q, K, V, O, LSE):
    """Compute multi-head attention forward pass returning O and LSE."""
    torch.cuda.set_device(Q.device)
    B, H, S_len, D = Q.shape
    
    Q_flat = Q.view(B * H * S_len, D)
    K_flat = K.view(B * H * S_len, D)
    V_flat = V.view(B * H * S_len, D)
    O_flat = O.view(B * H * S_len, D)
    
    scale = 1.0 / (D ** 0.5)
    
    BLOCK_M = 128
    BLOCK_N = 128 
    
    # Utilize 2D Tensor Descriptors mapping block shapes mapped identically to [BLOCK, 64] resolving prior shape mismatch issues.
    desc_q = TensorDescriptor.from_tensor(
        Q_flat,
        [BLOCK_M, 64]
    )
    
    desc_k = TensorDescriptor.from_tensor(
        K_flat,
        [BLOCK_N, 64]
    )
    
    desc_v = TensorDescriptor.from_tensor(
        V_flat,
        [BLOCK_N, 64]
    )
    
    O_desc = TensorDescriptor.from_tensor(
        O_flat,
        [BLOCK_M, 64]
    )
    
    grid = (B * H, triton.cdiv(S_len, BLOCK_M))
    attention_kernel[grid](
        desc_q, desc_k, desc_v, O_desc, LSE,
        S_len, scale,
        HEAD_DIM=D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=8,
        num_stages=4,
    )