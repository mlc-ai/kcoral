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
    BLOCK_K: tl.constexpr,
):
    """
    Hopper-optimized Attention kernel.
    
    Features:
      - TMA loads for Q, K, V
      - WGMMA for QK^T and PV
      - Double buffered K/V pipeline
      - Fast online softmax
    """
    bid_bh = tl.program_id(0)
    bid_s = tl.program_id(1)
    
    offset_bh = bid_bh
    row_starts = tl.arange(0, BLOCK_M)
    col_offsets = tl.arange(0, BLOCK_N)
    
    # Load Q tile
    q = desc_q.load([offset_bh, bid_s * BLOCK_M, 0])
    
    # Initialize accumulators
    out_acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    m = tl.full((BLOCK_M,), -float('inf'), tl.float32)
    l = tl.full((BLOCK_M,), 0.0, tl.float32)
    
    # Double buffering arrays
    k_tiles = [desc_k.load([offset_bh, 0, 0])]
    v_tiles = [desc_v.load([offset_bh, 0, 0])]
    
    num_blocks = (S_len + BLOCK_K - 1) // BLOCK_K
    
    for k_blk in range(num_blocks):
        phase = k_blk % 2
        
        if k_blk + 1 < num_blocks:
            k_tiles[1 - phase] = desc_k.load([offset_bh, (k_blk + 1) * BLOCK_K, 0])
            v_tiles[1 - phase] = desc_v.load([offset_bh, (k_blk + 1) * BLOCK_K, 0])
            
        k = k_tiles[phase]
        v = v_tiles[phase]
        
        # Compute QK^T
        score_acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        s = tl.dot(q, k.T, score_acc)
        s = s * scale
        
        # Mask out of boundary computations
        score_mask = (k_blk * BLOCK_K + col_offsets[None, :] < S_len)
        s = tl.where(score_mask, s, -float('inf'))
        
        # Fast softmax
        m_prev = m
        m_curr = tl.maximum(m, tl.max(s, axis=1))
        exp_scale = tl.math.exp(m_prev - m_curr)
        
        s = tl.math.exp(s - m_curr[:, None])
        
        row_sum = tl.sum(s, axis=1)
        l = l * exp_scale + row_sum
        m = m_curr
        
        out_acc = out_acc * exp_scale[:, None]
        
        # Compute PV
        s_bf16 = s.to(tl.bfloat16)
        out_acc = tl.dot(s_bf16, v, acc=out_acc)
        
    # Normalize
    out_acc = out_acc / l[:, None]
    
    # Save outputs
    out_ptr = O + offset_bh * S_len * HEAD_DIM
    out_offsets = out_ptr + (bid_s * BLOCK_M + row_starts)[:, None] * HEAD_DIM + col_offsets[None, :]
    out_mask = (bid_s * BLOCK_M + row_starts[:, None] < S_len)
    tl.store(out_offsets, out_acc.to(tl.bfloat16), mask=out_mask)
    
    lse_ptr = LSE + offset_bh * S_len
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
    BLOCK_K = 128
    
    # Tensor Descriptors setup
    # Strides mapped from standard contiguous PyTorch tensors.
    # For Q/K/V_flat (shape [BH, S, D]):
    #   - Last dim (D) stride is 1
    #   - Middle dim (S) stride is D
    #   - First dim (BH) stride is S * D
    desc_q = TensorDescriptor(
        base=Q_flat.data_ptr(),
        shape=[B * H, S_len, D],
        strides=[S_len * D, D, 1],
        block_shape=[1, BLOCK_M, HEAD_DIM],
        padding_option="zero"
    )
    
    desc_k = TensorDescriptor(
        base=K_flat.data_ptr(),
        shape=[B * H, S_len, D],
        strides=[S_len * D, D, 1],
        block_shape=[1, BLOCK_K, HEAD_DIM],
        padding_option="zero"
    )
    
    desc_v = TensorDescriptor(
        base=V_flat.data_ptr(),
        shape=[B * H, S_len, D],
        strides=[S_len * D, D, 1],
        block_shape=[1, BLOCK_K, HEAD_DIM],
        padding_option="zero"
    )
    
    grid = (B * H, triton.cdiv(S_len, BLOCK_M))
    attention_kernel[grid](
        desc_q, desc_k, desc_v, O, LSE,
        S_len, scale,
        HEAD_DIM=D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=2,
    )