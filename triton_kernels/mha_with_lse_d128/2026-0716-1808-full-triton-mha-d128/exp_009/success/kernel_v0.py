import torch
import triton
import triton.language as tl


@triton.jit
def attention_kernel(
    Q, K, V, O, LSE,
    S_len, scale,
    HEAD_DIM: tl.constexpr,
):
    """
    Attention kernel using FlashAttention algorithm adapted for head dimension 128.
    
    Grid mapping:
      - program_id(0): block row index for S dimension (each block is 128 rows)
      - program_id(1): combined batch * head index (B*H = 4*48 = 192)
      
    Execution model:
      - 4 warps (128 threads)
      - Each CTA computes a 128x128 tile of the output
      - Iterates sequentially over K blocks of size 128
      - Online softmax tracked through scalar m (max) and l (sum)
    """
    bid_s = tl.program_id(0)
    bid_bh = tl.program_id(1)
    
    row_starts = bid_s * 128 + tl.arange(0, 128)
    col_starts = tl.arange(0, 128)
    
    q_ptr = Q + bid_bh * S_len * HEAD_DIM
    k_ptr = K + bid_bh * S_len * HEAD_DIM
    v_ptr = V + bid_bh * S_len * HEAD_DIM
    
    q_offsets = q_ptr + row_starts[:, None] * HEAD_DIM + col_starts[None, :]
    q_mask = (row_starts < S_len)[:, None]
    q_tile = tl.load(q_offsets, mask=q_mask, other=0.0)
    
    o_acc = tl.zeros((128, 128), tl.float32)
    
    m = tl.full((128,), -float('inf'), tl.float32)
    l = tl.full((128,), 0.0, tl.float32)
    
    for k_blk in range(0, S_len, 128):
        kv_offsets = k_ptr + (k_blk + col_starts[:, None]) * HEAD_DIM + col_starts[None, :]
        kv_mask = (k_blk + col_starts < S_len)[:, None]
        k_tile = tl.load(kv_offsets, mask=kv_mask, other=0.0)
        
        vv_offsets = v_ptr + (k_blk + col_starts[:, None]) * HEAD_DIM + col_starts[None, :]
        v_tile = tl.load(vv_offsets, mask=kv_mask, other=0.0)
        
        s_tile = tl.dot(q_tile, k_tile.T) * scale
        
        score_mask = (k_blk + col_starts < S_len)[None, :]
        s_tile = tl.where(score_mask, s_tile, -float('inf'))
        
        row_max = tl.max(s_tile, axis=1)
        curr_max = tl.maximum(m, row_max)
        prev_max = m
        
        exp_scale = tl.math.exp(prev_max - curr_max)
        s_tile = tl.math.exp(s_tile - curr_max[:, None])
        
        row_sum = tl.sum(s_tile, axis=1)
        l = l * exp_scale + row_sum
        m = curr_max
        o_acc = o_acc * exp_scale[:, None]
        
        p_tile = s_tile.to(tl.bfloat16)
        o_acc = tl.dot(p_tile, v_tile, acc=o_acc)
    
    o_acc = o_acc / l[:, None]
    
    out_ptr = O + bid_bh * S_len * HEAD_DIM
    out_offsets = out_ptr + row_starts[:, None] * HEAD_DIM + col_starts[None, :]
    out_mask = (row_starts < S_len)[:, None]
    tl.store(out_offsets, o_acc.to(tl.bfloat16), mask=out_mask)
    
    lse_ptr = LSE + bid_bh * S_len
    lse_offsets = lse_ptr + row_starts
    lse_mask = (row_starts < S_len)
    tl.store(lse_offsets, m + tl.math.log(l), mask=lse_mask)


def run(Q, K, V, O, LSE):
    """Compute multi-head attention forward pass returning O and LSE."""
    torch.cuda.set_device(Q.device)
    S_len = Q.shape[2]
    scale = 1.0 / (128 ** 0.5)
    
    grid = (triton.cdiv(S_len, 128), 4 * 48)
    attention_kernel[grid](Q, K, V, O, LSE, S_len, scale, HEAD_DIM=128, num_warps=4, num_stages=2)