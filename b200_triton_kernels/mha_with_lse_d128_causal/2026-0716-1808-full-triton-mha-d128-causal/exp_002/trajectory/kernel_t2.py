import torch
import triton
import triton.language as tl


@triton.jit
def _attention_forward_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    lse_ptr,
    S,
    HEAD_DIM: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)
    
    b_h = pid_b * 48 + pid_h
    
    row_idx = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    valid_rows = row_idx < S
    
    q_base = q_ptr + b_h * S * HEAD_DIM
    k_base = k_ptr + b_h * S * HEAD_DIM
    v_base = v_ptr + b_h * S * HEAD_DIM
    o_base = o_ptr + b_h * S * HEAD_DIM
    
    m_i = tl.full([BLOCK_S], -float("inf"), dtype=tl.float32)
    D_i = tl.full([BLOCK_S], 0.0, dtype=tl.float32)
    
    out_acc0 = tl.zeros([BLOCK_S, HEAD_DIM // 2], dtype=tl.float32)
    out_acc1 = tl.zeros([BLOCK_S, HEAD_DIM // 2], dtype=tl.float32)
    
    scale = 1.0 / (HEAD_DIM ** 0.5)
    
    col_chunk0 = tl.arange(0, HEAD_DIM // 2)
    col_chunk1 = (HEAD_DIM // 2) + tl.arange(0, HEAD_DIM // 2)
    
    q0 = tl.load(q_base + row_idx[:, None] * HEAD_DIM + col_chunk0[None, :], mask=(row_idx[:, None] < S), other=0.0)
    q1 = tl.load(q_base + row_idx[:, None] * HEAD_DIM + col_chunk1[None, :], mask=(row_idx[:, None] < S), other=0.0)
    
    for k_tile in range(0, pid_s + 1):
        k_tile_row = k_tile * BLOCK_S + tl.arange(0, BLOCK_S)
        
        k0 = tl.load(k_base + k_tile_row[:, None] * HEAD_DIM + col_chunk0[None, :], mask=(k_tile_row[:, None] < S), other=0.0)
        k1 = tl.load(k_base + k_tile_row[:, None] * HEAD_DIM + col_chunk1[None, :], mask=(k_tile_row[:, None] < S), other=0.0)
        
        p = tl.dot(q0, k0.T)
        p = tl.dot(q1, k1.T, p)
        
        p = p * scale
        
        col_idx = k_tile_row
        mask = (row_idx[:, None] >= col_idx[None, :]) & (col_idx[None, :] < S)
        p = tl.where(mask, p, -float('inf'))
        
        curr_max = tl.maximum(m_i, tl.max(p, axis=1))
        exp_diff = tl.exp(m_i - curr_max)
        D_i = D_i * exp_diff
        D_i += tl.sum(tl.exp(p - curr_max[:, None]), axis=1)
        m_i = curr_max
        
        out_acc0 = out_acc0 * exp_diff[:, None]
        out_acc1 = out_acc1 * exp_diff[:, None]
        
        v0 = tl.load(v_base + k_tile_row[:, None] * HEAD_DIM + col_chunk0[None, :], mask=(k_tile_row[:, None] < S), other=0.0)
        v1 = tl.load(v_base + k_tile_row[:, None] * HEAD_DIM + col_chunk1[None, :], mask=(k_tile_row[:, None] < S), other=0.0)
        
        out = tl.exp(p - m_i[:, None])
        
        out_acc0 += tl.dot(out.to(tl.bfloat16), v0)
        out_acc1 += tl.dot(out.to(tl.bfloat16), v1)
        
    final_out0 = out_acc0.to(tl.bfloat16)
    final_out1 = out_acc1.to(tl.bfloat16)
    
    c0 = o_base + row_idx[:, None] * HEAD_DIM + col_chunk0[None, :]
    c1 = o_base + row_idx[:, None] * HEAD_DIM + col_chunk1[None, :]
    
    tl.store(c0, final_out0, mask=valid_rows[:, None])
    tl.store(c1, final_out1, mask=valid_rows[:, None])
    
    lse = m_i + tl.log(D_i)
    
    lse_ptr_bh = lse_ptr + b_h * S
    lse_store = tl.where(valid_rows, lse, 0.0)
    tl.store(lse_ptr_bh + row_idx, lse_store, mask=valid_rows)


def run(Q, K, V, O, LSE):
    """
    Computes causal multi-head attention O and LSE given Q, K, V on the current device.
    Expected shapes:
      Q, K, V : [B, H, S, D]
      O       : [B, H, S, D]
      LSE     : [B, H, S]
    Where B=4, H=48, D=128.
    """
    torch.cuda.set_device(Q.device)
    
    S = Q.shape[2]
    
    block_size = 256
    grid = (
        4,   # B
        48,  # H
        triton.cdiv(S, block_size)
    )
    
    _attention_forward_kernel[grid](
        Q, K, V, O, LSE,
        S,
        HEAD_DIM=128,
        BLOCK_S=block_size,
        num_warps=8,
        num_stages=1,
    )