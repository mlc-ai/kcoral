import torch
import triton
import triton.language as tl
import math


@triton.jit
def load_tile(ptr, row_off, col_off, BM, BN, stride_r, stride_c, total):
    rows = tl.arange(0, BM)
    cols = tl.arange(0, BN)
    r_idx = row_off + rows
    ptrs = ptr + r_idx[:, None] * stride_r + (col_off + cols)[None, :] * stride_c
    return tl.load(ptrs, mask=(r_idx[:, None] < total), other=0.0)

@triton.jit
def store_tile(ptr, row_off, col_off, BM, BN, stride_r, stride_c, value, total):
    rows = tl.arange(0, BM)
    cols = tl.arange(0, BN)
    r_idx = row_off + rows
    ptrs = ptr + r_idx[:, None] * stride_r + (col_off + cols)[None, :] * stride_c
    tl.store(ptrs, value, mask=(r_idx[:, None] < total))


@triton.jit
def _attention_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, LSE,
    total_rows, S_length,
    stride_row, stride_col,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    bh = pid_bh
    S = S_length
    row_offset = bh * S + pid_m * BLOCK_M
    
    q0 = load_tile(q_ptr, row_offset, 0, BLOCK_M, 64, stride_row, stride_col, total_rows)
    q1 = load_tile(q_ptr, row_offset, 64, BLOCK_M, 64, stride_row, stride_col, total_rows)
    
    acc_o0 = tl.zeros((BLOCK_M, 64), tl.float32)
    acc_o1 = tl.zeros((BLOCK_M, 64), tl.float32)
    acc_m = tl.full((BLOCK_M,), float('-inf'), tl.float32)
    acc_sum = tl.zeros((BLOCK_M,), tl.float32)
    
    scale = 1.0 / math.sqrt(128.0)
    
    next_k0 = load_tile(k_ptr, bh * S + 0 * BLOCK_N, 0, BLOCK_N, 64, stride_row, stride_col, total_rows)
    next_k1 = load_tile(k_ptr, bh * S + 0 * BLOCK_N, 64, BLOCK_N, 64, stride_row, stride_col, total_rows)
    next_v0 = load_tile(v_ptr, bh * S + 0 * BLOCK_N, 0, BLOCK_N, 64, stride_row, stride_col, total_rows)
    next_v1 = load_tile(v_ptr, bh * S + 0 * BLOCK_N, 64, BLOCK_N, 64, stride_row, stride_col, total_rows)
    
    for k_tile in range(S // 128):
        col_offset = bh * S + k_tile * BLOCK_N
        
        k0 = next_k0
        k1 = next_k1
        v0 = next_v0
        v1 = next_v1
        
        if k_tile < S // 128 - 1:
            next_col_offset = bh * S + (k_tile + 1) * BLOCK_N
            next_k0 = load_tile(k_ptr, next_col_offset, 0, BLOCK_N, 64, stride_row, stride_col, total_rows)
            next_k1 = load_tile(k_ptr, next_col_offset, 64, BLOCK_N, 64, stride_row, stride_col, total_rows)
            next_v0 = load_tile(v_ptr, next_col_offset, 0, BLOCK_N, 64, stride_row, stride_col, total_rows)
            next_v1 = load_tile(v_ptr, next_col_offset, 64, BLOCK_N, 64, stride_row, stride_col, total_rows)
            
        s = tl.dot(q0, k0.T) + tl.dot(q1, k1.T)
        s *= scale
        
        cols = tl.arange(0, BLOCK_N)
        col_valid = k_tile * BLOCK_N + cols < S_length
        s = tl.where(col_valid[None, :], s, float('-inf'))
        
        row_max = tl.max(s, axis=1)
        m_new = tl.maximum(acc_m, row_max)
        alpha = tl.exp(acc_m - m_new)
        p = tl.exp(s - m_new[:, None])
        
        acc_sum = acc_sum * alpha + tl.sum(p, axis=1)
        acc_m = m_new
        
        acc_o0 = acc_o0 * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v0)
        acc_o1 = acc_o1 * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v1)
        
    denom = acc_sum
    
    acc_o0 = acc_o0 / denom[:, None]
    acc_o1 = acc_o1 / denom[:, None]
    
    store_tile(o_ptr, row_offset, 0, BLOCK_M, 64, stride_row, stride_col, acc_o0.to(tl.bfloat16), total_rows)
    store_tile(o_ptr, row_offset, 64, BLOCK_M, 64, stride_row, stride_col, acc_o1.to(tl.bfloat16), total_rows)
    
    rows = tl.arange(0, BLOCK_M)
    lse = acc_m + tl.log(denom)
    
    lse_ptr_curr = LSE + (bh * S + pid_m * BLOCK_M) + rows
    row_valid = bh * S + pid_m * BLOCK_M + rows < (bh + 1) * S
    tl.store(lse_ptr_curr, lse, mask=row_valid)


def run(Q, K, V, O, LSE):
    """Compute Non-causal multi-head attention forward returning O and LSE."""
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    if S == 0:
        return
    
    Q = Q.reshape(B * H * S, D)
    K = K.reshape(B * H * S, D)
    V = V.reshape(B * H * S, D)
    O = O.reshape(B * H * S, D)
    
    total_rows = B * H * S
    
    grid = (triton.cdiv(S, 128), B * H)
    
    _attention_kernel[grid](
        Q, K, V, O, LSE,
        total_rows, S,
        D, 1,
        BLOCK_M=128, BLOCK_N=128,
        num_warps=8, num_stages=5
    )