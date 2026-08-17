import torch
import triton
import triton.language as tl


@triton.jit
def load_gm_padded(base_ptr, row_start, col_start, num_rows, num_cols, max_row, stride_row, stride_col):
    r_off = tl.arange(0, num_rows)
    c_off = tl.arange(0, num_cols)
    ptrs = base_ptr + (row_start + r_off)[:, None] * stride_row + (col_start + c_off)[None, :] * stride_col
    mask = (row_start + r_off)[:, None] < max_row
    return tl.load(ptrs, mask=mask, other=0.0)


@triton.jit
def store_gm(base_ptr, row_start, col_start, row_offsets, vals, stride_row, stride_col, mask):
    num_cols = vals.shape[1]
    c_off = tl.arange(0, num_cols)
    ptrs = base_ptr + (row_start + row_offsets)[:, None] * stride_row + (col_start + c_off)[None, :] * stride_col
    tl.store(ptrs, vals, mask=mask)


@triton.jit
def kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    S, scale,
    stride_Q_b, stride_Q_h, stride_Q_s, stride_Q_d,
    stride_K_b, stride_K_h, stride_K_s, stride_K_d,
    stride_V_b, stride_V_h, stride_V_s, stride_V_d,
    stride_O_b, stride_O_h, stride_O_s, stride_O_d,
    stride_LSE_b, stride_LSE_h, stride_LSE_s,
):
    bh = tl.program_id(1)
    q_block_id = tl.program_id(0)
    q_row_start = q_block_id * 128
    
    b_idx = bh // 48
    h_idx = bh % 48
    
    bh_off_Q = b_idx * stride_Q_b + h_idx * stride_Q_h
    bh_off_K = b_idx * stride_K_b + h_idx * stride_K_h
    bh_off_V = b_idx * stride_V_b + h_idx * stride_V_h
    bh_off_O = b_idx * stride_O_b + h_idx * stride_O_h
    bh_off_LSE = b_idx * stride_LSE_b + h_idx * stride_LSE_h
    
    m_arr = tl.arange(0, 128)
    
    Q_tile = load_gm_padded(Q_ptr + bh_off_Q, q_row_start, 0, 128, 128, S, stride_Q_s, stride_Q_d).to(tl.bfloat16)
    
    O_acc = tl.zeros((128, 128), tl.float32)
    
    running_max = tl.full((128,), -1e38, tl.float32)
    running_sum = tl.full((128,), 0.0, tl.float32)
    
    query_valid = (q_row_start + m_arr) < S
    
    for kv_block_id in range(q_block_id + 1):
        K_tile = load_gm_padded(K_ptr + bh_off_K, kv_block_id * 128, 0, 128, 128, S, stride_K_s, stride_K_d).to(tl.bfloat16)
        
        P = tl.dot(Q_tile, K_tile.T)
        P *= scale
        
        k_arr = tl.arange(0, 128)
        causal_valid = (q_row_start + m_arr[:, None]) >= (kv_block_id * 128 + k_arr[None, :])
        mask = query_valid[:, None] & causal_valid
        P = tl.where(mask, P, -1e38)
        
        row_max = tl.max(P, axis=1)
        new_max = tl.maximum(running_max, row_max)
        
        # Correctly rescale the previously accumulated outputs to align with the new block maximum
        running_sum *= tl.exp(running_max - new_max)
        O_acc *= tl.exp(running_max - new_max)[:, None]
        
        P_scaled = P - new_max[:, None]
        exp_P = tl.where(mask, tl.exp(P_scaled), 0.0).to(tl.float32)
        row_sum = tl.sum(exp_P, axis=1)
        running_sum += row_sum
        
        V_tile = load_gm_padded(V_ptr + bh_off_V, kv_block_id * 128, 0, 128, 128, S, stride_V_s, stride_V_d).to(tl.float32)
        O_acc += tl.dot(exp_P, V_tile)
                
        running_max = new_max
    
    O_acc /= running_sum[:, None]
    
    lse = running_max + tl.log(running_sum)
    
    row_mask = query_valid
    out_ptr_base = O_ptr + bh_off_O
    store_gm(out_ptr_base, q_row_start, 0, m_arr, O_acc, stride_O_s, stride_O_d, row_mask[:, None])
    
    lse_ptr = LSE_ptr + bh_off_LSE + q_row_start * stride_LSE_s
    tl.store(lse_ptr + m_arr, lse, mask=row_mask)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    scale = 1.0 / (D ** 0.5)
    
    num_blocks = triton.cdiv(S, 128)
    grid = (num_blocks, B * H)
    
    kernel[grid](
        Q, K, V, O, LSE,
        S, scale,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        num_warps=4,
    )