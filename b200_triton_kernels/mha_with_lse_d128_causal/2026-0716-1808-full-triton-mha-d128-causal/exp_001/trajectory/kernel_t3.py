import torch
import triton
import triton.language as tl


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
    stride_Q_bh, stride_Q_s, stride_Q_d,
    stride_K_bh, stride_K_s, stride_K_d,
    stride_V_bh, stride_V_s, stride_V_d,
    stride_O_bh, stride_O_s, stride_O_d,
    stride_LSE_b, stride_LSE_h,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    bh = tl.program_id(1)
    q_block_id = tl.program_id(0)
    q_row_start = q_block_id * BLOCK_M
    
    bh_off_Q = bh * stride_Q_bh
    bh_off_K = bh * stride_K_bh
    bh_off_V = bh * stride_V_bh
    bh_off_O = bh * stride_O_bh
    
    Q_desc = tl.make_tensor_descriptor(
        Q_ptr + bh_off_Q, shape=[S, BLOCK_N], strides=[BLOCK_N, 1],
        block_shape=[BLOCK_M, BLOCK_N], padding_option="zero")
    
    K_desc = tl.make_tensor_descriptor(
        K_ptr + bh_off_K, shape=[S, BLOCK_N], strides=[BLOCK_N, 1],
        block_shape=[BLOCK_M, BLOCK_N], padding_option="zero")
        
    V_desc = tl.make_tensor_descriptor(
        V_ptr + bh_off_V, shape=[S, BLOCK_N], strides=[BLOCK_N, 1],
        block_shape=[BLOCK_M, BLOCK_N], padding_option="zero")
    
    m_arr = tl.arange(0, BLOCK_M)
    n_arr = tl.arange(0, BLOCK_N)
    
    Q_tile = Q_desc.load([q_row_start, 0]).to(tl.float32)
    
    O_acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    
    running_max = tl.full((BLOCK_M,), -1e38, tl.float32)
    running_sum = tl.full((BLOCK_M,), 0.0, tl.float32)
    
    query_valid = (q_row_start + m_arr) < S
    
    for kv_block_id in range(q_block_id + 1):
        kv_row_start = kv_block_id * BLOCK_N
        
        K_tile = K_desc.load([kv_row_start, 0]).to(tl.float32)
        
        P = tl.dot(Q_tile, K_tile.T)
        P *= scale
        
        causal_mask = (q_row_start + m_arr[:, None]) >= (kv_row_start + n_arr[None, :])
        mask = query_valid[:, None] & causal_mask
        P = tl.where(mask, P, -1e38)
        
        row_max = tl.max(P, axis=1)
        new_max = tl.maximum(running_max, row_max)
        
        running_sum *= tl.exp(running_max - new_max)
        O_acc *= tl.exp(running_max - new_max)[:, None]
        
        P_scaled = P - new_max[:, None]
        exp_P = tl.where(mask, tl.exp(P_scaled), 0.0)
        row_sum = tl.sum(exp_P, axis=1)
        running_sum += row_sum
        
        V_tile = V_desc.load([kv_row_start, 0]).to(tl.float32)
        O_acc += tl.dot(exp_P, V_tile)
                
        running_max = new_max
    
    O_acc /= running_sum[:, None]
    
    lse = running_max + tl.log(running_sum)
    
    row_mask = query_valid
    out_ptr_base = O_ptr + bh_off_O
    store_gm(out_ptr_base, q_row_start, 0, m_arr, O_acc.to(tl.bfloat16), stride_Q_s, stride_Q_d, row_mask[:, None])
    
    lse_ptr = LSE_ptr + bh * stride_LSE_h + q_row_start
    tl.store(lse_ptr + m_arr, lse, mask=row_mask)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    scale = 1.0 / (D ** 0.5)
    
    BLOCK_M = 128
    BLOCK_N = 128
    
    num_blocks = triton.cdiv(S, BLOCK_M)
    grid = (num_blocks, B * H)
    
    kernel[grid](
        Q, K, V, O, LSE,
        S, scale,
        Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(1), K.stride(2), K.stride(3),
        V.stride(1), V.stride(2), V.stride(3),
        O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, 
        num_warps=4, num_stages=3,
    )