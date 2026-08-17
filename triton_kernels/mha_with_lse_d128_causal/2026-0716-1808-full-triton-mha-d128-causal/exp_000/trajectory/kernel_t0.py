import math
import torch
import triton
import triton.language as tl
from triton import cdiv


@triton.jit
def load_async_gated(ptr, mask, other=0.0):
    return tl.load(ptr, mask=mask, other=other)


@triton.jit
def store_async_gated(ptr, value, mask):
    tl.store(ptr, value, mask=mask)


@triton.jit
def mha_with_lse_opt_kernel(
    q, k, v, o, lse_ptr,
    seq_len, head_dim, num_heads,
    stride_bh_q, stride_s_q, stride_d_q,
    stride_bh_k, stride_s_k, stride_d_k,
    stride_bh_v, stride_s_v, stride_d_v,
    stride_bh_o, stride_s_o, stride_d_o,
    stride_bh_lse, stride_s_lse,
    scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HALF_HEAD_DIM: tl.constexpr
):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    
    bh_id = pid_x
    q_tile_idx = pid_y
    q_start = q_tile_idx * BLOCK_M
    
    if q_start >= seq_len:
        return
    
    row = tl.arange(0, BLOCK_M)[:, None]
    col = tl.arange(0, BLOCK_N)[None, :]
    
    batch_id = bh_id // num_heads
    head_id = bh_id % num_heads
    
    bh_offset = batch_id * num_heads * seq_len * head_dim + head_id * seq_len * head_dim
    
    q_ptr_base = q + bh_offset * stride_bh_q + q_start * stride_s_q
    k_ptr_base = k + bh_offset * stride_bh_k
    v_ptr_base = v + bh_offset * stride_bh_v
    out_ptr_base = o + bh_offset * stride_bh_o + q_start * stride_s_o
    
    lse_ptr_base = lse_ptr + bh_offset * stride_bh_lse + q_start * stride_s_lse
    
    col_0 = col
    col_1 = col + HALF_HEAD_DIM
    
    offsets_q0 = q_ptr_base + row * stride_s_q + col_0 * stride_d_q
    offsets_q1 = q_ptr_base + row * stride_s_q + col_1 * stride_d_q
    
    offsets_lse = lse_ptr_base + row.squeeze()
    
    mask_q = (q_start + row < seq_len)
    
    q0 = load_async_gated(offsets_q0, mask_q, other=0.0)
    q1 = load_async_gated(offsets_q1, mask_q, other=0.0)
    
    s_row_max = extern.shared_array((BLOCK_M,), dtype=tl.float32)
    s_row_sum = extern.shared_array((BLOCK_M,), dtype=tl.float32)
    
    ptr_max = s_row_max
    ptr_sum = s_row_sum
    
    base_max = ptr_max.to_bits()
    base_sum = ptr_sum.to_bits()
    
    prev_max = -1e20
    prev_sum = 0.0
    
    for k_tile in range(q_tile_idx + 1):
        k_start = k_tile * BLOCK_N
        
        offsets_k0 = k_ptr_base + k_start * stride_s_k + col_0 * stride_d_k
        offsets_k1 = k_ptr_base + k_start * stride_s_k + col_1 * stride_d_k
        
        mask_k = (col < seq_len)
        
        k0 = load_async_gated(offsets_k0, mask_k, other=0.0)
        k1 = load_async_gated(offsets_k1, mask_k, other=0.0)
        
        acc_p = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        
        acc_p = tl.dot(q0, k0.T, acc_p)
        acc_p = tl.dot(q1, k1.T, acc_p)
        
        p = acc_p
        p = p * scale
        
        global_row = q_start + row
        global_col = k_start + col
        mask = (global_row >= global_col) & (global_col < seq_len)
        
        p = tl.where(mask, p, -float('inf'))
        
        row_max = tl.max(p, axis=1)
        p = p - row_max[:, None]
        p_exp = tl.exp(p)
        row_sum = tl.sum(p_exp, axis=1)
        
        curr_max = row_max
        curr_sum = row_sum
        
        new_max = tl.maximum(prev_max, curr_max)
        p_exp = p_exp * tl.exp(curr_max - new_max)
        new_sum = prev_sum * tl.exp(prev_max - new_max) + tl.sum(p_exp, axis=1)
        
        prev_max = new_max
        prev_sum = new_sum
        
    lse = tl.log(prev_sum) + prev_max
    prev_sum = lse
    
    store_async_gated(offsets_lse, prev_sum.to(lse_ptr_base.dtype.element_ty), (row.squeeze() < seq_len))
    
    for i in range(0, BLOCK_M, 4):
        chunk_max = prev_max[i : i + 4]
        addr_max = base_max + (i * 4)
        tl.put_shared_memory(addr_max, chunk_max)
        
        chunk_sum = prev_sum[i : i + 4]
        addr_sum = base_sum + (i * 4)
        tl.put_shared_memory(addr_sum, chunk_sum)
    
    tl.program_barrier()
    
    acc_o0 = tl.zeros((BLOCK_M, HALF_HEAD_DIM), dtype=tl.float32)
    acc_o1 = tl.zeros((BLOCK_M, HALF_HEAD_DIM), dtype=tl.float32)
    
    prev_max_o = -1e20
    prev_sum_o = 0.0
    
    for k_tile in range(q_tile_idx + 1):
        k_start = k_tile * BLOCK_N
        
        offsets_k0 = k_ptr_base + k_start * stride_s_k + col_0 * stride_d_k
        offsets_k1 = k_ptr_base + k_start * stride_s_k + col_1 * stride_d_k
        
        mask_k = (col < seq_len)
        
        k0 = load_async_gated(offsets_k0, mask_k, other=0.0)
        k1 = load_async_gated(offsets_k1, mask_k, other=0.0)
        
        acc_p = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        
        acc_p = tl.dot(q0, k0.T, acc_p)
        acc_p = tl.dot(q1, k1.T, acc_p)
        
        p = acc_p
        p = p * scale
        
        global_row = q_start + row
        global_col = k_start + col
        mask = (global_row >= global_col) & (global_col < seq_len)
        
        p = tl.where(mask, p, -float('inf'))
        
        row_max_o = tl.max(p, axis=1)
        p = p - row_max_o[:, None]
        p_exp = tl.exp(p)
        row_sum_o = tl.sum(p_exp, axis=1)
        
        curr_max_o = tl.max(p, axis=1)
        curr_sum_o = tl.sum(p_exp, axis=1)
        
        new_max_o = tl.maximum(prev_max_o, curr_max_o)
        p_exp = p_exp * tl.exp(curr_max_o - new_max_o)
        new_sum_o = prev_sum_o * tl.exp(prev_max_o - new_max_o) + tl.sum(p_exp, axis=1)
        
        prev_max_o = new_max_o
        prev_sum_o = new_sum_o
        
        p_exp = p_exp / row_sum_o[:, None]
        
        offsets_v0 = v_ptr_base + k_start * stride_s_v + col_0 * stride_d_v
        offsets_v1 = v_ptr_base + k_start * stride_s_v + col_1 * stride_d_v
        
        mask_v = (k_start + col < seq_len)
        
        v0 = load_async_gated(offsets_v0, mask_v, other=0.0)
        v1 = load_async_gated(offsets_v1, mask_v, other=0.0)
        
        acc_o0 = tl.dot(p_exp, v0, acc_o0)
        acc_o1 = tl.dot(p_exp, v1, acc_o1)
        
    final_max = tl.zeros((BLOCK_M,), dtype=tl.float32)
    final_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
    
    for i in range(0, BLOCK_M, 4):
        addr_max = base_max + (i * 4)
        chunk_max = tl.get_shared_memory_addr_range(addr_max, 4, dtype=tl.float32)
        final_max[i : i + 4] = chunk_max
        
        addr_sum = base_sum + (i * 4)
        chunk_sum = tl.get_shared_memory_addr_range(addr_sum, 4, dtype=tl.float32)
        final_sum[i : i + 4] = chunk_sum
        
    p_exp = p_exp * tl.exp(final_max - final_max[:, None])
    p_exp = p_exp / final_sum[:, None]
    
    acc_o0 = tl.dot(p_exp, v0, acc_o0)
    acc_o1 = tl.dot(p_exp, v1, acc_o1)
    
    offsets_o0 = out_ptr_base + row * stride_s_o + col_0 * stride_d_o
    offsets_o1 = out_ptr_base + row * stride_s_o + col_1 * stride_d_o
    
    store_async_gated(offsets_o0, acc_o0.to(out_ptr_base.dtype.element_ty), (row < seq_len))
    store_async_gated(offsets_o1, acc_o1.to(out_ptr_base.dtype.element_ty), (row < seq_len))


def run(Q, K, V, O, LSE):
    """
    Computes Multi-Head Attention forward with causal mask and outputs Log-Sum-Exp (LSE).
    Signature follows standard destination-passing style.
    """
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    num_heads = H
    
    scale = 1.0 / math.sqrt(D)
    
    BLOCK_M = 128
    BLOCK_N = 128
    HALF_HEAD_DIM = 64
    
    stride_bh_q = Q.stride(0)
    stride_s_q = Q.stride(2)
    stride_d_q = Q.stride(3)
    
    stride_bh_k = K.stride(0)
    stride_s_k = K.stride(2)
    stride_d_k = K.stride(3)
    
    stride_bh_v = V.stride(0)
    stride_s_v = V.stride(2)
    stride_d_v = V.stride(3)
    
    stride_bh_o = O.stride(0)
    stride_s_o = O.stride(2)
    stride_d_o = O.stride(3)
    
    stride_bh_lse = LSE.stride(0)
    stride_s_lse = LSE.stride(2)
    
    grid = lambda META: (B * H, cdiv(S, META["BLOCK_M"]))
    
    mha_with_lse_opt_kernel[grid](
        Q, K, V, O, LSE,
        S, D, num_heads,
        stride_bh_q, stride_s_q, stride_d_q,
        stride_bh_k, stride_s_k, stride_d_k,
        stride_bh_v, stride_s_v, stride_d_v,
        stride_bh_o, stride_s_o, stride_d_o,
        stride_bh_lse, stride_s_lse,
        scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HALF_HEAD_DIM=HALF_HEAD_DIM,
        num_warps=4, num_stages=3,
    )