import math
import torch
import triton
import triton.language as tl
from triton import cdiv


@triton.jit
def mha_with_lse_opt_kernel(
    q, k, v, o, lse_ptr,
    seq_len, num_heads,
    stride_b_q, stride_h_q, stride_s_q, stride_d_q,
    stride_b_k, stride_h_k, stride_s_k, stride_d_k,
    stride_b_v, stride_h_v, stride_s_v, stride_d_v,
    stride_b_o, stride_h_o, stride_s_o, stride_d_o,
    stride_b_lse, stride_h_lse, stride_s_lse,
    scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HALF_HEAD_DIM: tl.constexpr, HEAD_DIM: tl.constexpr
):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    
    q_start = pid_y * BLOCK_M
    if q_start >= seq_len:
        return
    
    batch_id = pid_x // num_heads
    head_id = pid_x % num_heads
    
    h_off_q = batch_id * stride_b_q + head_id * stride_h_q
    h_off_k = batch_id * stride_b_k + head_id * stride_h_k
    h_off_v = batch_id * stride_b_v + head_id * stride_h_v
    h_off_o = batch_id * stride_b_o + head_id * stride_h_o
    h_off_lse = batch_id * stride_b_lse + head_id * stride_h_lse
    
    seq_len_i32 = seq_len.to(tl.int32)
    q_start_i32 = q_start.to(tl.int32)
    
    row = tl.arange(0, BLOCK_M)
    col = tl.arange(0, HALF_HEAD_DIM)
    
    mask_q = (q_start_i32 + row.to(tl.int32) < seq_len_i32)[:, None]
    
    q_ptr_base = q + h_off_q + q_start * stride_s_q
    offsets_q0 = q_ptr_base + row[:, None] * stride_s_q + col[None, :] * stride_d_q
    offsets_q1 = q_ptr_base + row[:, None] * stride_s_q + (col + HALF_HEAD_DIM)[None, :] * stride_d_q
    
    q0 = tl.load(offsets_q0, mask=mask_q, other=0.0)
    q1 = tl.load(offsets_q1, mask=mask_q, other=0.0)
    
    acc_o0 = tl.zeros((BLOCK_M, HALF_HEAD_DIM), dtype=tl.float32)
    acc_o1 = tl.zeros((BLOCK_M, HALF_HEAD_DIM), dtype=tl.float32)
    
    prev_max = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    prev_sum = tl.full((BLOCK_M,), 0.0, dtype=tl.float32)
    
    num_k_tiles = min(pid_y + 1, (seq_len + BLOCK_N - 1) // BLOCK_N)
    
    # ==========================================
    # PASS 1: Compute row-wise max and sum
    # ==========================================
    for k_tile in range(num_k_tiles):
        k_start = k_tile * BLOCK_N
        k_start_i32 = k_start.to(tl.int32)
        
        col_k = tl.arange(0, BLOCK_N)
        mask_k = (k_start_i32 + col_k.to(tl.int32) < seq_len_i32)[:, None]
        
        k_ptr_base = k + h_off_k + k_start * stride_s_k
        offsets_k0 = k_ptr_base + col_k[:, None] * stride_s_k + col[None, :] * stride_d_k
        offsets_k1 = k_ptr_base + col_k[:, None] * stride_s_k + (col + HALF_HEAD_DIM)[None, :] * stride_d_k
        
        k0 = tl.load(offsets_k0, mask=mask_k, other=0.0)
        k1 = tl.load(offsets_k1, mask=mask_k, other=0.0)
        
        acc_p = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        
        acc_p = tl.dot(q0, k0.T, acc_p)
        acc_p = tl.dot(q1, k1.T, acc_p)
        
        p = acc_p * scale
        
        global_row = q_start_i32 + row.to(tl.int32)
        global_col = k_start_i32 + col_k.to(tl.int32)
        mask_causal = global_row[:, None] >= global_col[None, :]
        
        p = tl.where(mask_causal, p, -float('inf'))
        
        row_max_p = tl.max(p, axis=1)
        new_max = tl.maximum(prev_max, row_max_p)
        p = p - new_max[:, None]
        p_exp = tl.exp(p)
        
        row_sum_p = tl.sum(p_exp, axis=1)
        new_sum = prev_sum * tl.exp(prev_max - new_max) + row_sum_p
        
        prev_max = new_max
        prev_sum = new_sum
        
    # ==========================================
    # PASS 2: Compute Output O
    # ==========================================
    for k_tile in range(num_k_tiles):
        k_start = k_tile * BLOCK_N
        k_start_i32 = k_start.to(tl.int32)
        
        col_k = tl.arange(0, BLOCK_N)
        mask_k = (k_start_i32 + col_k.to(tl.int32) < seq_len_i32)[:, None]
        
        k_ptr_base = k + h_off_k + k_start * stride_s_k
        offsets_k0 = k_ptr_base + col_k[:, None] * stride_s_k + col[None, :] * stride_d_k
        offsets_k1 = k_ptr_base + col_k[:, None] * stride_s_k + (col + HALF_HEAD_DIM)[None, :] * stride_d_k
        
        k0 = tl.load(offsets_k0, mask=mask_k, other=0.0)
        k1 = tl.load(offsets_k1, mask=mask_k, other=0.0)
        
        acc_p = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        
        acc_p = tl.dot(q0, k0.T, acc_p)
        acc_p = tl.dot(q1, k1.T, acc_p)
        
        p = acc_p * scale
        
        global_row = q_start_i32 + row.to(tl.int32)
        global_col = k_start_i32 + col_k.to(tl.int32)
        mask_causal = global_row[:, None] >= global_col[None, :]
        
        p = tl.where(mask_causal, p, -float('inf'))
        
        row_max_p = tl.max(p, axis=1)
        p = p - row_max_p[:, None]
        p_exp = tl.exp(p)
        
        # Rescale current tile's exponentials to the global maximum scale
        p_exp = p_exp * tl.exp(row_max_p[:, None] - prev_max[:, None])
        # Normalize using the global sum
        p_exp = p_exp / prev_sum[:, None]
        
        v_ptr_base = v + h_off_v + k_start * stride_s_v
        offsets_v0 = v_ptr_base + col_k[:, None] * stride_s_v + col[None, :] * stride_d_v
        offsets_v1 = v_ptr_base + col_k[:, None] * stride_s_v + (col + HALF_HEAD_DIM)[None, :] * stride_d_v
        
        v0 = tl.load(offsets_v0, mask=mask_k, other=0.0)
        v1 = tl.load(offsets_v1, mask=mask_k, other=0.0)
        
        acc_o0 = tl.dot(p_exp, v0, acc_o0)
        acc_o1 = tl.dot(p_exp, v1, acc_o1)
        
    lse = tl.log(prev_sum) + prev_max
    
    lse_ptr0 = lse_ptr + h_off_lse + q_start * stride_s_lse
    offsets_lse = lse_ptr0 + row
    
    tl.store(offsets_lse, lse, mask=(q_start_i32 + row.to(tl.int32) < seq_len_i32))
    
    out_ptr_base = o + h_off_o + q_start * stride_s_o
    offsets_o0 = out_ptr_base + row[:, None] * stride_s_o + col[None, :] * stride_d_o
    offsets_o1 = out_ptr_base + row[:, None] * stride_s_o + (col + HALF_HEAD_DIM)[None, :] * stride_d_o
    
    tl.store(offsets_o0, acc_o0, mask=mask_q)
    tl.store(offsets_o1, acc_o1, mask=mask_q)


def run(Q, K, V, O, LSE):
    """
    Computes Multi-Head Attention forward with causal mask and outputs Log-Sum-Exp (LSE).
    Signature follows standard destination-passing style.
    """
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    num_heads = H
    
    scale = 1.0 / math.sqrt(D)
    
    BLOCK_M = 64
    BLOCK_N = 64
    HALF_HEAD_DIM = 64
    HEAD_DIM = 128
    
    stride_b_q = Q.stride(0)
    stride_h_q = Q.stride(1)
    stride_s_q = Q.stride(2)
    stride_d_q = Q.stride(3)
    
    stride_b_k = K.stride(0)
    stride_h_k = K.stride(1)
    stride_s_k = K.stride(2)
    stride_d_k = K.stride(3)
    
    stride_b_v = V.stride(0)
    stride_h_v = V.stride(1)
    stride_s_v = V.stride(2)
    stride_d_v = V.stride(3)
    
    stride_b_o = O.stride(0)
    stride_h_o = O.stride(1)
    stride_s_o = O.stride(2)
    stride_d_o = O.stride(3)
    
    stride_b_lse = LSE.stride(0)
    stride_h_lse = LSE.stride(1)
    stride_s_lse = LSE.stride(2)
    
    grid = lambda META: (B * H, cdiv(S, META["BLOCK_M"]))
    
    mha_with_lse_opt_kernel[grid](
        Q, K, V, O, LSE,
        S, num_heads,
        stride_b_q, stride_h_q, stride_s_q, stride_d_q,
        stride_b_k, stride_h_k, stride_s_k, stride_d_k,
        stride_b_v, stride_h_v, stride_s_v, stride_d_v,
        stride_b_o, stride_h_o, stride_s_o, stride_d_o,
        stride_b_lse, stride_h_lse, stride_s_lse,
        scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HALF_HEAD_DIM=HALF_HEAD_DIM, HEAD_DIM=HEAD_DIM,
        num_warps=4, num_stages=3,
    )