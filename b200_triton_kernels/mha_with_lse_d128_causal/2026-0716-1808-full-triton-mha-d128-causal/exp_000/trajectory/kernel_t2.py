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
    TILE_Q: tl.constexpr, TILE_K: tl.constexpr, HALF_HEAD_DIM: tl.constexpr
):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    
    q_start = pid_y * TILE_Q
    if q_start >= seq_len:
        return
    
    batch_id = pid_x // num_heads
    head_id = pid_x % num_heads
    
    off_b = batch_id * num_heads * seq_len * 128
    off_h = head_id * seq_len * 128
    
    row = tl.arange(0, TILE_Q)[:, None]
    col = tl.arange(0, HALF_HEAD_DIM)[None, :]
    col1 = col + HALF_HEAD_DIM
    
    mask_q = (q_start + row.squeeze() < seq_len)
    
    q_ptr = q + off_b + off_h + q_start * stride_s_q
    offsets_q0 = q_ptr + row * stride_s_q + col * stride_d_q
    offsets_q1 = q_ptr + row * stride_s_q + col1 * stride_d_q
    
    q0 = tl.load(offsets_q0, mask=mask_q, other=0.0)
    q1 = tl.load(offsets_q1, mask=mask_q, other=0.0)
    
    out_acc0 = extern.shared_array((TILE_Q, HALF_HEAD_DIM), dtype=tl.float32)
    out_acc1 = extern.shared_array((TILE_Q, HALF_HEAD_DIM), dtype=tl.float32)
    
    ptr0 = out_acc0
    ptr1 = out_acc1
    
    for i in range(0, TILE_Q * HALF_HEAD_DIM, 4):
        val0 = tl.zeros((4,), dtype=tl.float32)
        addr0 = ptr0.to_bits() + (i * 4)
        tl.put_shared_memory(addr0, val0)
        
        val1 = tl.zeros((4,), dtype=tl.float32)
        addr1 = ptr1.to_bits() + (i * 4)
        tl.put_shared_memory(addr1, val1)
        
    p_exp_buf = extern.shared_array((TILE_Q, TILE_K), dtype=tl.float32)
    p_buf = extern.shared_array((TILE_Q, TILE_K), dtype=tl.float32)
    ptr_p = p_buf
    ptr_p_exp = p_exp_buf
    
    prev_max = tl.full((TILE_Q,), -1e20, dtype=tl.float32)
    prev_sum = tl.full((TILE_Q,), 0.0, dtype=tl.float32)
    
    for k_tile in range((q_start // TILE_K) + 1):
        k_start = k_tile * TILE_K
        
        col_k = tl.arange(0, TILE_K)[:, None]
        col_d = tl.arange(0, HALF_HEAD_DIM)[None, :]
        col_d1 = col_d + HALF_HEAD_DIM
        
        mask_k = (k_start + col_k.squeeze() < seq_len)
        
        k_ptr = k + off_b + off_h + k_start * stride_s_k
        offsets_k0 = k_ptr + col_k * stride_s_k + col_d * stride_d_k
        offsets_k1 = k_ptr + col_k * stride_s_k + col_d1 * stride_d_k
        
        k0 = tl.load(offsets_k0, mask=mask_k, other=0.0)
        k1 = tl.load(offsets_k1, mask=mask_k, other=0.0)
        
        acc_p = tl.zeros((TILE_Q, TILE_K), dtype=tl.float32)
        
        acc_p = tl.dot(q0, k0.T, acc_p)
        acc_p = tl.dot(q1, k1.T, acc_p)
        
        p = acc_p
        p = p * scale
        
        global_row = q_start + row.squeeze()
        global_col = k_start + col_k.squeeze()
        mask_causal = (global_row[:, None] >= global_col[None, :]) & \
                      (global_row[:, None] < seq_len) & \
                      (global_col[None, :] < seq_len)
        
        p = tl.where(mask_causal, p, -float('inf'))
        
        p_flat = p.flatten()
        for i in range(0, TILE_Q * TILE_K, 4):
            chunk = p_flat[i : i + 4]
            addr = ptr_p.to_bits() + (i * 4)
            tl.put_shared_memory(addr, chunk)
        
        curr_max = tl.full((TILE_Q,), -1e20, dtype=tl.float32)
        
        for i in range(TILE_Q):
            row_vals = tl.get_shared_memory_addr_range(ptr_p.to_bits() + (i * TILE_K * 4), TILE_K, dtype=tl.float32)
            row_max_val = tl.reduce_max(row_vals)
            curr_max[i] = row_max_val
            
        new_max = tl.maximum(prev_max, curr_max)
        
        for i in range(TILE_Q):
            scale_o = tl.exp(prev_max[i] - new_max[i])
            
            row_out0 = tl.get_shared_memory_addr_range(ptr0.to_bits() + (i * HALF_HEAD_DIM * 4), HALF_HEAD_DIM, dtype=tl.float32)
            row_out0 = row_out0 * scale_o
            out_acc0[i, :] = row_out0
            
            row_out1 = tl.get_shared_memory_addr_range(ptr1.to_bits() + (i * HALF_HEAD_DIM * 4), HALF_HEAD_DIM, dtype=tl.float32)
            row_out1 = row_out1 * scale_o
            out_acc1[i, :] = row_out1
            
        p = p - new_max[:, None]
        p_exp = tl.exp(p)
        
        curr_sum = tl.sum(p_exp, axis=1)
        new_sum = prev_sum * tl.exp(prev_max - new_max) + curr_sum
        
        prev_max = new_max
        prev_sum = new_sum
        
        p_exp_flat = p_exp.flatten()
        for i in range(0, TILE_Q * TILE_K, 4):
            chunk = p_exp_flat[i : i + 4]
            addr = ptr_p_exp.to_bits() + (i * 4)
            tl.put_shared_memory(addr, chunk)
            
        v_ptr = v + off_b + off_h + k_start * stride_s_v
        offsets_v0 = v_ptr + col_k * stride_s_v + col_d * stride_d_v
        offsets_v1 = v_ptr + col_k * stride_s_v + col_d1 * stride_d_v
        
        v0 = tl.load(offsets_v0, mask=mask_k, other=0.0)
        v1 = tl.load(offsets_v1, mask=mask_k, other=0.0)
        
        for i in range(HALF_HEAD_DIM):
            col_v = v0[:, i]
            dot_val = tl.dot(p_exp, col_v[None, :].T)
            out_acc0[:, i] += dot_val.squeeze()
            
            col_v1 = v1[:, i]
            dot_val1 = tl.dot(p_exp, col_v1[None, :].T)
            out_acc1[:, i] += dot_val1.squeeze()
    
    acc_o0 = tl.zeros((TILE_Q, HALF_HEAD_DIM), dtype=tl.float32)
    acc_o1 = tl.zeros((TILE_Q, HALF_HEAD_DIM), dtype=tl.float32)
    
    for i in range(TILE_Q):
        row_out0 = tl.get_shared_memory_addr_range(ptr0.to_bits() + (i * HALF_HEAD_DIM * 4), HALF_HEAD_DIM, dtype=tl.float32)
        acc_o0[i, :] = row_out0
        
        row_out1 = tl.get_shared_memory_addr_range(ptr1.to_bits() + (i * HALF_HEAD_DIM * 4), HALF_HEAD_DIM, dtype=tl.float32)
        acc_o1[i, :] = row_out1
        
    acc_o0 = acc_o0 / prev_sum[:, None]
    acc_o1 = acc_o1 / prev_sum[:, None]
    
    lse = tl.log(prev_sum) + prev_max
    
    lse_ptr0 = lse_ptr + (off_b // (seq_len * 128)) * stride_b_lse + (off_h // (seq_len * 128)) * stride_h_lse + q_start * stride_s_lse
    offsets_lse = lse_ptr0 + row.squeeze()
    
    tl.store(offsets_lse, lse, mask=(row.squeeze() < seq_len))
    
    out_ptr = o + off_b + off_h + q_start * stride_s_o
    offsets_o0 = out_ptr + row * stride_s_o + col * stride_d_o
    offsets_o1 = out_ptr + row * stride_s_o + col1 * stride_d_o
    
    tl.store(offsets_o0, acc_o0.to(out_ptr.dtype.element_ty), mask=mask_q)
    tl.store(offsets_o1, acc_o1.to(out_ptr.dtype.element_ty), mask=mask_q)


def run(Q, K, V, O, LSE):
    """
    Computes Multi-Head Attention forward with causal mask and outputs Log-Sum-Exp (LSE).
    Signature follows standard destination-passing style.
    """
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    num_heads = H
    
    scale = 1.0 / math.sqrt(D)
    
    TILE_Q = 128
    TILE_K = 128
    HALF_HEAD_DIM = 64
    
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
    
    grid = lambda META: (B * H, cdiv(S, META["TILE_Q"]))
    
    mha_with_lse_opt_kernel[grid](
        Q, K, V, O, LSE,
        S, num_heads,
        stride_b_q, stride_h_q, stride_s_q, stride_d_q,
        stride_b_k, stride_h_k, stride_s_k, stride_d_k,
        stride_b_v, stride_h_v, stride_s_v, stride_d_v,
        stride_b_o, stride_h_o, stride_s_o, stride_d_o,
        stride_b_lse, stride_h_lse, stride_s_lse,
        scale,
        TILE_Q=TILE_Q, TILE_K=TILE_K, HALF_HEAD_DIM=HALF_HEAD_DIM,
        num_warps=4, num_stages=3,
    )