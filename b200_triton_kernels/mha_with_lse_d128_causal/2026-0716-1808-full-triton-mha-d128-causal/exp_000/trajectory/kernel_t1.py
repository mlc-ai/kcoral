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
    
    head_offset_q = batch_id * stride_b_q + head_id * stride_h_q
    head_offset_k = batch_id * stride_b_k + head_id * stride_h_k
    head_offset_v = batch_id * stride_b_v + head_id * stride_h_v
    head_offset_o = batch_id * stride_b_o + head_id * stride_h_o
    head_offset_lse = batch_id * stride_b_lse + head_id * stride_h_lse
    
    row = tl.arange(0, TILE_Q)[:, None]
    col = tl.arange(0, HALF_HEAD_DIM)[None, :]
    col1 = col + HALF_HEAD_DIM
    
    mask_q = (q_start + row.squeeze() < seq_len)
    
    q_ptr0 = q + head_offset_q + q_start * stride_s_q
    offsets_q0 = q_ptr0 + row * stride_s_q + col * stride_d_q
    q0 = tl.load(offsets_q0, mask=mask_q, other=0.0)
    
    offsets_q1 = q_ptr0 + row * stride_s_q + col1 * stride_d_q
    q1 = tl.load(offsets_q1, mask=mask_q, other=0.0)
    
    acc_o0 = tl.zeros((TILE_Q, HALF_HEAD_DIM), dtype=tl.float32)
    acc_o1 = tl.zeros((TILE_Q, HALF_HEAD_DIM), dtype=tl.float32)
    
    prev_max = -1e20
    prev_sum = 0.0
    
    for k_tile in range((q_start // TILE_K) + 1):
        k_start = k_tile * TILE_K
        
        col_k = tl.arange(0, TILE_K)[:, None]
        col_d = tl.arange(0, HALF_HEAD_DIM)[None, :]
        col_d1 = col_d + HALF_HEAD_DIM
        
        mask_k = (k_start + col_k.squeeze() < seq_len)
        
        k_ptr0 = k + head_offset_k + k_start * stride_s_k
        offsets_k0 = k_ptr0 + col_k * stride_s_k + col_d * stride_d_k
        k0 = tl.load(offsets_k0, mask=mask_k, other=0.0)
        
        offsets_k1 = k_ptr0 + col_k * stride_s_k + col_d1 * stride_d_k
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
        
        row_max = tl.max(p, axis=1)
        p = p - row_max[:, None]
        p_exp = tl.exp(p)
        
        row_sum = tl.sum(p_exp, axis=1)
        
        new_max = tl.maximum(prev_max, row_max)
        p_exp = p_exp * tl.exp(row_max - new_max)
        new_sum = prev_sum * tl.exp(prev_max - new_max) + tl.sum(p_exp, axis=1)
        
        prev_max = new_max
        prev_sum = new_sum
        
        p_exp = p_exp / prev_sum[:, None]
        
        v_ptr0 = v + head_offset_v + k_start * stride_s_v
        offsets_v0 = v_ptr0 + col_k * stride_s_v + col_d * stride_d_v
        v0 = tl.load(offsets_v0, mask=mask_k, other=0.0)
        
        offsets_v1 = v_ptr0 + col_k * stride_s_v + col_d1 * stride_d_v
        v1 = tl.load(offsets_v1, mask=mask_k, other=0.0)
        
        acc_o0 = tl.dot(p_exp, v0, acc_o0)
        acc_o1 = tl.dot(p_exp, v1, acc_o1)
        
    lse = tl.log(prev_sum) + prev_max
    
    lse_ptr0 = lse_ptr + head_offset_lse + q_start * stride_s_lse
    offsets_lse = lse_ptr0 + row.squeeze()
    
    tl.store(offsets_lse, lse, mask=(row.squeeze() < seq_len))
    
    out_ptr0 = o + head_offset_o + q_start * stride_s_o
    offsets_o0 = out_ptr0 + row * stride_s_o + col * stride_d_o
    offsets_o1 = out_ptr0 + row * stride_s_o + col1 * stride_d_o
    
    tl.store(offsets_o0, acc_o0.to(out_ptr0.dtype.element_ty), mask=mask_q)
    tl.store(offsets_o1, acc_o1.to(out_ptr0.dtype.element_ty), mask=mask_q)


def run(Q, K, V, O, LSE):
    """
    Computes Multi-Head Attention forward with causal mask and outputs Log-Sum-Exp (LSE).
    Signature follows standard destination-passing style.
    """
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    num_heads = H
    
    scale = 1.0 / math.sqrt(D)
    
    TILE_Q = 64
    TILE_K = 64
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