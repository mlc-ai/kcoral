import math

import torch
import triton
import triton.language as tl


@triton.jit
def _mha_fwd_kernel(
    Q, K, V, O, LSE,
    Q_stride_b, Q_stride_h,
    K_stride_b, K_stride_h,
    V_stride_b, V_stride_h,
    LSE_stride_b, LSE_stride_h,
    S,
    scale,
    H,
    NUM_STAGES: tl.constexpr,
):
    """Optimized non-causal multi-head attention forward kernel."""
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    
    q_start = pid_x * 64
    b_idx = pid_y // H
    h_idx = pid_y % H
    
    q_ptr = Q + b_idx * Q_stride_b + h_idx * Q_stride_h
    k_ptr = K + b_idx * K_stride_b + h_idx * K_stride_h
    v_ptr = V + b_idx * V_stride_b + h_idx * V_stride_h
    o_ptr = O + b_idx * Q_stride_b + h_idx * Q_stride_h
    lse_ptr = LSE + b_idx * LSE_stride_b + h_idx * LSE_stride_h
    
    q_row = q_start + tl.arange(0, 64)
    col_D = tl.arange(0, 128)
    
    q = tl.load(q_ptr + q_row[:, None] * 128 + col_D[None, :],
                mask=(q_row[:, None] < S) & (col_D[None, :] < 128), other=0.0)
    
    o = tl.zeros((64, 128), tl.float32)
    m = tl.full((64, 1), -float('inf'), tl.float32)
    l = tl.full((64, 1), 0.0, tl.float32)
    
    num_kv_tiles = tl.cdiv(S, 64)
    
    for kv_idx in range(num_kv_tiles):
        k_start = kv_idx * 64
        k_row = k_start + tl.arange(0, 64)
        
        k = tl.load(k_ptr + k_row[:, None] * 128 + col_D[None, :],
                    mask=(k_row[:, None] < S) & (col_D[None, :] < 128), other=0.0)
        v = tl.load(v_ptr + k_row[:, None] * 128 + col_D[None, :] * 1,
                    mask=(k_row[:, None] < S) & (col_D[None, :] < 128), other=0.0)
        
        s = tl.dot(q, k) * scale
        
        m_prev = m
        m = tl.maximum(m, tl.max(s, axis=1, keep_dims=True))
        p = tl.exp(s - m)
        
        l_scale = tl.exp(m_prev - m)
        l = l * l_scale + tl.sum(p, axis=1, keep_dims=True)
        
        o = o * l_scale + tl.dot(p, v)
    
    o_final = (o / l).to(tl.bfloat16)
    out_ptr = o_ptr + q_row[:, None] * 128 + col_D[None, :]
    tl.store(out_ptr, o_final, mask=q_row[:, None] < S)
    
    lse = m + tl.log(l)
    lse_row = q_start + tl.arange(0, 64)
    mask_lse = lse_row < S
    lse_val = tl.where(mask_lse, lse.squeeze(), -float('inf'))
    tl.store(lse_ptr + lse_row, lse_val, mask=mask_lse)


def run(Q, K, V, O, LSE):
    """Compute non-causal MHA forward pass returning O and LSE."""
    torch.cuda.set_device(Q.device)
    B, H, S = Q.shape[0], Q.shape[1], Q.shape[2]
    
    scale = 1.0 / math.sqrt(128)
    
    grid = (triton.cdiv(S, 64), B * H)
    
    _mha_fwd_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1),
        K.stride(0), K.stride(1),
        V.stride(0), V.stride(1),
        LSE.stride(0), LSE.stride(1),
        S,
        scale,
        H,
        NUM_STAGES=2,
        num_warps=8,
        num_stages=2,
    )