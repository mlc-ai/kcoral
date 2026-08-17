import math
import torch
import triton
import triton.language as tl


@triton.jit
def _mha_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, lse_out_ptr,
    S, D, scale,
    stride_b, stride_h, stride_s,
    lse_stride_b, lse_stride_h, lse_strides,
):
    pid = tl.program_id(0)
    num_batches = 4
    num_heads = 48
    
    b = pid // (num_heads * tl.cdiv(S, 128))
    remaining = pid % (num_heads * tl.cdiv(S, 128))
    h = remaining // tl.cdiv(S, 128)
    m_idx = remaining % tl.cdiv(S, 128)
    
    start_m = m_idx * 128
    
    q_base = Q_ptr + b * stride_b + h * stride_h + start_m * stride_s
    row_idx = tl.arange(0, 128)[:, None]
    col_idx_0 = tl.arange(0, 64)[None, :]
    col_idx_1 = 64 + tl.arange(0, 64)[None, :]
    
    q0 = tl.load(q_base + row_idx * stride_s + col_idx_0, mask=(row_idx < S) & (col_idx_0 < 128), other=0.0)
    q1 = tl.load(q_base + row_idx * stride_s + col_idx_1, mask=(row_idx < S) & (col_idx_1 < 128), other=0.0)
    
    o_acc_left = tl.zeros((128, 64), tl.float32)
    o_acc_right = tl.zeros((128, 64), tl.float32)
    
    global_max = tl.full((1,), -float('inf'))
    denominator = tl.zeros((1,), tl.float32)
    
    num_blocks_k = tl.cdiv(S, 64)
    
    for k in range(num_blocks_k):
        k_base = K_ptr + b * stride_b + h * stride_h + k * stride_s
        k_idx = tl.arange(0, 64)[:, None]
        k0 = tl.load(k_base + k_idx * stride_s + col_idx_0, mask=(k_idx < S) & (col_idx_0 < 128), other=0.0)
        k1 = tl.load(k_base + k_idx * stride_s + col_idx_1, mask=(k_idx < S) & (col_idx_1 < 128), other=0.0)
        
        v_base = V_ptr + b * stride_b + h * stride_h + k * stride_s
        v0 = tl.load(v_base + k_idx * stride_s + col_idx_0, mask=(k_idx < S) & (col_idx_0 < 128), other=0.0)
        v1 = tl.load(v_base + k_idx * stride_s + col_idx_1, mask=(k_idx < S) & (col_idx_1 < 128), other=0.0)
        
        k0_t = k0.T
        k1_t = k1.T
        
        dot0 = tl.dot(q0, k0_t)
        dot1 = tl.dot(q1, k1_t)
        s = (dot0 + dot1) * scale
        
        new_max = tl.max(tl.max(s, dim=1), global_max)
        scaling = tl.exp(global_max - new_max)
        
        o_acc_left = o_acc_left * scaling
        o_acc_right = o_acc_right * scaling
        denominator = denominator * scaling
        
        s = s - new_max
        exp_scores = tl.exp(s)
        block_sum = tl.sum(exp_scores, dim=1)
        denominator = denominator + block_sum
        
        o_acc_left = tl.dot(exp_scores, v0, o_acc_left)
        o_acc_right = tl.dot(exp_scores, v1, o_acc_right)
        
        global_max = new_max

    o_left = o_acc_left / denominator
    o_right = o_acc_right / denominator
    
    out_base = O_ptr + b * stride_b + h * stride_h + start_m * stride_s
    row_idx = tl.arange(0, 128)[:, None]
    col_idx_left = tl.arange(0, 64)[None, :]
    col_idx_right = 64 + tl.arange(0, 64)[None, :]
    tl.store(out_base + row_idx * stride_s + col_idx_left, o_left)
    tl.store(out_base + row_idx * stride_s + col_idx_right, o_right)
    
    lse = global_max + tl.log(denominator)
    lse_base = lse_out_ptr + b * lse_stride_b + h * lse_stride_h + start_m * lse_strides
    tl.store(lse_base + row_idx, lse)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    scale = 1.0 / math.sqrt(D)
    
    stride_b = Q.stride(0)
    stride_h = Q.stride(1)
    stride_s = Q.stride(2)
    
    lse_stride_b = LSE.stride(0)
    lse_stride_h = LSE.stride(1)
    lse_strides = LSE.stride(2)
    
    grid = (B * H * triton.cdiv(S, 128),)
    _mha_kernel[grid](
        Q, K, V, O, LSE, S, D, scale,
        stride_b, stride_h, stride_s,
        lse_stride_b, lse_stride_h, lse_strides,
    )