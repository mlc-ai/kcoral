import math
import torch
import triton
import triton.language as tl


@triton.jit
def _mha_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, lse_out_ptr,
    S, scale,
    stride_b, stride_h, stride_s, stride_d,
    lse_stride_b, lse_stride_h, lse_stride_s,
):
    pid = tl.program_id(0)
    num_heads = 48
    
    b = pid // (num_heads * tl.cdiv(S, 64))
    remaining = pid % (num_heads * tl.cdiv(S, 64))
    h = remaining // tl.cdiv(S, 64)
    m_idx = remaining % tl.cdiv(S, 64)
    
    start_m = m_idx * 64
    num_k_blocks = tl.cdiv(S, 64)
    
    row_idx = tl.arange(0, 64)
    k_row_idx = tl.arange(0, 64)
    col_idx = tl.arange(0, 128)
    
    # Load Q tile - shape (64, 128)
    q_ptr_0 = Q_ptr + b * stride_b + h * stride_h + start_m * stride_s
    Q_tile = tl.load(q_ptr_0 + row_idx[:, None] * stride_s + col_idx[None, :] * stride_d,
                      mask=(start_m + row_idx[:, None] < S), other=0.0)
    
    o_acc = tl.zeros((64, 128), tl.float32)
    global_max = tl.full((64,), -10000.0, dtype=tl.float32)
    denominator = tl.zeros((64,), tl.float32)
    
    # Explicit double buffering management for K and V to overlap load latency with computation
    k_0 = tl.zeros((64, 128), tl.bfloat16)
    v_0 = tl.zeros((64, 128), tl.bfloat16)
    k_1 = tl.zeros((64, 128), tl.bfloat16)
    v_1 = tl.zeros((64, 128), tl.bfloat16)
    
    if num_k_blocks > 0:
        k_ptr_0 = K_ptr + b * stride_b + h * stride_h + 0 * stride_s
        k_0 = tl.load(k_ptr_0 + k_row_idx[:, None] * stride_s + col_idx[None, :] * stride_d,
                       mask=(0 + k_row_idx[:, None] < S), other=0.0)
        v_ptr_0 = V_ptr + b * stride_b + h * stride_h + 0 * stride_s
        v_0 = tl.load(v_ptr_0 + k_row_idx[:, None] * stride_s + col_idx[None, :] * stride_d,
                       mask=(0 + k_row_idx[:, None] < S), other=0.0)
    
    for k in range(num_k_blocks):
        idx = k % 2
        next_k = k + 1
        
        if next_k < num_k_blocks:
            k_ptr_next = K_ptr + b * stride_b + h * stride_h + next_k * stride_s
            v_ptr_next = V_ptr + b * stride_b + h * stride_h + next_k * stride_s
            if idx == 0:
                k_1 = tl.load(k_ptr_next + k_row_idx[:, None] * stride_s + col_idx[None, :] * stride_d,
                               mask=(next_k + k_row_idx[:, None] < S), other=0.0)
                v_1 = tl.load(v_ptr_next + k_row_idx[:, None] * stride_s + col_idx[None, :] * stride_d,
                               mask=(next_k + k_row_idx[:, None] < S), other=0.0)
            else:
                k_0 = tl.load(k_ptr_next + k_row_idx[:, None] * stride_s + col_idx[None, :] * stride_d,
                               mask=(next_k + k_row_idx[:, None] < S), other=0.0)
                v_0 = tl.load(v_ptr_next + k_row_idx[:, None] * stride_s + col_idx[None, :] * stride_d,
                               mask=(next_k + k_row_idx[:, None] < S), other=0.0)
        else:
            if idx == 0:
                k_1 = tl.zeros((64, 128), tl.bfloat16)
                v_1 = tl.zeros((64, 128), tl.bfloat16)
            else:
                k_0 = tl.zeros((64, 128), tl.bfloat16)
                v_0 = tl.zeros((64, 128), tl.bfloat16)
        
        if idx == 0:
            K_curr = k_0
            V_curr = v_0
        else:
            K_curr = k_1
            V_curr = v_1
        
        s = tl.dot(Q_tile, K_curr.T) * scale
        
        col_mask = (k * 64 + tl.arange(0, 64) < S)[None, :]
        s = tl.where(col_mask, s, -10000.0)
        
        block_max = tl.max(s, dim=1)
        new_max = tl.maximum(global_max, block_max)
        scaling = tl.exp(global_max - new_max)
        
        o_acc = o_acc * scaling[:, None]
        denominator = denominator * scaling
        
        p = tl.exp(s - new_max[:, None])
        
        block_sum = tl.sum(p, dim=1)
        denominator = denominator + block_sum
        
        o_acc = tl.dot(p, V_curr, o_acc)
        
        global_max = new_max
        
        tl.debug_barrier()

    o_out = o_acc / denominator[:, None]
    
    out_base = O_ptr + b * stride_b + h * stride_h + start_m * stride_s
    mask_o = (start_m + row_idx[:, None] < S)
    tl.store(out_base + row_idx[:, None] * stride_s + col_idx[None, :] * stride_d, o_out, mask=mask_o)
    
    row_idx_flat = tl.arange(0, 64)
    lse = global_max + tl.log(denominator)
    lse_base = lse_out_ptr + b * lse_stride_b + h * lse_stride_h + start_m * lse_stride_s
    mask_lse = start_m + row_idx_flat < S
    tl.store(lse_base + row_idx_flat, lse, mask=mask_lse)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    scale = 1.0 / math.sqrt(D)
    
    stride_b = Q.stride(0)
    stride_h = Q.stride(1)
    stride_s = Q.stride(2)
    stride_d = Q.stride(3)
    
    lse_stride_b = LSE.stride(0)
    lse_stride_h = LSE.stride(1)
    lse_stride_s = LSE.stride(2)
    
    grid = (B * H * triton.cdiv(S, 64),)
    _mha_kernel[grid](
        Q, K, V, O, LSE, S, scale,
        stride_b, stride_h, stride_s, stride_d,
        lse_stride_b, lse_stride_h, lse_stride_s,
        num_stages=3,
        num_warps=4,
    )