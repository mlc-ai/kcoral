import torch
import triton
import triton.language as tl


@triton.jit
def max_or_zero(x):
    return tl.maximum(tl.max(x, axis=1), 0.0)


@triton.jit
def sum_or_zero(x):
    return tl.maximum(tl.sum(x, axis=1), 0.0)


@triton.jit(__launch_bounds__(128, 1))
def kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    S, scale,
    stride_Q_bh, stride_K_bh, stride_V_bh, stride_O_bh, stride_LSE_bh,
    stride_Q_s, stride_K_s, stride_V_s, stride_O_s, stride_LSE_s,
):
    global_q = tl.program_id(0)
    bh = tl.program_id(1)
    
    bh_off_Q = bh * stride_Q_bh
    bh_off_K = bh * stride_K_bh
    bh_off_V = bh * stride_V_bh
    bh_off_O = bh * stride_O_bh
    
    c = tl.arange(0, 128)
    Q_row = tl.load(Q_ptr + bh_off_Q + global_q * stride_Q_s + c, mask=global_q < S, other=0.0).to(tl.float32)
    
    O_acc_0 = tl.zeros((1, 64), tl.float32)
    O_acc_1 = tl.zeros((1, 64), tl.float32)
    
    running_max = -1e38
    running_sum = 0.0
    
    c_0 = tl.arange(0, 64)
    c_1 = tl.arange(0, 64)
    
    for kv_block_id in range(global_q // 128 + 1):
        kv_start = kv_block_id * 128
        
        K_tile = tl.load(K_ptr + bh_off_K + (kv_start + c[:, None]) * stride_K_s + c[None, :], 
                         mask=(kv_start + c[:, None]) < S, other=0.0).to(tl.float32)
        V_tile_0 = tl.load(V_ptr + bh_off_V + (kv_start + c[:, None]) * stride_V_s + c_0[None, :], 
                           mask=(kv_start + c[:, None]) < S, other=0.0).to(tl.float32)
        V_tile_1 = tl.load(V_ptr + bh_off_V + (kv_start + c[:, None]) * stride_V_s + (c_1 + 64)[None, :], 
                           mask=(kv_start + c[:, None]) < S, other=0.0).to(tl.float32)
        
        P_row = tl.dot(Q_row, K_tile.T, use_mma=False)
        P_row *= scale
        
        causal_mask = (global_q >= kv_start + c[None, :]) & (global_q < S)
        P_row = tl.where(causal_mask, P_row, -1e38)
        
        block_max = max_or_zero(P_row)
        new_max = max(block_max, running_max)
        
        P_row_scaled = P_row - new_max[None, :]
        
        exp_P = P_row * 0.0
        exp_P = tl.where(causal_mask, tl.exp(P_row_scaled), exp_P)
        
        block_sum = sum_or_zero(exp_P)
        
        running_sum *= tl.exp(running_max - new_max)
        running_sum += block_sum * tl.exp(block_max - new_max)
        
        O_acc_0 *= tl.exp(running_max - new_max)
        O_acc_1 *= tl.exp(running_max - new_max)
        
        O_acc_0 += tl.dot(exp_P, V_tile_0, use_mma=False)
        O_acc_1 += tl.dot(exp_P, V_tile_1, use_mma=False)
        
        running_max = new_max
    
    final_scale = 1.0 / running_sum
    O_acc_0 *= final_scale
    O_acc_1 *= final_scale
    
    lse_val = running_max + tl.log(running_sum)
    
    O_acc_0_bf = O_acc_0.to(tl.bfloat16)
    O_acc_1_bf = O_acc_1.to(tl.bfloat16)
    
    tl.store(O_ptr + bh_off_O + global_q * stride_O_s + c_0, O_acc_0_bf, mask=global_q < S)
    tl.store(O_ptr + bh_off_O + global_q * stride_O_s + (c_1 + 64), O_acc_1_bf, mask=global_q < S)
    
    tl.store(LSE_ptr + bh * stride_LSE_bh + global_q * stride_LSE_s, lse_val, mask=global_q < S)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    scale = 1.0 / (D ** 0.5)
    
    grid = (S, B * H)
    
    kernel[grid](
        Q, K, V, O, LSE,
        S, scale,
        Q.stride(1), K.stride(1), V.stride(1), O.stride(1), LSE.stride(1),
        Q.stride(2), K.stride(2), V.stride(2), O.stride(2), LSE.stride(2),
        num_warps=4,
    )