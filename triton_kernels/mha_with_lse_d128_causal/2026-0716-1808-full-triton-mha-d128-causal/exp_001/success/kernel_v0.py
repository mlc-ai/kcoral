import torch
import triton
import triton.language as tl


@triton.jit
def kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    S, scale,
    stride_Q_bh, stride_K_bh, stride_V_bh, stride_O_bh,
    stride_LSE_h,
):
    bh = tl.program_id(1)
    q_block_id = tl.program_id(0)
    q_row_start = q_block_id * 128
    
    bh_off_Q = bh * stride_Q_bh
    bh_off_K = bh * stride_K_bh
    bh_off_V = bh * stride_V_bh
    bh_off_O = bh * stride_O_bh
    
    r = tl.arange(0, 128)
    c = tl.arange(0, 128)
    
    Q_ptrs = Q_ptr + bh_off_Q + (q_row_start + r)[:, None] * 128 + c[None, :] * 1
    Q_tile = tl.load(Q_ptrs, mask=(q_row_start + r)[:, None] < S, other=0.0)
    
    O_acc_0 = tl.zeros((128, 64), tl.float32)
    O_acc_1 = tl.zeros((128, 64), tl.float32)
    
    running_max = tl.full((128,), -1e38, tl.float32)
    running_sum = tl.full((128,), 0.0, tl.float32)
    
    query_valid = (q_row_start + r) < S
    
    for kv_block_id in range(q_block_id + 1):
        kv_row_start = kv_block_id * 128
        
        k_r = tl.arange(0, 128)
        k_c = tl.arange(0, 128)
        
        K_ptrs = K_ptr + bh_off_K + (kv_row_start + k_r)[:, None] * 128 + k_c[None, :] * 1
        K_tile = tl.load(K_ptrs, mask=(kv_row_start + k_r)[:, None] < S, other=0.0)
        
        P = tl.dot(Q_tile, K_tile.T)
        P *= scale
        
        causal_mask = (q_row_start + r[:, None]) >= (kv_row_start + k_c[None, :])
        mask = query_valid[:, None] & causal_mask
        P = tl.where(mask, P, -1e38)
        
        row_max = tl.max(P, axis=1)
        new_max = tl.maximum(running_max, row_max)
        
        running_sum *= tl.exp(running_max - new_max)
        O_acc_0 *= tl.exp(running_max - new_max)[:, None]
        O_acc_1 *= tl.exp(running_max - new_max)[:, None]
        
        P_scaled = P - new_max[:, None]
        exp_P = tl.where(mask, tl.exp(P_scaled), 0.0)
        row_sum = tl.sum(exp_P, axis=1)
        running_sum += row_sum
        
        exp_P_bf16 = exp_P.to(tl.bfloat16)
        
        k_c_half = tl.arange(0, 64)
        
        V_0_ptrs = V_ptr + bh_off_V + (kv_row_start + k_r)[:, None] * 128 + k_c_half[None, :] * 1
        V_0 = tl.load(V_0_ptrs, mask=(kv_row_start + k_r)[:, None] < S, other=0.0)
        
        V_1_ptrs = V_ptr + bh_off_V + (kv_row_start + k_r)[:, None] * 128 + (k_c_half + 64)[None, :] * 1
        V_1 = tl.load(V_1_ptrs, mask=(kv_row_start + k_r)[:, None] < S, other=0.0)
        
        O_acc_0 += tl.dot(exp_P_bf16, V_0)
        O_acc_1 += tl.dot(exp_P_bf16, V_1)
                
        running_max = new_max
    
    O_acc_0 /= running_sum[:, None]
    O_acc_1 /= running_sum[:, None]
    
    lse = running_max + tl.log(running_sum)
    
    row_mask = (q_row_start + r) < S
    safe_lse = tl.where(row_mask, lse, 0.0)
    
    c_0 = tl.arange(0, 64)
    O_0 = O_acc_0.to(tl.bfloat16)
    O_0_ptrs = O_ptr + bh_off_O + (q_row_start + r)[:, None] * 128 + c_0[None, :] * 1
    tl.store(O_0_ptrs, O_0, mask=(q_row_start + r)[:, None] < S)
    
    c_1 = tl.arange(0, 64)
    O_1 = O_acc_1.to(tl.bfloat16)
    O_1_ptrs = O_ptr + bh_off_O + (q_row_start + r)[:, None] * 128 + (c_1 + 64)[None, :] * 1
    tl.store(O_1_ptrs, O_1, mask=(q_row_start + r)[:, None] < S)
    
    lse_ptr = LSE_ptr + bh * stride_LSE_h + q_row_start
    tl.store(lse_ptr + r, safe_lse, mask=row_mask)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    scale = 1.0 / (D ** 0.5)
    
    num_blocks = triton.cdiv(S, 128)
    grid = (num_blocks, B * H)
    
    kernel[grid](
        Q, K, V, O, LSE,
        S, scale,
        Q.stride(1), K.stride(1), V.stride(1), O.stride(1),
        LSE.stride(1),
        num_warps=4, num_stages=3,
    )