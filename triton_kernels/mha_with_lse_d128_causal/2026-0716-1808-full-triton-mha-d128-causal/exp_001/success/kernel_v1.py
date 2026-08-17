import torch
import triton
import triton.language as tl


@triton.jit
def kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    S, scale,
    stride_Q_bh, stride_K_bh, stride_V_bh, stride_O_bh, stride_LSE_bh,
    stride_Q_s, stride_K_s, stride_V_s, stride_O_s, stride_LSE_s,
):
    q_block_id = tl.program_id(0)
    bh = tl.program_id(1)
    
    q_start = q_block_id * 128
    
    bh_off_Q = bh * stride_Q_bh
    bh_off_K = bh * stride_K_bh
    bh_off_V = bh * stride_V_bh
    bh_off_O = bh * stride_O_bh
    bh_off_LSE = bh * stride_LSE_bh
    
    r = tl.arange(0, 128)
    c = tl.arange(0, 128)
    
    Q_ptrs = Q_ptr + bh_off_Q + (q_start + r)[:, None] * stride_Q_s + c[None, :]
    Q_tile = tl.load(Q_ptrs, mask=(q_start + r)[:, None] < S, other=0.0)
    
    O_acc_0 = tl.zeros((128, 64), tl.float32)
    O_acc_1 = tl.zeros((128, 64), tl.float32)
    
    running_max = tl.full((128,), -1e38, tl.float32)
    running_sum = tl.full((128,), 0.0, tl.float32)
    
    query_valid = (q_start + r) < S
    
    for k_block_id in range(q_block_id + 1):
        kv_start = k_block_id * 128
        
        K_ptrs = K_ptr + bh_off_K + (kv_start + r)[:, None] * stride_K_s + c[None, :]
        K_tile = tl.load(K_ptrs, mask=(kv_start + r)[:, None] < S, other=0.0)
        
        P = tl.dot(Q_tile, K_tile.T)
        P *= scale
        
        causal_mask = (q_start + r[:, None]) >= (kv_start + c[None, :])
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
        
        c_0 = tl.arange(0, 64)
        c_1 = tl.arange(0, 64)
        
        V_0_ptrs = V_ptr + bh_off_V + (kv_start + r)[:, None] * stride_V_s + c_0[None, :]
        V_0 = tl.load(V_0_ptrs, mask=(kv_start + r)[:, None] < S, other=0.0)
        
        V_1_ptrs = V_ptr + bh_off_V + (kv_start + r)[:, None] * stride_V_s + (c_1 + 64)[None, :]
        V_1 = tl.load(V_1_ptrs, mask=(kv_start + r)[:, None] < S, other=0.0)
        
        O_acc_0 += tl.dot(exp_P_bf16, V_0)
        O_acc_1 += tl.dot(exp_P_bf16, V_1)
                
        running_max = new_max
    
    final_scale = 1.0 / running_sum
    O_acc_0 *= final_scale[:, None]
    O_acc_1 *= final_scale[:, None]
    
    lse = running_max + tl.log(running_sum)
    
    row_mask = query_valid
    
    c_0 = tl.arange(0, 64)
    O_0 = O_acc_0.to(tl.bfloat16)
    O_0_ptrs = O_ptr + bh_off_O + (q_start + r)[:, None] * stride_O_s + c_0[None, :]
    tl.store(O_0_ptrs, O_0, mask=row_mask[:, None])
    
    c_1 = tl.arange(0, 64)
    O_1 = O_acc_1.to(tl.bfloat16)
    O_1_ptrs = O_ptr + bh_off_O + (q_start + r)[:, None] * stride_O_s + (c_1 + 64)[None, :]
    tl.store(O_1_ptrs, O_1, mask=row_mask[:, None])
    
    lse_ptr = LSE_ptr + bh_off_LSE + q_start * stride_LSE_s
    tl.store(lse_ptr + r, lse, mask=row_mask)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    scale = 1.0 / (D ** 0.5)
    
    BLOCK_M = 128
    num_blocks = triton.cdiv(S, BLOCK_M)
    grid = (num_blocks, B * H)
    
    kernel[grid](
        Q, K, V, O, LSE,
        S, scale,
        Q.stride(1), K.stride(1), V.stride(1), O.stride(1), LSE.stride(1),
        Q.stride(2), K.stride(2), V.stride(2), O.stride(2), LSE.stride(2),
        num_warps=4, num_stages=3,
    )