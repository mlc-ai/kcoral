import torch
import triton
import triton.language as tl


@triton.jit
def P_kernel(
    Q_ptr, K_ptr, p_buf_ptr, local_max_buf_ptr, g_max_ptr, g_sum_ptr,
    S, scale, stride_Q_bh, stride_K_bh, num_blocks,
):
    k_block_id = tl.program_id(0)
    q_block_id = tl.program_id(1)
    bh = tl.program_id(2)
    
    global_q_start = q_block_id * 256
    kv_start = k_block_id * 256
    
    bh_off_Q = bh * stride_Q_bh
    bh_off_K = bh * stride_K_bh
    
    r = tl.arange(0, 256)
    c = tl.arange(0, 256)
    
    q_ptr = Q_ptr + bh_off_Q + (global_q_start + r)[:, None] * 128 + c[None, :] * 1
    q_tile = tl.load(q_ptr, mask=(global_q_start + r)[:, None] < S, other=0.0)
    
    k_ptr = K_ptr + bh_off_K + (kv_start + r)[:, None] * 128 + c[None, :] * 1
    k_tile = tl.load(k_ptr, mask=(kv_start + r)[:, None] < S, other=0.0)
    
    p_local = tl.dot(q_tile, k_tile.T)
    p_local *= scale
    
    causal_mask = (global_q_start + r[:, None]) >= (kv_start + c[None, :])
    query_valid = (global_q_start + r) < S
    mask = query_valid[:, None] & causal_mask
    p_local = tl.where(mask, p_local, -1e38)
    
    local_max = tl.max(p_local, axis=1)
    p_local_scaled = p_local - local_max[:, None]
    exp_p = tl.where(mask, tl.exp(p_local_scaled), 0.0)
    local_sum = tl.sum(exp_p, axis=1)
    
    p_ptr = p_buf_ptr + (q_block_id * num_blocks + k_block_id) * 256 * 256 + r[:, None] * 256 + c[None, :]
    tl.store(p_ptr, p_local_scaled, mask=query_valid[:, None])
    
    lm_ptr = local_max_buf_ptr + (q_block_id * num_blocks + k_block_id) * 256 + r
    tl.store(lm_ptr, local_max, mask=query_valid)
    
    q_id = global_q_start // 256
    old_max = tl.load(g_max_ptr + q_id)
    new_max = tl.maximum(old_max, local_max)
    factor = tl.exp(old_max - new_max)
    
    tl.atomic_add(g_sum_ptr + q_id, g_sum_ptr[q_id] * (factor - 1))
    tl.atomic_max(g_max_ptr + q_id, new_max)
    
    adjusted_sum = local_sum * tl.exp(local_max - new_max)
    tl.atomic_add(g_sum_ptr + q_id, adjusted_sum)


@triton.jit
def P_rescale_kernel(
    p_buf_ptr, local_max_buf_ptr, g_max_ptr, g_sum_ptr, LSE_ptr,
    S, stride_LSE_b, stride_LSE_h, stride_LSE_s, num_blocks,
):
    k_block_id = tl.program_id(0)
    q_block_id = tl.program_id(1)
    bh = tl.program_id(2)
    
    global_q_start = q_block_id * 256
    
    r = tl.arange(0, 256)
    c = tl.arange(0, 256)
    
    p_ptr = p_buf_ptr + (q_block_id * num_blocks + k_block_id) * 256 * 256 + r[:, None] * 256 + c[None, :]
    p_local = tl.load(p_ptr, mask=(global_q_start + r)[:, None] < S, other=0.0)
    
    lm_ptr = local_max_buf_ptr + (q_block_id * num_blocks + k_block_id) * 256 + r
    local_max = tl.load(lm_ptr, mask=(global_q_start + r) < S, other=-1e38)
    
    q_id = global_q_start // 256
    g_max_val = tl.load(g_max_ptr + q_id)
    g_sum_val = tl.load(g_sum_ptr + q_id)
    
    final_p = tl.exp(p_local + local_max[:, None] - g_max_val[:, None]) / g_sum_val[:, None]
    
    tl.store(p_ptr, final_p, mask=(global_q_start + r)[:, None] < S)
    
    if c[0] == 0:
        lse = g_max_val + tl.log(g_sum_val)
        b_idx = bh // 48
        h_idx = bh % 48
        lse_ptr = LSE_ptr + b_idx * stride_LSE_b + h_idx * stride_LSE_h + global_q_start + r * stride_LSE_s
        row_mask = (global_q_start + r) < S
        tl.store(lse_ptr, lse, mask=row_mask)


@triton.jit
def O_kernel(
    Q_ptr, V_ptr, p_buf_ptr, O_ptr, S,
    stride_Q_bh, stride_V_bh, stride_O_bh,
    stride_O_s, stride_O_d, num_blocks,
):
    q_block_id = tl.program_id(0)
    bh = tl.program_id(1)
    
    global_q_start = q_block_id * 256
    
    bh_off_Q = bh * stride_Q_bh
    bh_off_V = bh * stride_V_bh
    bh_off_O = bh * stride_O_bh
    
    r = tl.arange(0, 256)
    c = tl.arange(0, 256)
    
    q_ptr = Q_ptr + bh_off_Q + (global_q_start + r)[:, None] * 128 + c[None, :] * 1
    q_tile = tl.load(q_ptr, mask=(global_q_start + r)[:, None] < S, other=0.0)
    
    out_acc_0 = tl.zeros((256, 64), tl.float32)
    out_acc_1 = tl.zeros((256, 64), tl.float32)
    
    min_k_id = q_block_id - 1
    if min_k_id < 0:
        min_k_id = 0
        
    for k_block_id in range(min_k_id, q_block_id):
        kv_start = k_block_id * 256
        
        p_ptr = p_buf_ptr + (q_block_id * num_blocks + k_block_id) * 256 * 256 + r[:, None] * 256 + c[None, :]
        p0 = tl.load(p_ptr, mask=(global_q_start + r)[:, None] < S, other=0.0)
        p1 = tl.load(p_ptr + 64, mask=(global_q_start + r)[:, None] < S, other=0.0)
        
        c_0 = tl.arange(0, 64)
        v0_ptr = V_ptr + bh_off_V + (kv_start + r)[:, None] * 128 + c_0[None, :] * 1
        v0 = tl.load(v0_ptr, mask=(kv_start + r)[:, None] < S, other=0.0)
        
        c_1 = tl.arange(0, 64)
        v1_ptr = V_ptr + bh_off_V + (kv_start + r)[:, None] * 128 + (c_1 + 64)[None, :] * 1
        v1 = tl.load(v1_ptr, mask=(kv_start + r)[:, None] < S, other=0.0)
        
        out_acc_0 += p0 @ v0
        out_acc_1 += p1 @ v1
        
    for k_block_id in range(q_block_id, num_blocks):
        kv_start = k_block_id * 256
        
        p_ptr = p_buf_ptr + (q_block_id * num_blocks + k_block_id) * 256 * 256 + r[:, None] * 256 + c[None, :]
        p0 = tl.load(p_ptr, mask=(global_q_start + r)[:, None] < S, other=0.0)
        p1 = tl.load(p_ptr + 64, mask=(global_q_start + r)[:, None] < S, other=0.0)
        
        c_0 = tl.arange(0, 64)
        v0_ptr = V_ptr + bh_off_V + (kv_start + r)[:, None] * 128 + c_0[None, :] * 1
        v0 = tl.load(v0_ptr, mask=(kv_start + r)[:, None] < S, other=0.0)
        
        c_1 = tl.arange(0, 64)
        v1_ptr = V_ptr + bh_off_V + (kv_start + r)[:, None] * 128 + (c_1 + 64)[None, :] * 1
        v1 = tl.load(v1_ptr, mask=(kv_start + r)[:, None] < S, other=0.0)
        
        out_acc_0 += p0 @ v0
        out_acc_1 += p1 @ v1
        
    O_acc_0 = out_acc_0.to(tl.bfloat16)
    O_acc_1 = out_acc_1.to(tl.bfloat16)
    
    c_0 = tl.arange(0, 64)
    O_0_ptr = O_ptr + bh_off_O + (global_q_start + r)[:, None] * 128 + c_0[None, :] * 1
    tl.store(O_0_ptr, O_acc_0, mask=(global_q_start + r)[:, None] < S)
    
    c_1 = tl.arange(0, 64)
    O_1_ptr = O_ptr + bh_off_O + (global_q_start + r)[:, None] * 128 + (c_1 + 64)[None, :] * 1
    tl.store(O_1_ptr, O_acc_1, mask=(global_q_start + r)[:, None] < S)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    scale = 1.0 / (D ** 0.5)
    
    BLOCK_M = 256
    BLOCK_N = 256
    NUM_SMS = 132
    
    num_blocks = S // 256
    
    p_buf = torch.empty(num_blocks * num_blocks * BLOCK_M * BLOCK_N, device=Q.device, dtype=torch.float32)
    local_max_buf = torch.empty(num_blocks * num_blocks * BLOCK_M, device=Q.device, dtype=torch.float32)
    
    g_max_global = torch.full((num_blocks,), -1e38, device=Q.device, dtype=torch.float32)
    g_sum_global = torch.full((num_blocks,), 0.0, device=Q.device, dtype=torch.float32)
    
    grid_p = (min(num_blocks, NUM_SMS), min(num_blocks, NUM_SMS), B * H)
    P_kernel[grid_p](
        Q, K, p_buf, local_max_buf, g_max_global, g_sum_global,
        S, scale, Q.stride(1), K.stride(1), num_blocks,
        num_warps=8, num_stages=3,
    )
    
    grid_r = (min(num_blocks, NUM_SMS), min(num_blocks, NUM_SMS), B * H)
    P_rescale_kernel[grid_r](
        p_buf, local_max_buf, g_max_global, g_sum_global, LSE,
        S, LSE.stride(0), LSE.stride(1), LSE.stride(2), num_blocks,
        num_warps=8, num_stages=3,
    )
    
    grid_o = (num_blocks, B * H)
    O_kernel[grid_o](
        Q, V, p_buf, O, S,
        Q.stride(1), V.stride(1), O.stride(1),
        O.stride(2), O.stride(3), num_blocks,
        num_warps=8, num_stages=3,
    )