import torch
import triton
import triton.language as tl


@triton.jit
def _compute_D_kernel(
    dO_ptr,
    O_ptr,
    D_ptr,
    N,  
    HEAD_DIM,
    BLOCK: tl.constexpr,
):
    base_idx = tl.program_id(0) * BLOCK
    idx = base_idx + tl.arange(0, BLOCK)
    mask = idx < N
    
    sum_val = tl.zeros((BLOCK,), tl.float32)
    for chunk in range(0, HEAD_DIM, 64):
        off = idx[:, None] * HEAD_DIM + chunk + tl.arange(0, 64)[None, :]
        val_do = tl.load(dO_ptr + off, mask=(idx[:, None] < N), other=0.0)
        val_o = tl.load(O_ptr + off, mask=(idx[:, None] < N), other=0.0)
        sum_val += (val_do * val_o).sum(axis=1)
    
    tl.store(D_ptr + idx, sum_val, mask=mask)


@triton.jit
def _bwd_Q_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, D_ptr, dQ_ptr,
    B, H, S_len, scale, NUM_SMS: tl.constexpr,
):
    num_pid_s = tl.cdiv(S_len, 128)
    total_tiles = B * H * num_pid_s
    grid_size = min(NUM_SMS, total_tiles)
    
    for i in tl.range(tl.program_id(0), total_tiles, grid_size):
        b_h_idx = i // num_pid_s
        q_idx = i % num_pid_s
        q_idx_base = q_idx * 128
        
        s_offs_0 = q_idx_base + tl.arange(0, 128)
        
        q_base_0 = tl.make_block_ptr(Q_ptr + b_h_idx * S_len * 128, 
            shape=[S_len, 128], strides=[128, 1], 
            offsets=[q_idx_base, 0], block_shape=[128, 64], order=[1, 0])
        q_base_1 = tl.make_block_ptr(Q_ptr + b_h_idx * S_len * 128, 
            shape=[S_len, 128], strides=[128, 1], 
            offsets=[q_idx_base, 64], block_shape=[128, 64], order=[1, 0])
        
        do_base_0 = tl.make_block_ptr(dO_ptr + b_h_idx * S_len * 128, 
            shape=[S_len, 128], strides=[128, 1], 
            offsets=[q_idx_base, 0], block_shape=[128, 64], order=[1, 0])
        do_base_1 = tl.make_block_ptr(dO_ptr + b_h_idx * S_len * 128, 
            shape=[S_len, 128], strides=[128, 1], 
            offsets=[q_idx_base, 64], block_shape=[128, 64], order=[1, 0])
        
        Q_0 = tl.cast(tl.load(q_base_0, boundary_check=(0,)), tl.float32)
        Q_1 = tl.cast(tl.load(q_base_1, boundary_check=(0,)), tl.float32)
        dO_0 = tl.cast(tl.load(do_base_0, boundary_check=(0,)), tl.float32)
        dO_1 = tl.cast(tl.load(do_base_1, boundary_check=(0,)), tl.float32)
        
        mask_l_0 = s_offs_0 < S_len
        l_offs_0 = b_h_idx * S_len + s_offs_0
        L_val_0 = tl.load(L_ptr + l_offs_0, mask=mask_l_0, other=0.0)
        D_val_0 = tl.load(D_ptr + l_offs_0, mask=mask_l_0, other=0.0)
        
        acc_dQ_0 = tl.zeros((128, 64), tl.float32)
        acc_dQ_1 = tl.zeros((128, 64), tl.float32)
        
        k_base_0 = tl.make_block_ptr(K_ptr + b_h_idx * S_len * 128, 
            shape=[S_len, 128], strides=[128, 1], 
            offsets=[0, 0], block_shape=[128, 64], order=[1, 0])
        v_base_0 = tl.make_block_ptr(V_ptr + b_h_idx * S_len * 128, 
            shape=[S_len, 128], strides=[128, 1], 
            offsets=[0, 0], block_shape=[128, 64], order=[1, 0])
        
        for kv_idx_base in range(0, S_len, 128):
            K_0 = tl.cast(tl.load(tl.advance(k_base_0, [kv_idx_base, 0]), boundary_check=(0,)), tl.float32)
            K_1 = tl.cast(tl.load(tl.advance(k_base_0, [kv_idx_base, 64]), boundary_check=(0,)), tl.float32)
            V_0 = tl.cast(tl.load(tl.advance(v_base_0, [kv_idx_base, 0]), boundary_check=(0,)), tl.float32)
            V_1 = tl.cast(tl.load(tl.advance(v_base_0, [kv_idx_base, 64]), boundary_check=(0,)), tl.float32)
            
            S_scores = (tl.dot(Q_0, K_0.T) + tl.dot(Q_1, K_1.T)) * scale
            P = tl.exp(S_scores - L_val_0[:, None])
            
            dP = (tl.dot(dO_0, V_0.T) + tl.dot(dO_1, V_1.T))
            dS = P * (dP - D_val_0[:, None]) * scale
            
            acc_dQ_0 += tl.dot(dS, K_0)
            acc_dQ_1 += tl.dot(dS, K_1)
            
        mask_q_0 = s_offs_0 < S_len
        store_offs_0 = (b_h_idx * S_len + s_offs_0[:, None]) * 128 + tl.arange(0, 64)[None, :]
        tl.store(dQ_ptr + store_offs_0, acc_dQ_0.to(tl.bfloat16), mask=mask_q_0[:, None])
        
        store_offs_1 = (b_h_idx * S_len + s_offs_0[:, None]) * 128 + (64 + tl.arange(0, 64))[None, :]
        tl.store(dQ_ptr + store_offs_1, acc_dQ_1.to(tl.bfloat16), mask=mask_q_0[:, None])


@triton.jit
def _bwd_KV_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, D_ptr, dK_ptr, dV_ptr,
    B, H, S_len, scale, NUM_SMS: tl.constexpr,
):
    num_pid_s = tl.cdiv(S_len, 128)
    total_tiles = B * H * num_pid_s
    grid_size = min(NUM_SMS, total_tiles)
    
    for i in tl.range(tl.program_id(0), total_tiles, grid_size):
        kv_idx = i // (B * H)
        b_h_idx = i % (B * H)
        kv_idx_base = kv_idx * 128
        
        s_offs_0 = kv_idx_base + tl.arange(0, 128)
        
        k_base_0 = tl.make_block_ptr(K_ptr + b_h_idx * S_len * 128, 
            shape=[S_len, 128], strides=[128, 1], 
            offsets=[kv_idx_base, 0], block_shape=[128, 64], order=[1, 0])
        k_base_1 = tl.make_block_ptr(K_ptr + b_h_idx * S_len * 128, 
            shape=[S_len, 128], strides=[128, 1], 
            offsets=[kv_idx_base, 64], block_shape=[128, 64], order=[1, 0])
        v_base_0 = tl.make_block_ptr(V_ptr + b_h_idx * S_len * 128, 
            shape=[S_len, 128], strides=[128, 1], 
            offsets=[kv_idx_base, 0], block_shape=[128, 64], order=[1, 0])
        v_base_1 = tl.make_block_ptr(V_ptr + b_h_idx * S_len * 128, 
            shape=[S_len, 128], strides=[128, 1], 
            offsets=[kv_idx_base, 64], block_shape=[128, 64], order=[1, 0])
        
        K_0 = tl.cast(tl.load(k_base_0, boundary_check=(0,)), tl.float32)
        K_1 = tl.cast(tl.load(k_base_1, boundary_check=(0,)), tl.float32)
        V_0 = tl.cast(tl.load(v_base_0, boundary_check=(0,)), tl.float32)
        V_1 = tl.cast(tl.load(v_base_1, boundary_check=(0,)), tl.float32)
        
        acc_dK_0 = tl.zeros((128, 64), tl.float32)
        acc_dK_1 = tl.zeros((128, 64), tl.float32)
        acc_dV_0 = tl.zeros((128, 64), tl.float32)
        acc_dV_1 = tl.zeros((128, 64), tl.float32)
        
        q_base_0 = tl.make_block_ptr(Q_ptr + b_h_idx * S_len * 128, 
            shape=[S_len, 128], strides=[128, 1], 
            offsets=[0, 0], block_shape=[128, 64], order=[1, 0])
        q_base_1 = tl.make_block_ptr(Q_ptr + b_h_idx * S_len * 128, 
            shape=[S_len, 128], strides=[128, 1], 
            offsets=[0, 64], block_shape=[128, 64], order=[1, 0])
        do_base_0 = tl.make_block_ptr(dO_ptr + b_h_idx * S_len * 128, 
            shape=[S_len, 128], strides=[128, 1], 
            offsets=[0, 0], block_shape=[128, 64], order=[1, 0])
        do_base_1 = tl.make_block_ptr(dO_ptr + b_h_idx * S_len * 128, 
            shape=[S_len, 128], strides=[128, 1], 
            offsets=[0, 64], block_shape=[128, 64], order=[1, 0])
        
        for q_idx_base in range(0, S_len, 128):
            Q_0 = tl.cast(tl.load(tl.advance(q_base_0, [q_idx_base, 0]), boundary_check=(0,)), tl.float32)
            Q_1 = tl.cast(tl.load(tl.advance(q_base_1, [q_idx_base, 0]), boundary_check=(0,)), tl.float32)
            dO_0 = tl.cast(tl.load(tl.advance(do_base_0, [q_idx_base, 0]), boundary_check=(0,)), tl.float32)
            dO_1 = tl.cast(tl.load(tl.advance(do_base_1, [q_idx_base, 0]), boundary_check=(0,)), tl.float32)
            
            q_offs_0 = q_idx_base + tl.arange(0, 128)
            mask_l_0 = q_offs_0 < S_len
            l_offs_0 = b_h_idx * S_len + q_offs_0
            L_val_0 = tl.load(L_ptr + l_offs_0, mask=mask_l_0, other=0.0)
            D_val_0 = tl.load(D_ptr + l_offs_0, mask=mask_l_0, other=0.0)
            
            S_scores = (tl.dot(Q_0, K_0.T) + tl.dot(Q_1, K_1.T)) * scale
            P = tl.exp(S_scores - L_val_0[:, None])
            
            dP = (tl.dot(dO_0, V_0.T) + tl.dot(dO_1, V_1.T))
            dS = P * (dP - D_val_0[:, None]) * scale
            
            dS_T = dS.T
            P_T = P.T
            
            acc_dK_0 += tl.dot(dS_T, Q_0)
            acc_dK_1 += tl.dot(dS_T, Q_1)
            acc_dV_0 += tl.dot(P_T, dO_0)
            acc_dV_1 += tl.dot(P_T, dO_1)
            
        mask_kv_0 = s_offs_0 < S_len
        store_offs_0 = (b_h_idx * S_len + s_offs_0[:, None]) * 128 + tl.arange(0, 64)[None, :]
        tl.store(dK_ptr + store_offs_0, acc_dK_0.to(tl.bfloat16), mask=mask_kv_0[:, None])
        tl.store(dV_ptr + store_offs_0, acc_dV_0.to(tl.bfloat16), mask=mask_kv_0[:, None])
        
        store_offs_1 = (b_h_idx * S_len + s_offs_0[:, None]) * 128 + (64 + tl.arange(0, 64))[None, :]
        tl.store(dK_ptr + store_offs_1, acc_dK_1.to(tl.bfloat16), mask=mask_kv_0[:, None])
        tl.store(dV_ptr + store_offs_1, acc_dV_1.to(tl.bfloat16), mask=mask_kv_0[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Execute optimized multi-head attention backward pass."""
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    scale = 1.0 / (d ** 0.5)
    
    D = torch.empty((B, H, S), dtype=torch.float32, device=Q.device)
    
    grid_D = (triton.cdiv(B * H * S, 128),)
    _compute_D_kernel[grid_D](
        dO, O, D, B * H * S, 128, BLOCK=128
    )
    
    num_pid_s = triton.cdiv(S, 128)
    total_tiles = B * H * num_pid_s
    grid_size = min(132, total_tiles)
    grid = (grid_size,)
    
    _bwd_Q_kernel[grid](
        Q, K, V, dO, L, D, dQ,
        B, H, S, scale, NUM_SMS=132,
        num_warps=8, num_stages=3
    )
    
    _bwd_KV_kernel[grid](
        Q, K, V, dO, L, D, dK, dV,
        B, H, S, scale, NUM_SMS=132,
        num_warps=8, num_stages=3
    )