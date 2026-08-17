import torch
import triton
import triton.language as tl
import math


@triton.jit
def bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr_flat, dQ_ptr,
    S_len, sqrt_d,
):
    NUM_STAGES: tl.constexpr = 3
    
    q_tile = tl.program_id(0)
    b_h = tl.program_id(1)
    
    q_offset = q_tile * 128
    if q_offset >= S_len:
        return
    
    q_rows_start = q_offset
    
    Q0_buf = [None] * NUM_STAGES
    Q1_buf = [None] * NUM_STAGES
    dO0_buf = [None] * NUM_STAGES
    dO1_buf = [None] * NUM_STAGES
    K0_buf = [None] * NUM_STAGES
    K1_buf = [None] * NUM_STAGES
    V0_buf = [None] * NUM_STAGES
    V1_buf = [None] * NUM_STAGES
    
    num_k_tiles = tl.cdiv(S_len, 128)
    
    for k_tile in range(min(num_k_tiles, NUM_STAGES)):
        Q0_buf[k_tile] = tl.load(Q_ptr + (b_h * S_len + q_rows_start + tl.arange(0, 128)[:, None]) * 128 + tl.arange(0, 64)[None, :], mask=(q_rows_start + tl.arange(0, 128))[:, None] < S_len, other=0.0)
        Q1_buf[k_tile] = tl.load(Q_ptr + (b_h * S_len + q_rows_start + tl.arange(0, 128)[:, None]) * 128 + (tl.arange(0, 64)[None, :] + 64), mask=(q_rows_start + tl.arange(0, 128))[:, None] < S_len, other=0.0)
        dO0_buf[k_tile] = tl.load(dO_ptr + (b_h * S_len + q_rows_start + tl.arange(0, 128)[:, None]) * 128 + tl.arange(0, 64)[None, :], mask=(q_rows_start + tl.arange(0, 128))[:, None] < S_len, other=0.0)
        dO1_buf[k_tile] = tl.load(dO_ptr + (b_h * S_len + q_rows_start + tl.arange(0, 128)[:, None]) * 128 + (tl.arange(0, 64)[None, :] + 64), mask=(q_rows_start + tl.arange(0, 128))[:, None] < S_len, other=0.0)
        
        k_offset = k_tile * 128
        K0_buf[k_tile] = tl.load(K_ptr + (b_h * S_len + k_offset + tl.arange(0, 128)[:, None]) * 128 + tl.arange(0, 64)[None, :], mask=(k_offset + tl.arange(0, 128))[:, None] < S_len, other=0.0)
        K1_buf[k_tile] = tl.load(K_ptr + (b_h * S_len + k_offset + tl.arange(0, 128)[:, None]) * 128 + (tl.arange(0, 64)[None, :] + 64), mask=(k_offset + tl.arange(0, 128))[:, None] < S_len, other=0.0)
        V0_buf[k_tile] = tl.load(V_ptr + (b_h * S_len + k_offset + tl.arange(0, 128)[:, None]) * 128 + tl.arange(0, 64)[None, :], mask=(k_offset + tl.arange(0, 128))[:, None] < S_len, other=0.0)
        V1_buf[k_tile] = tl.load(V_ptr + (b_h * S_len + k_offset + tl.arange(0, 128)[:, None]) * 128 + (tl.arange(0, 64)[None, :] + 64), mask=(k_offset + tl.arange(0, 128))[:, None] < S_len, other=0.0)
        
    L_q = tl.load(L_ptr_flat + b_h * S_len + q_rows_start + tl.arange(0, 128), mask=(q_rows_start + tl.arange(0, 128)) < S_len, other=0.0)
    L_q_expanded = L_q[:, None]
    
    acc_dQ0 = tl.zeros((128, 64), tl.float32)
    acc_dQ1 = tl.zeros((128, 64), tl.float32)
    
    for k_tile in range(num_k_tiles):
        k_offset = k_tile * 128
        if k_offset >= S_len:
            break
            
        buf_idx = k_tile % NUM_STAGES
        Q0 = Q0_buf[buf_idx]
        Q1 = Q1_buf[buf_idx]
        dO0 = dO0_buf[buf_idx]
        dO1 = dO1_buf[buf_idx]
        K0 = K0_buf[buf_idx]
        K1 = K1_buf[buf_idx]
        V0 = V0_buf[buf_idx]
        V1 = V1_buf[buf_idx]
        
        S_acc = tl.zeros((128, 128), tl.float32)
        d_acc = tl.zeros((128, 128), tl.float32)
        
        S_acc += tl.dot(Q0, K0.T)
        d_acc += tl.dot(dO0, V0.T)
        S_acc += tl.dot(Q1, K1.T)
        d_acc += tl.dot(dO1, V1.T)
        
        S_scaled = S_acc * sqrt_d
        k_rows = k_offset + tl.arange(0, 128)
        valid_mask = k_rows[None, :] < S_len
        S_scaled = tl.where(valid_mask, S_scaled, -1e20)
        
        P = tl.exp(S_scaled - L_q_expanded)
        P = tl.where(valid_mask, P, 0.0)
        
        ds = d_acc * P
        
        acc_dQ0 += tl.dot(ds, K0)
        acc_dQ1 += tl.dot(ds, K1)
        
        next_load_idx = k_tile + NUM_STAGES
        if next_load_idx < num_k_tiles:
            next_buf_idx = next_load_idx % NUM_STAGES
            next_k_offset = next_load_idx * 128
            
            K0_buf[next_buf_idx] = tl.load(K_ptr + (b_h * S_len + next_k_offset + tl.arange(0, 128)[:, None]) * 128 + tl.arange(0, 64)[None, :], mask=(next_k_offset + tl.arange(0, 128))[:, None] < S_len, other=0.0)
            K1_buf[next_buf_idx] = tl.load(K_ptr + (b_h * S_len + next_k_offset + tl.arange(0, 128)[:, None]) * 128 + (tl.arange(0, 64)[None, :] + 64), mask=(next_k_offset + tl.arange(0, 128))[:, None] < S_len, other=0.0)
            V0_buf[next_buf_idx] = tl.load(V_ptr + (b_h * S_len + next_k_offset + tl.arange(0, 128)[:, None]) * 128 + tl.arange(0, 64)[None, :], mask=(next_k_offset + tl.arange(0, 128))[:, None] < S_len, other=0.0)
            V1_buf[next_buf_idx] = tl.load(V_ptr + (b_h * S_len + next_k_offset + tl.arange(0, 128)[:, None]) * 128 + (tl.arange(0, 64)[None, :] + 64), mask=(next_k_offset + tl.arange(0, 128))[:, None] < S_len, other=0.0)
            
    out_ptr_h0 = dQ_ptr + (b_h * S_len + q_rows_start + tl.arange(0, 128)[:, None]) * 128 + tl.arange(0, 64)[None, :]
    out_ptr_h1 = out_ptr_h0 + 64
    
    mask_q = (q_rows_start + tl.arange(0, 128))[:, None] < S_len
    
    tl.store(out_ptr_h0, (acc_dQ0 * sqrt_d).to(tl.bfloat16), mask=mask_q)
    tl.store(out_ptr_h1, (acc_dQ1 * sqrt_d).to(tl.bfloat16), mask=mask_q)


@triton.jit
def bwd_dk_dv_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr_flat, dK_ptr, dV_ptr,
    S_len, sqrt_d,
):
    NUM_STAGES: tl.constexpr = 3
    
    k_tile = tl.program_id(0)
    b_h = tl.program_id(1)
    
    k_offset = k_tile * 128
    if k_offset >= S_len:
        return
        
    k_rows = k_offset + tl.arange(0, 128)

    K0_buf = [None] * NUM_STAGES
    K1_buf = [None] * NUM_STAGES
    V0_buf = [None] * NUM_STAGES
    V1_buf = [None] * NUM_STAGES
    Q0_buf = [None] * NUM_STAGES
    Q1_buf = [None] * NUM_STAGES
    dO0_buf = [None] * NUM_STAGES
    dO1_buf = [None] * NUM_STAGES
    
    num_q_tiles = tl.cdiv(S_len, 128)
    
    for q_tile in range(min(num_q_tiles, NUM_STAGES)):
        Q0_buf[q_tile] = tl.load(Q_ptr + (b_h * S_len + q_tile * 128 + tl.arange(0, 128)[:, None]) * 128 + tl.arange(0, 64)[None, :], mask=(q_tile * 128 + tl.arange(0, 128))[:, None] < S_len, other=0.0)
        Q1_buf[q_tile] = tl.load(Q_ptr + (b_h * S_len + q_tile * 128 + tl.arange(0, 128)[:, None]) * 128 + (tl.arange(0, 64)[None, :] + 64), mask=(q_tile * 128 + tl.arange(0, 128))[:, None] < S_len, other=0.0)
        dO0_buf[q_tile] = tl.load(dO_ptr + (b_h * S_len + q_tile * 128 + tl.arange(0, 128)[:, None]) * 128 + tl.arange(0, 64)[None, :], mask=(q_tile * 128 + tl.arange(0, 128))[:, None] < S_len, other=0.0)
        dO1_buf[q_tile] = tl.load(dO_ptr + (b_h * S_len + q_tile * 128 + tl.arange(0, 128)[:, None]) * 128 + (tl.arange(0, 64)[None, :] + 64), mask=(q_tile * 128 + tl.arange(0, 128))[:, None] < S_len, other=0.0)
        
        K0_buf[q_tile] = tl.load(K_ptr + (b_h * S_len + k_offset + tl.arange(0, 128)[:, None]) * 128 + tl.arange(0, 64)[None, :], mask=(k_offset + tl.arange(0, 128))[:, None] < S_len, other=0.0)
        K1_buf[q_tile] = tl.load(K_ptr + (b_h * S_len + k_offset + tl.arange(0, 128)[:, None]) * 128 + (tl.arange(0, 64)[None, :] + 64), mask=(k_offset + tl.arange(0, 128))[:, None] < S_len, other=0.0)
        V0_buf[q_tile] = tl.load(V_ptr + (b_h * S_len + k_offset + tl.arange(0, 128)[:, None]) * 128 + tl.arange(0, 64)[None, :], mask=(k_offset + tl.arange(0, 128))[:, None] < S_len, other=0.0)
        V1_buf[q_tile] = tl.load(V_ptr + (b_h * S_len + k_offset + tl.arange(0, 128)[:, None]) * 128 + (tl.arange(0, 64)[None, :] + 64), mask=(k_offset + tl.arange(0, 128))[:, None] < S_len, other=0.0)
        
    L_k = tl.load(L_ptr_flat + b_h * S_len + k_rows, mask=k_rows < S_len, other=0.0)
    L_k_expanded = L_k[:, None]
    
    acc_dK0 = tl.zeros((128, 64), tl.float32)
    acc_dK1 = tl.zeros((128, 64), tl.float32)
    acc_dV0 = tl.zeros((128, 64), tl.float32)
    acc_dV1 = tl.zeros((128, 64), tl.float32)
    
    for q_tile in range(num_q_tiles):
        q_offset = q_tile * 128
        if q_offset >= S_len:
            break
            
        buf_idx = q_tile % NUM_STAGES
        Q0 = Q0_buf[buf_idx]
        Q1 = Q1_buf[buf_idx]
        dO0 = dO0_buf[buf_idx]
        dO1 = dO1_buf[buf_idx]
        K0 = K0_buf[buf_idx]
        K1 = K1_buf[buf_idx]
        V0 = V0_buf[buf_idx]
        V1 = V1_buf[buf_idx]
        
        S_acc = tl.zeros((128, 128), tl.float32)
        d_acc = tl.zeros((128, 128), tl.float32)
        
        S_acc += tl.dot(Q0, K0.T)
        d_acc += tl.dot(dO0, V0.T)
        S_acc += tl.dot(Q1, K1.T)
        d_acc += tl.dot(dO1, V1.T)
        
        S_scaled = S_acc * sqrt_d
        q_rows = q_offset + tl.arange(0, 128)
        valid_mask = q_rows[:, None] < S_len
        S_scaled = tl.where(valid_mask, S_scaled, -1e20)
        
        L_q = tl.load(L_ptr_flat + b_h * S_len + q_rows, mask=q_rows < S_len, other=0.0)
        L_q_expanded = L_q[:, None]
        
        P = tl.exp(S_scaled - L_q_expanded)
        valid_k = k_rows[None, :] < S_len
        P = tl.where(valid_mask & valid_k, P, 0.0)
        
        ds = d_acc * P
        
        acc_dK0 += tl.dot(ds.T, Q0)
        acc_dK1 += tl.dot(ds.T, Q1)
        acc_dV0 += tl.dot(P.T, dO0)
        acc_dV1 += tl.dot(P.T, dO1)
        
        next_load_idx = q_tile + NUM_STAGES
        if next_load_idx < num_q_tiles:
            next_buf_idx = next_load_idx % NUM_STAGES
            next_q_offset = next_load_idx * 128
            
            Q0_buf[next_buf_idx] = tl.load(Q_ptr + (b_h * S_len + next_q_offset + tl.arange(0, 128)[:, None]) * 128 + tl.arange(0, 64)[None, :], mask=(next_q_offset + tl.arange(0, 128))[:, None] < S_len, other=0.0)
            Q1_buf[next_buf_idx] = tl.load(Q_ptr + (b_h * S_len + next_q_offset + tl.arange(0, 128)[:, None]) * 128 + (tl.arange(0, 64)[None, :] + 64), mask=(next_q_offset + tl.arange(0, 128))[:, None] < S_len, other=0.0)
            dO0_buf[next_buf_idx] = tl.load(dO_ptr + (b_h * S_len + next_q_offset + tl.arange(0, 128)[:, None]) * 128 + tl.arange(0, 64)[None, :], mask=(next_q_offset + tl.arange(0, 128))[:, None] < S_len, other=0.0)
            dO1_buf[next_buf_idx] = tl.load(dO_ptr + (b_h * S_len + next_q_offset + tl.arange(0, 128)[:, None]) * 128 + (tl.arange(0, 64)[None, :] + 64), mask=(next_q_offset + tl.arange(0, 128))[:, None] < S_len, other=0.0)
            
    out_ptr_k_h0 = dK_ptr + (b_h * S_len + k_offset + tl.arange(0, 128)[:, None]) * 128 + tl.arange(0, 64)[None, :]
    out_ptr_k_h1 = out_ptr_k_h0 + 64
    out_ptr_v_h0 = dV_ptr + (b_h * S_len + k_offset + tl.arange(0, 128)[:, None]) * 128 + tl.arange(0, 64)[None, :]
    out_ptr_v_h1 = out_ptr_v_h0 + 64
    
    mask_k = (k_offset + tl.arange(0, 128))[:, None] < S_len
    
    tl.store(out_ptr_k_h0, (acc_dK0 * sqrt_d).to(tl.bfloat16), mask=mask_k)
    tl.store(out_ptr_k_h1, (acc_dK1 * sqrt_d).to(tl.bfloat16), mask=mask_k)
    tl.store(out_ptr_v_h0, acc_dV0.to(tl.bfloat16), mask=mask_k)
    tl.store(out_ptr_v_h1, acc_dV1.to(tl.bfloat16), mask=mask_k)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S_len, d = Q.shape
    sqrt_d = 1.0 / math.sqrt(d)
    
    Q_ptr = Q.flatten(0, 2)
    K_ptr = K.flatten(0, 2)
    V_ptr = V.flatten(0, 2)
    dO_ptr = dO.flatten(0, 2)
    L_ptr = L.flatten(0, 1)
    dQ_ptr = dQ.flatten(0, 2)
    dK_ptr = dK.flatten(0, 2)
    dV_ptr = dV.flatten(0, 2)
    
    grid_dq = (triton.cdiv(S_len, 128), B * H)
    grid_dk_dv = (triton.cdiv(S_len, 128), B * H)

    bwd_dq_kernel[grid_dq](
        Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, dQ_ptr,
        S_len, sqrt_d,
        num_warps=8, num_stages=3)
    
    bwd_dk_dv_kernel[grid_dk_dv](
        Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, dK_ptr, dV_ptr,
        S_len, sqrt_d,
        num_warps=8, num_stages=3)