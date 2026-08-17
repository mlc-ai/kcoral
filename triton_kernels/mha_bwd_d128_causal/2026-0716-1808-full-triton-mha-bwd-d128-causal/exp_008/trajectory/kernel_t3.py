import torch
import triton
import triton.language as tl
import math


@triton.jit
def load_2d_tile(base, bh, row_start, col_start, S, max_row, max_col, stride_b, stride_r):
    row_idx = row_start + tl.arange(0, max_row)
    col_idx = col_start + tl.arange(0, max_col)
    
    tile_base = base + bh * stride_b + row_idx[:, None] * stride_r + col_idx[None, :] * 1
    
    mask_row = row_idx[:, None] < S
    
    my_tile = tl.load(tile_base, mask=mask_row, other=0.0)
    return my_tile


@triton.jit
def bwd_dQ_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dQ_ptr,
    S, scale, num_blocks,
    stride_Q_b, stride_K_b, stride_V_b, stride_O_b, stride_dO_b, stride_dQ_b,
):
    bh = tl.program_id(1)
    q_blk = tl.program_id(0)
    
    q_rows_0 = q_blk * 128 + tl.arange(0, 64)
    q_rows_1 = q_blk * 128 + 64 + tl.arange(0, 64)
    
    Q_00 = load_2d_tile(Q_ptr, bh, q_blk * 128, 0, S, 64, 64, stride_Q_b, 128)
    Q_01 = load_2d_tile(Q_ptr, bh, q_blk * 128, 64, S, 64, 64, stride_Q_b, 128)
    Q_10 = load_2d_tile(Q_ptr, bh, q_blk * 128 + 64, 0, S, 64, 64, stride_Q_b, 128)
    Q_11 = load_2d_tile(Q_ptr, bh, q_blk * 128 + 64, 64, S, 64, 64, stride_Q_b, 128)
    
    O_00 = load_2d_tile(O_ptr, bh, q_blk * 128, 0, S, 64, 64, stride_O_b, 128)
    O_01 = load_2d_tile(O_ptr, bh, q_blk * 128, 64, S, 64, 64, stride_O_b, 128)
    O_10 = load_2d_tile(O_ptr, bh, q_blk * 128 + 64, 0, S, 64, 64, stride_O_b, 128)
    O_11 = load_2d_tile(O_ptr, bh, q_blk * 128 + 64, 64, S, 64, 64, stride_O_b, 128)
    
    dO_00 = load_2d_tile(dO_ptr, bh, q_blk * 128, 0, S, 64, 64, stride_dO_b, 128)
    dO_01 = load_2d_tile(dO_ptr, bh, q_blk * 128, 64, S, 64, 64, stride_dO_b, 128)
    dO_10 = load_2d_tile(dO_ptr, bh, q_blk * 128 + 64, 0, S, 64, 64, stride_dO_b, 128)
    dO_11 = load_2d_tile(dO_ptr, bh, q_blk * 128 + 64, 64, S, 64, 64, stride_dO_b, 128)
    
    D_0 = tl.sum(O_00 * dO_00, -1, keep_dims=True) + tl.sum(O_01 * dO_01, -1, keep_dims=True)
    D_1 = tl.sum(O_10 * dO_10, -1, keep_dims=True) + tl.sum(O_11 * dO_11, -1, keep_dims=True)
    
    L_0 = tl.load(L_ptr + bh * S + q_rows_0, mask=q_rows_0 < S, other=0.0)
    L_1 = tl.load(L_ptr + bh * S + q_rows_1, mask=q_rows_1 < S, other=0.0)
    
    dQ_0 = 0.0
    dQ_1 = 0.0
    dQ_2 = 0.0
    dQ_3 = 0.0
    
    for k_blk in range(0, q_blk + 1):
        k_rows_0 = k_blk * 128 + tl.arange(0, 64)
        k_rows_1 = k_blk * 128 + 64 + tl.arange(0, 64)
        
        K_00 = load_2d_tile(K_ptr, bh, k_blk * 128, 0, S, 64, 64, stride_K_b, 128)
        K_01 = load_2d_tile(K_ptr, bh, k_blk * 128, 64, S, 64, 64, stride_K_b, 128)
        K_10 = load_2d_tile(K_ptr, bh, k_blk * 128 + 64, 0, S, 64, 64, stride_K_b, 128)
        K_11 = load_2d_tile(K_ptr, bh, k_blk * 128 + 64, 64, S, 64, 64, stride_K_b, 128)
        
        V_00 = load_2d_tile(V_ptr, bh, k_blk * 128, 0, S, 64, 64, stride_V_b, 128)
        V_01 = load_2d_tile(V_ptr, bh, k_blk * 128, 64, S, 64, 64, stride_V_b, 128)
        V_10 = load_2d_tile(V_ptr, bh, k_blk * 128 + 64, 0, S, 64, 64, stride_V_b, 128)
        V_11 = load_2d_tile(V_ptr, bh, k_blk * 128 + 64, 64, S, 64, 64, stride_V_b, 128)
        
        S_00 = tl.dot(Q_00, K_00.T)
        S_00 = tl.dot(Q_01, K_01.T, acc=S_00)
        S_01 = tl.dot(Q_00, K_10.T)
        S_01 = tl.dot(Q_01, K_11.T, acc=S_01)
        S_10 = tl.dot(Q_10, K_00.T)
        S_10 = tl.dot(Q_11, K_01.T, acc=S_10)
        S_11 = tl.dot(Q_10, K_10.T)
        S_11 = tl.dot(Q_11, K_11.T, acc=S_11)
        
        dP_00 = tl.dot(dO_00, V_00.T)
        dP_00 = tl.dot(dO_01, V_01.T, acc=dP_00)
        dP_01 = tl.dot(dO_00, V_10.T)
        dP_01 = tl.dot(dO_01, V_11.T, acc=dP_01)
        dP_10 = tl.dot(dO_10, V_00.T)
        dP_10 = tl.dot(dO_11, V_01.T, acc=dP_10)
        dP_11 = tl.dot(dO_10, V_10.T)
        dP_11 = tl.dot(dO_11, V_11.T, acc=dP_11)
        
        mask00 = (q_rows_0[:, None] >= k_rows_0[None, :])
        mask01 = (q_rows_0[:, None] >= k_rows_1[None, :])
        mask10 = (q_rows_1[:, None] >= k_rows_0[None, :])
        mask11 = (q_rows_1[:, None] >= k_rows_1[None, :])
        
        S_00 = S_00 * scale - L_0[None, :]
        S_01 = S_01 * scale - L_0[None, :]
        S_10 = S_10 * scale - L_1[None, :]
        S_11 = S_11 * scale - L_1[None, :]
        
        P_00 = tl.where(mask00, tl.exp(S_00), 0.0)
        P_01 = tl.where(mask01, tl.exp(S_01), 0.0)
        P_10 = tl.where(mask10, tl.exp(S_10), 0.0)
        P_11 = tl.where(mask11, tl.exp(S_11), 0.0)
            
        dS_00 = P_00 * (dP_00 - D_0) * scale
        dS_01 = P_01 * (dP_01 - D_0) * scale
        dS_10 = P_10 * (dP_10 - D_1) * scale
        dS_11 = P_11 * (dP_11 - D_1) * scale
        
        dQ_0 = tl.dot(dS_00, K_00, acc=dQ_0)
        dQ_0 = tl.dot(dS_01, K_10, acc=dQ_0)
        dQ_1 = tl.dot(dS_00, K_01, acc=dQ_1)
        dQ_1 = tl.dot(dS_01, K_11, acc=dQ_1)
        
        dQ_2 = tl.dot(dS_10, K_00, acc=dQ_2)
        dQ_2 = tl.dot(dS_11, K_10, acc=dQ_2)
        dQ_3 = tl.dot(dS_10, K_01, acc=dQ_3)
        dQ_3 = tl.dot(dS_11, K_11, acc=dQ_3)
        
    ptr_Q0 = dQ_ptr + bh * stride_dQ_b + q_rows_0[:, None] * 128 + tl.arange(0, 64)[None, :]
    ptr_Q1 = dQ_ptr + bh * stride_dQ_b + q_rows_0[:, None] * 128 + 64 + tl.arange(0, 64)[None, :]
    ptr_Q2 = dQ_ptr + bh * stride_dQ_b + q_rows_1[:, None] * 128 + tl.arange(0, 64)[None, :]
    ptr_Q3 = dQ_ptr + bh * stride_dQ_b + q_rows_1[:, None] * 128 + 64 + tl.arange(0, 64)[None, :]
    
    tl.store(ptr_Q0, dQ_0.to(tl.bfloat16), mask=q_rows_0[:, None] < S)
    tl.store(ptr_Q1, dQ_1.to(tl.bfloat16), mask=q_rows_0[:, None] < S)
    tl.store(ptr_Q2, dQ_2.to(tl.bfloat16), mask=q_rows_1[:, None] < S)
    tl.store(ptr_Q3, dQ_3.to(tl.bfloat16), mask=q_rows_1[:, None] < S)


@triton.jit
def bwd_dKV_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dK_ptr, dV_ptr,
    S, scale, num_blocks,
    stride_Q_b, stride_K_b, stride_V_b, stride_O_b, stride_dO_b, stride_dK_b, stride_dV_b,
):
    bh = tl.program_id(1)
    k_blk = tl.program_id(0)
    
    k_rows_0 = k_blk * 128 + tl.arange(0, 64)
    k_rows_1 = k_blk * 128 + 64 + tl.arange(0, 64)
    
    K_00 = load_2d_tile(K_ptr, bh, k_blk * 128, 0, S, 64, 64, stride_K_b, 128)
    K_01 = load_2d_tile(K_ptr, bh, k_blk * 128, 64, S, 64, 64, stride_K_b, 128)
    K_10 = load_2d_tile(K_ptr, bh, k_blk * 128 + 64, 0, S, 64, 64, stride_K_b, 128)
    K_11 = load_2d_tile(K_ptr, bh, k_blk * 128 + 64, 64, S, 64, 64, stride_K_b, 128)
    
    V_00 = load_2d_tile(V_ptr, bh, k_blk * 128, 0, S, 64, 64, stride_V_b, 128)
    V_01 = load_2d_tile(V_ptr, bh, k_blk * 128, 64, S, 64, 64, stride_V_b, 128)
    V_10 = load_2d_tile(V_ptr, bh, k_blk * 128 + 64, 0, S, 64, 64, stride_V_b, 128)
    V_11 = load_2d_tile(V_ptr, bh, k_blk * 128 + 64, 64, S, 64, 64, stride_V_b, 128)
    
    d_K_0 = 0.0
    d_K_1 = 0.0
    d_K_2 = 0.0
    d_K_3 = 0.0
    
    d_V_0 = 0.0
    d_V_1 = 0.0
    d_V_2 = 0.0
    d_V_3 = 0.0
    
    for q_blk in range(k_blk, num_blocks):
        q_rows_0 = q_blk * 128 + tl.arange(0, 64)
        q_rows_1 = q_blk * 128 + 64 + tl.arange(0, 64)
        
        Q_00 = load_2d_tile(Q_ptr, bh, q_blk * 128, 0, S, 64, 64, stride_Q_b, 128)
        Q_01 = load_2d_tile(Q_ptr, bh, q_blk * 128, 64, S, 64, 64, stride_Q_b, 128)
        Q_10 = load_2d_tile(Q_ptr, bh, q_blk * 128 + 64, 0, S, 64, 64, stride_Q_b, 128)
        Q_11 = load_2d_tile(Q_ptr, bh, q_blk * 128 + 64, 64, S, 64, 64, stride_Q_b, 128)
        
        O_00 = load_2d_tile(O_ptr, bh, q_blk * 128, 0, S, 64, 64, stride_O_b, 128)
        O_01 = load_2d_tile(O_ptr, bh, q_blk * 128, 64, S, 64, 64, stride_O_b, 128)
        O_10 = load_2d_tile(O_ptr, bh, q_blk * 128 + 64, 0, S, 64, 64, stride_O_b, 128)
        O_11 = load_2d_tile(O_ptr, bh, q_blk * 128 + 64, 64, S, 64, 64, stride_O_b, 128)
        
        dO_00 = load_2d_tile(dO_ptr, bh, q_blk * 128, 0, S, 64, 64, stride_dO_b, 128)
        dO_01 = load_2d_tile(dO_ptr, bh, q_blk * 128, 64, S, 64, 64, stride_dO_b, 128)
        dO_10 = load_2d_tile(dO_ptr, bh, q_blk * 128 + 64, 0, S, 64, 64, stride_dO_b, 128)
        dO_11 = load_2d_tile(dO_ptr, bh, q_blk * 128 + 64, 64, S, 64, 64, stride_dO_b, 128)
        
        D_0 = tl.sum(O_00 * dO_00, -1, keep_dims=True) + tl.sum(O_01 * dO_01, -1, keep_dims=True)
        D_1 = tl.sum(O_10 * dO_10, -1, keep_dims=True) + tl.sum(O_11 * dO_11, -1, keep_dims=True)
        
        L_0 = tl.load(L_ptr + bh * S + q_rows_0, mask=q_rows_0 < S, other=0.0)
        L_1 = tl.load(L_ptr + bh * S + q_rows_1, mask=q_rows_1 < S, other=0.0)
        
        S_00 = tl.dot(Q_00, K_00.T)
        S_00 = tl.dot(Q_01, K_01.T, acc=S_00)
        S_01 = tl.dot(Q_00, K_10.T)
        S_01 = tl.dot(Q_01, K_11.T, acc=S_01)
        S_10 = tl.dot(Q_10, K_00.T)
        S_10 = tl.dot(Q_11, K_01.T, acc=S_10)
        S_11 = tl.dot(Q_10, K_10.T)
        S_11 = tl.dot(Q_11, K_11.T, acc=S_11)
        
        dP_00 = tl.dot(dO_00, V_00.T)
        dP_00 = tl.dot(dO_01, V_01.T, acc=dP_00)
        dP_01 = tl.dot(dO_00, V_10.T)
        dP_01 = tl.dot(dO_01, V_11.T, acc=dP_01)
        dP_10 = tl.dot(dO_10, V_00.T)
        dP_10 = tl.dot(dO_11, V_01.T, acc=dP_10)
        dP_11 = tl.dot(dO_10, V_10.T)
        dP_11 = tl.dot(dO_11, V_11.T, acc=dP_11)
        
        mask00 = (q_rows_0[:, None] >= k_rows_0[None, :])
        mask01 = (q_rows_0[:, None] >= k_rows_1[None, :])
        mask10 = (q_rows_1[:, None] >= k_rows_0[None, :])
        mask11 = (q_rows_1[:, None] >= k_rows_1[None, :])
        
        S_00 = S_00 * scale - L_0[None, :]
        S_01 = S_01 * scale - L_0[None, :]
        S_10 = S_10 * scale - L_1[None, :]
        S_11 = S_11 * scale - L_1[None, :]
        
        P_00 = tl.where(mask00, tl.exp(S_00), 0.0)
        P_01 = tl.where(mask01, tl.exp(S_01), 0.0)
        P_10 = tl.where(mask10, tl.exp(S_10), 0.0)
        P_11 = tl.where(mask11, tl.exp(S_11), 0.0)
            
        dS_00 = P_00 * (dP_00 - D_0) * scale
        dS_01 = P_01 * (dP_01 - D_0) * scale
        dS_10 = P_10 * (dP_10 - D_1) * scale
        dS_11 = P_11 * (dP_11 - D_1) * scale
        
        d_K_0 = tl.dot(dS_00.T, Q_00, acc=d_K_0)
        d_K_0 = tl.dot(dS_10.T, Q_10, acc=d_K_0)
        d_K_1 = tl.dot(dS_00.T, Q_01, acc=d_K_1)
        d_K_1 = tl.dot(dS_10.T, Q_11, acc=d_K_1)
        d_K_2 = tl.dot(dS_01.T, Q_00, acc=d_K_2)
        d_K_2 = tl.dot(dS_11.T, Q_10, acc=d_K_2)
        d_K_3 = tl.dot(dS_01.T, Q_01, acc=d_K_3)
        d_K_3 = tl.dot(dS_11.T, Q_11, acc=d_K_3)
        
        d_V_0 = tl.dot(P_00.T, dO_00, acc=d_V_0)
        d_V_0 = tl.dot(P_10.T, dO_10, acc=d_V_0)
        d_V_1 = tl.dot(P_00.T, dO_01, acc=d_V_1)
        d_V_1 = tl.dot(P_10.T, dO_11, acc=d_V_1)
        d_V_2 = tl.dot(P_01.T, dO_00, acc=d_V_2)
        d_V_2 = tl.dot(P_11.T, dO_10, acc=d_V_2)
        d_V_3 = tl.dot(P_01.T, dO_01, acc=d_V_3)
        d_V_3 = tl.dot(P_11.T, dO_11, acc=d_V_3)
        
    ptr_K0 = dK_ptr + bh * stride_dK_b + k_rows_0[:, None] * 128 + tl.arange(0, 64)[None, :]
    ptr_K1 = dK_ptr + bh * stride_dK_b + k_rows_0[:, None] * 128 + 64 + tl.arange(0, 64)[None, :]
    ptr_K2 = dK_ptr + bh * stride_dK_b + k_rows_1[:, None] * 128 + tl.arange(0, 64)[None, :]
    ptr_K3 = dK_ptr + bh * stride_dK_b + k_rows_1[:, None] * 128 + 64 + tl.arange(0, 64)[None, :]
    
    ptr_V0 = dV_ptr + bh * stride_dV_b + k_rows_0[:, None] * 128 + tl.arange(0, 64)[None, :]
    ptr_V1 = dV_ptr + bh * stride_dV_b + k_rows_0[:, None] * 128 + 64 + tl.arange(0, 64)[None, :]
    ptr_V2 = dV_ptr + bh * stride_dV_b + k_rows_1[:, None] * 128 + tl.arange(0, 64)[None, :]
    ptr_V3 = dV_ptr + bh * stride_dV_b + k_rows_1[:, None] * 128 + 64 + tl.arange(0, 64)[None, :]
    
    tl.store(ptr_K0, d_K_0.to(tl.bfloat16), mask=k_rows_0[:, None] < S)
    tl.store(ptr_K1, d_K_1.to(tl.bfloat16), mask=k_rows_0[:, None] < S)
    tl.store(ptr_K2, d_K_2.to(tl.bfloat16), mask=k_rows_1[:, None] < S)
    tl.store(ptr_K3, d_K_3.to(tl.bfloat16), mask=k_rows_1[:, None] < S)
    
    tl.store(ptr_V0, d_V_0.to(tl.bfloat16), mask=k_rows_0[:, None] < S)
    tl.store(ptr_V1, d_V_1.to(tl.bfloat16), mask=k_rows_0[:, None] < S)
    tl.store(ptr_V2, d_V_2.to(tl.bfloat16), mask=k_rows_1[:, None] < S)
    tl.store(ptr_V3, d_V_3.to(tl.bfloat16), mask=k_rows_1[:, None] < S)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    
    scale = 1.0 / math.sqrt(d)
    
    num_blocks = (S + 127) // 128
    grid = (num_blocks, B * H)
    
    stride_Q_b = Q.stride(1)
    stride_K_b = K.stride(1)
    stride_V_b = V.stride(1)
    stride_O_b = O.stride(1)
    stride_dO_b = dO.stride(1)
    stride_dQ_b = dQ.stride(1)
    stride_dK_b = dK.stride(1)
    stride_dV_b = dV.stride(1)
    
    bwd_dKV_kernel[grid](
        Q_ptr=Q, K_ptr=K, V_ptr=V, O_ptr=O, dO_ptr=dO, L_ptr=L,
        dK_ptr=dK, dV_ptr=dV, S=S, scale=scale, num_blocks=num_blocks,
        stride_Q_b=stride_Q_b, stride_K_b=stride_K_b, stride_V_b=stride_V_b,
        stride_O_b=stride_O_b, stride_dO_b=stride_dO_b, stride_dK_b=stride_dK_b,
        stride_dV_b=stride_dV_b,
        num_stages=3
    )
    
    bwd_dQ_kernel[grid](
        Q_ptr=Q, K_ptr=K, V_ptr=V, O_ptr=O, dO_ptr=dO, L_ptr=L,
        dQ_ptr=dQ, S=S, scale=scale, num_blocks=num_blocks,
        stride_Q_b=stride_Q_b, stride_K_b=stride_K_b, stride_V_b=stride_V_b,
        stride_O_b=stride_O_b, stride_dO_b=stride_dO_b, stride_dQ_b=stride_dQ_b,
        num_stages=3
    )