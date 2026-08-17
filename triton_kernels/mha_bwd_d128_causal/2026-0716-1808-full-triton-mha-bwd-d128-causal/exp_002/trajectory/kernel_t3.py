import torch
import triton
import triton.language as tl
import math

B = 4
H = 48
d = 128

@triton.jit
def load_tile(ptr, base_offset, row_start, col_start, row_mask, stride_row, stride_col):
    off = base_offset + row_start * stride_row + col_start * stride_col
    row_off = tl.arange(0, 64)[:, None] * stride_row
    col_off = tl.arange(0, 64)[None, :] * stride_col
    ptrs = off + row_off + col_off
    col_mask = (col_start + tl.arange(0, 64)) < 128
    return tl.load(ptrs, mask=row_mask[:, None] & col_mask[None, :], other=0.0, padding_option="zero")

@triton.jit
def store_tile(acc, ptr, base_offset, row_start, col_start, row_mask, stride_row, stride_col):
    off = base_offset + row_start * stride_row + col_start * stride_col
    row_off = tl.arange(0, 64)[:, None] * stride_row
    col_off = tl.arange(0, 64)[None, :] * stride_col
    ptrs = off + row_off + col_off
    col_mask = (col_start + tl.arange(0, 64)) < 128
    tl.store(ptrs, acc.to(tl.bfloat16), mask=row_mask[:, None] & col_mask[None, :], eviction_policy="evict_first")

@triton.jit
def phase_1_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dQ_ptr,
    S, scale,
    stride_b, stride_h, stride_s, stride_d
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    i_tile = tl.program_id(2)
    
    s_off_i = tl.arange(0, 64)
    s_off_j = tl.arange(0, 64)
    i_start = i_tile * 64
    
    s_mask_i = (i_start + s_off_i) < S
    
    base_offset = b * stride_b + h * stride_h
    base_offset_L = b * (H * S) + h * S
    
    Q_0 = load_tile(Q_ptr, base_offset, i_start, 0, s_mask_i, stride_s, stride_d)
    Q_1 = load_tile(Q_ptr, base_offset, i_start, 64, s_mask_i, stride_s, stride_d)
    O_0 = load_tile(O_ptr, base_offset, i_start, 0, s_mask_i, stride_s, stride_d)
    O_1 = load_tile(O_ptr, base_offset, i_start, 64, s_mask_i, stride_s, stride_d)
    dO_0 = load_tile(dO_ptr, base_offset, i_start, 0, s_mask_i, stride_s, stride_d)
    dO_1 = load_tile(dO_ptr, base_offset, i_start, 64, s_mask_i, stride_s, stride_d)
    
    D_O_0 = O_0 * dO_0
    D_O_1 = O_1 * dO_1
    
    L_expanded = tl.load(L_ptr + base_offset_L + i_start + s_off_i, mask=s_mask_i, other=0.0)
    
    acc_dQ_0 = tl.zeros((64, 64), tl.float32)
    acc_dQ_1 = tl.zeros((64, 64), tl.float32)
    
    for j_tile in range(i_tile + 1):
        j_start = j_tile * 64
        s_mask_j = (j_start + s_off_j) < S
        
        K_0 = load_tile(K_ptr, base_offset, j_start, 0, s_mask_j, stride_s, stride_d)
        K_1 = load_tile(K_ptr, base_offset, j_start, 64, s_mask_j, stride_s, stride_d)
        V_0 = load_tile(V_ptr, base_offset, j_start, 0, s_mask_j, stride_s, stride_d)
        V_1 = load_tile(V_ptr, base_offset, j_start, 64, s_mask_j, stride_s, stride_d)
        
        S_val = tl.dot(Q_0, K_0) + tl.dot(Q_1, K_1)
        S_val = S_val * scale
        
        mask_curr = ((i_start + s_off_i[:, None]) >= (j_start + s_off_j[None, :])) & s_mask_i[:, None] & s_mask_j[None, :]
        A_curr = tl.where(mask_curr, tl.exp(S_val - L_expanded[:, None]), 0.0)
        
        dP_curr_0 = dO_0 @ V_0
        dP_curr_1 = dO_1 @ V_1
        
        dA_curr_0 = A_curr * (dP_curr_0 - D_O_0)
        dA_curr_1 = A_curr * (dP_curr_1 - D_O_1)
        
        acc_dQ_0 = acc_dQ_0 + dA_curr_0 @ K_0
        acc_dQ_1 = acc_dQ_1 + dA_curr_1 @ K_1
        
    acc_dQ_0 = acc_dQ_0 * scale
    acc_dQ_1 = acc_dQ_1 * scale
    
    ptr_dQ = dQ_ptr + base_offset
    store_tile(acc_dQ_0, ptr_dQ, i_start, 0, s_mask_i, stride_s, stride_d)
    store_tile(acc_dQ_1, ptr_dQ, i_start, 64, s_mask_i, stride_s, stride_d)


@triton.jit
def phase_2_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dK_ptr, dV_ptr,
    S, scale,
    stride_b, stride_h, stride_s, stride_d
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    j_tile = tl.program_id(2)
    
    s_off_i = tl.arange(0, 64)
    s_off_j = tl.arange(0, 64)
    j_start = j_tile * 64
    
    s_mask_j = (j_start + s_off_j) < S
    
    base_offset = b * stride_b + h * stride_h
    base_offset_L = b * (H * S) + h * S
    
    K_0 = load_tile(K_ptr, base_offset, j_start, 0, s_mask_j, stride_s, stride_d)
    K_1 = load_tile(K_ptr, base_offset, j_start, 64, s_mask_j, stride_s, stride_d)
    V_0 = load_tile(V_ptr, base_offset, j_start, 0, s_mask_j, stride_s, stride_d)
    V_1 = load_tile(V_ptr, base_offset, j_start, 64, s_mask_j, stride_s, stride_d)
    
    acc_dK_0 = tl.zeros((64, 64), tl.float32)
    acc_dK_1 = tl.zeros((64, 64), tl.float32)
    acc_dV_0 = tl.zeros((64, 64), tl.float32)
    acc_dV_1 = tl.zeros((64, 64), tl.float32)
    
    num_tiles = (S + 64 - 1) // 64
    
    for i_tile in range(j_tile, num_tiles):
        i_start = i_tile * 64
        s_mask_i = (i_start + s_off_i) < S
        
        Q_0 = load_tile(Q_ptr, base_offset, i_start, 0, s_mask_i, stride_s, stride_d)
        Q_1 = load_tile(Q_ptr, base_offset, i_start, 64, s_mask_i, stride_s, stride_d)
        O_0 = load_tile(O_ptr, base_offset, i_start, 0, s_mask_i, stride_s, stride_d)
        O_1 = load_tile(O_ptr, base_offset, i_start, 64, s_mask_i, stride_s, stride_d)
        dO_0 = load_tile(dO_ptr, base_offset, i_start, 0, s_mask_i, stride_s, stride_d)
        dO_1 = load_tile(dO_ptr, base_offset, i_start, 64, s_mask_i, stride_s, stride_d)
        
        D_O_0 = O_0 * dO_0
        D_O_1 = O_1 * dO_1
        
        L_expanded = tl.load(L_ptr + base_offset_L + i_start + s_off_i, mask=s_mask_i, other=0.0)
        
        S_val = tl.dot(Q_0, K_0) + tl.dot(Q_1, K_1)
        S_val = S_val * scale
        
        mask_curr = ((i_start + s_off_i[:, None]) >= (j_start + s_off_j[None, :])) & s_mask_i[:, None] & s_mask_j[None, :]
        A_curr = tl.where(mask_curr, tl.exp(S_val - L_expanded[:, None]), 0.0)
        
        dP_curr_0 = dO_0 @ V_0
        dP_curr_1 = dO_1 @ V_1
        
        dA_curr_0 = A_curr * (dP_curr_0 - D_O_0)
        dA_curr_1 = A_curr * (dP_curr_1 - D_O_1)
        
        acc_dK_0 = acc_dK_0 + dA_curr_0.T @ Q_0
        acc_dK_1 = acc_dK_1 + dA_curr_1.T @ Q_1
        acc_dV_0 = acc_dV_0 + A_curr.T @ dO_0
        acc_dV_1 = acc_dV_1 + A_curr.T @ dO_1
        
    acc_dK_0 = acc_dK_0 * scale
    acc_dK_1 = acc_dK_1 * scale
    
    ptr_dK = dK_ptr + base_offset
    ptr_dV = dV_ptr + base_offset
    store_tile(acc_dK_0, ptr_dK, j_start, 0, s_mask_j, stride_s, stride_d)
    store_tile(acc_dK_1, ptr_dK, j_start, 64, s_mask_j, stride_s, stride_d)
    store_tile(acc_dV_0, ptr_dV, j_start, 0, s_mask_j, stride_s, stride_d)
    store_tile(acc_dV_1, ptr_dV, j_start, 64, s_mask_j, stride_s, stride_d)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    S = Q.shape[2]
    
    dQ = dQ.to(torch.bfloat16)
    dK = dK.to(torch.bfloat16)
    dV = dV.to(torch.bfloat16)
    
    scale = 1.0 / math.sqrt(d)
    
    stride_b = H * S * d
    stride_h = S * d
    stride_s = d
    stride_d = 1
    
    num_tiles = (S + 64 - 1) // 64
    
    grid = (num_tiles, B, H)
    
    phase_1_kernel[grid](
        Q, K, V, O, dO, L, dQ,
        S, scale,
        stride_b, stride_h, stride_s, stride_d,
        num_warps=4, num_stages=2,
    )
    
    phase_2_kernel[grid](
        Q, K, V, O, dO, L, dK, dV,
        S, scale,
        stride_b, stride_h, stride_s, stride_d,
        num_warps=4, num_stages=2,
    )