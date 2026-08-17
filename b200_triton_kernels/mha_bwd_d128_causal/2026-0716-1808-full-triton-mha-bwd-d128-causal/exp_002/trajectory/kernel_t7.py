import torch
import triton
import triton.language as tl
import math

B = 4
H = 48
d = 128
BLOCK = 64

@triton.jit
def load_tile(ptr, row_start, col_start, row_mask, stride_row, stride_col):
    s_off_r = tl.arange(0, 64)
    s_off_c = tl.arange(0, 64)
    off = ptr + row_start * stride_row + col_start * stride_col
    row_off = s_off_r[:, None] * stride_row
    col_off = s_off_c[None, :] * stride_col
    ptrs = off + row_off + col_off
    col_mask = (col_start + s_off_c) < 128
    return tl.load(ptrs, mask=row_mask[:, None] & col_mask[None, :], other=0.0)

@triton.jit
def store_tile(acc, ptr, row_start, col_start, row_mask, stride_row, stride_col):
    s_off_r = tl.arange(0, 64)
    s_off_c = tl.arange(0, 64)
    off = ptr + row_start * stride_row + col_start * stride_col
    row_off = s_off_r[:, None] * stride_row
    col_off = s_off_c[None, :] * stride_col
    ptrs = off + row_off + col_off
    col_mask = (col_start + s_off_c) < 128
    tl.store(ptrs, acc.to(tl.bfloat16), mask=row_mask[:, None] & col_mask[None, :])

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
    
    Q_0 = load_tile(Q_ptr + base_offset, i_start, 0, s_mask_i, stride_s, stride_d)
    Q_1 = load_tile(Q_ptr + base_offset, i_start, 64, s_mask_i, stride_s, stride_d)
    O_0 = load_tile(O_ptr + base_offset, i_start, 0, s_mask_i, stride_s, stride_d)
    O_1 = load_tile(O_ptr + base_offset, i_start, 64, s_mask_i, stride_s, stride_d)
    dO_0 = load_tile(dO_ptr + base_offset, i_start, 0, s_mask_i, stride_s, stride_d)
    dO_1 = load_tile(dO_ptr + base_offset, i_start, 64, s_mask_i, stride_s, stride_d)
    
    D_0 = tl.sum(O_0 * dO_0, 1)
    D_1 = tl.sum(O_1 * dO_1, 1)
    D = D_0 + D_1
    
    L_expanded = tl.load(L_ptr + base_offset_L + i_start + s_off_i, mask=s_mask_i, other=0.0)
    
    acc_dQ_0 = tl.zeros((64, 64), tl.float32)
    acc_dQ_1 = tl.zeros((64, 64), tl.float32)
    
    for j_tile in range(i_tile + 1):
        j_start = j_tile * 64
        s_mask_j = (j_start + s_off_j) < S
        
        K_0 = load_tile(K_ptr + base_offset, j_start, 0, s_mask_j, stride_s, stride_d)
        K_1 = load_tile(K_ptr + base_offset, j_start, 64, s_mask_j, stride_s, stride_d)
        V_0 = load_tile(V_ptr + base_offset, j_start, 0, s_mask_j, stride_s, stride_d)
        V_1 = load_tile(V_ptr + base_offset, j_start, 64, s_mask_j, stride_s, stride_d)
        
        S_val = tl.dot(Q_0, K_0.T) + tl.dot(Q_1, K_1.T)
        S_val = S_val * scale
        
        mask_curr = ((i_start + s_off_i[:, None]) >= (j_start + s_off_j[None, :])) & s_mask_i[:, None] & s_mask_j[None, :]
        A_curr = tl.where(mask_curr, tl.exp(S_val - L_expanded[:, None]), 0.0)
        
        dP_curr = tl.dot(dO_0, V_0.T) + tl.dot(dO_1, V_1.T)
        
        dA_curr = A_curr * (dP_curr - D[:, None])
        
        acc_dQ_0 = tl.dot(dA_curr, K_0, acc_dQ_0)
        acc_dQ_1 = tl.dot(dA_curr, K_1, acc_dQ_1)
        
    acc_dQ_0 = acc_dQ_0 * scale
    acc_dQ_1 = acc_dQ_1 * scale
    
    store_tile(acc_dQ_0, dQ_ptr + base_offset, i_start, 0, s_mask_i, stride_s, stride_d)
    store_tile(acc_dQ_1, dQ_ptr + base_offset, i_start, 64, s_mask_i, stride_s, stride_d)


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
    
    K_0 = load_tile(K_ptr + base_offset, j_start, 0, s_mask_j, stride_s, stride_d)
    K_1 = load_tile(K_ptr + base_offset, j_start, 64, s_mask_j, stride_s, stride_d)
    V_0 = load_tile(V_ptr + base_offset, j_start, 0, s_mask_j, stride_s, stride_d)
    V_1 = load_tile(V_ptr + base_offset, j_start, 64, s_mask_j, stride_s, stride_d)
    
    acc_dK_0 = tl.zeros((64, 64), tl.float32)
    acc_dK_1 = tl.zeros((64, 64), tl.float32)
    acc_dV_0 = tl.zeros((64, 64), tl.float32)
    acc_dV_1 = tl.zeros((64, 64), tl.float32)
    
    num_tiles = (S + 64 - 1) // 64
    
    for i_tile in range(num_tiles - 1, j_tile, -1):
        i_start = i_tile * 64
        s_mask_i = (i_start + s_off_i) < S
        
        Q_0 = load_tile(Q_ptr + base_offset, i_start, 0, s_mask_i, stride_s, stride_d)
        Q_1 = load_tile(Q_ptr + base_offset, i_start, 64, s_mask_i, stride_s, stride_d)
        O_0 = load_tile(O_ptr + base_offset, i_start, 0, s_mask_i, stride_s, stride_d)
        O_1 = load_tile(O_ptr + base_offset, i_start, 64, s_mask_i, stride_s, stride_d)
        dO_0 = load_tile(dO_ptr + base_offset, i_start, 0, s_mask_i, stride_s, stride_d)
        dO_1 = load_tile(dO_ptr + base_offset, i_start, 64, s_mask_i, stride_s, stride_d)
        
        D_0 = tl.sum(O_0 * dO_0, 1)
        D_1 = tl.sum(O_1 * dO_1, 1)
        D = D_0 + D_1
        
        L_expanded = tl.load(L_ptr + base_offset_L + i_start + s_off_i, mask=s_mask_i, other=0.0)
        
        S_val = tl.dot(Q_0, K_0.T) + tl.dot(Q_1, K_1.T)
        S_val = S_val * scale
        
        mask_curr = ((i_start + s_off_i[:, None]) >= (j_start + s_off_j[None, :])) & s_mask_i[:, None] & s_mask_j[None, :]
        A_curr = tl.where(mask_curr, tl.exp(S_val - L_expanded[:, None]), 0.0)
        
        dP_curr = tl.dot(dO_0, V_0.T) + tl.dot(dO_1, V_1.T)
        
        dA_curr = A_curr * (dP_curr - D[:, None])
        
        acc_dK_0 = tl.dot(dA_curr.T, Q_0, acc_dK_0)
        acc_dK_1 = tl.dot(dA_curr.T, Q_1, acc_dK_1)
        
        acc_dV_0 = tl.dot(A_curr.T, dO_0, acc_dV_0)
        acc_dV_1 = tl.dot(A_curr.T, dO_1, acc_dV_1)
        
    curr_start_q = min(j_tile * 64, S - 1)
    s_mask_i = (curr_start_q + s_off_i) < S
    
    Q_0 = load_tile(Q_ptr + base_offset, curr_start_q, 0, s_mask_i, stride_s, stride_d)
    Q_1 = load_tile(Q_ptr + base_offset, curr_start_q, 64, s_mask_i, stride_s, stride_d)
    O_0 = load_tile(O_ptr + base_offset, curr_start_q, 0, s_mask_i, stride_s, stride_d)
    O_1 = load_tile(O_ptr + base_offset, curr_start_q, 64, s_mask_i, stride_s, stride_d)
    dO_0 = load_tile(dO_ptr + base_offset, curr_start_q, 0, s_mask_i, stride_s, stride_d)
    dO_1 = load_tile(dO_ptr + base_offset, curr_start_q, 64, s_mask_i, stride_s, stride_d)
    
    D_0 = tl.sum(O_0 * dO_0, 1)
    D_1 = tl.sum(O_1 * dO_1, 1)
    D = D_0 + D_1
    
    L_expanded = tl.load(L_ptr + base_offset_L + curr_start_q + s_off_i, mask=s_mask_i, other=0.0)
    
    S_val = tl.dot(Q_0, K_0.T) + tl.dot(Q_1, K_1.T)
    S_val = S_val * scale
    
    mask_curr = ((curr_start_q + s_off_i[:, None]) >= (j_start + s_off_j[None, :])) & s_mask_i[:, None] & s_mask_j[None, :]
    A_curr = tl.where(mask_curr, tl.exp(S_val - L_expanded[:, None]), 0.0)
    
    dP_curr = tl.dot(dO_0, V_0.T) + tl.dot(dO_1, V_1.T)
    
    dA_curr = A_curr * (dP_curr - D[:, None])
    
    acc_dK_0 = tl.dot(dA_curr.T, Q_0, acc_dK_0)
    acc_dK_1 = tl.dot(dA_curr.T, Q_1, acc_dK_1)
    
    acc_dV_0 = tl.dot(A_curr.T, dO_0, acc_dV_0)
    acc_dV_1 = tl.dot(A_curr.T, dO_1, acc_dV_1)
        
    acc_dK_0 = acc_dK_0 * scale
    acc_dK_1 = acc_dK_1 * scale
    
    store_tile(acc_dK_0, dK_ptr + base_offset, j_start, 0, s_mask_j, stride_s, stride_d)
    store_tile(acc_dK_1, dK_ptr + base_offset, j_start, 64, s_mask_j, stride_s, stride_d)
    store_tile(acc_dV_0, dV_ptr + base_offset, j_start, 0, s_mask_j, stride_s, stride_d)
    store_tile(acc_dV_1, dV_ptr + base_offset, j_start, 64, s_mask_j, stride_s, stride_d)


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
    
    num_tiles = (S + BLOCK - 1) // BLOCK
    
    grid = (B, H, num_tiles)
    
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