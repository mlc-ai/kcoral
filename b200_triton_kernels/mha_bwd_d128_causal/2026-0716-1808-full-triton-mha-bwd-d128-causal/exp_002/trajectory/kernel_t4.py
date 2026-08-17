import torch
import triton
import triton.language as tl
import math

B = 4
H = 48
d = 128

@triton.jit
def load_tile(ptr, base_offset, row_start, col_start, row_mask, stride_row, stride_col):
    s_off = tl.arange(0, 64)
    off = ptr + base_offset + row_start * stride_row + col_start * stride_col
    row_off = s_off[:, None] * stride_row
    col_off = s_off[None, :] * stride_col
    ptrs = off + row_off + col_off
    col_mask = (col_start + s_off) < 128
    return tl.load(ptrs, mask=row_mask[:, None] & col_mask[None, :], other=0.0, padding_option="zero")

@triton.jit
def store_tile(acc, ptr, base_offset, row_start, col_start, row_mask, stride_row, stride_col):
    s_off = tl.arange(0, 64)
    off = ptr + base_offset + row_start * stride_row + col_start * stride_col
    row_off = s_off[:, None] * stride_row
    col_off = s_off[None, :] * stride_col
    ptrs = off + row_off + col_off
    col_mask = (col_start + s_off) < 128
    tl.store(ptrs, acc.to(tl.bfloat16), mask=row_mask[:, None] & col_mask[None, :], eviction_policy="evict_first")

@triton.jit
def load_row_sh(sh, row_idx):
    return sh[row_idx, :]

@triton.jit
def load_col_sh(sh, col_idx):
    col = tl.zeros((64,), tl.float32)
    for i in range(64):
        col[i] = sh[i, col_idx]
    return col

@triton.jit
def gemm_64x64x64(acc, A_sh, B_sh, trans_B=False):
    for k in range(64):
        a = load_row_sh(A_sh, k)
        b = load_row_sh(B_sh, k)
        if trans_B:
            acc = tl.dot(a, b.T, acc)
        else:
            acc = tl.dot(a, b, acc)
    return acc

@triton.jit
def store_tile_sh(acc, sh, start_row, start_col, row_mask):
    for i in range(64):
        if row_mask[i]:
            for j in range(64):
                sh[start_row + i, start_col + j] = acc[i, j]

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
    
    Q_0: tl.extern_shared_array((64, 64), ty=tl.bfloat16)
    Q_1: tl.extern_shared_array((64, 64), ty=tl.bfloat16)
    O_0: tl.extern_shared_array((64, 64), ty=tl.bfloat16)
    O_1: tl.extern_shared_array((64, 64), ty=tl.bfloat16)
    dO_0: tl.extern_shared_array((64, 64), ty=tl.bfloat16)
    dO_1: tl.extern_shared_array((64, 64), ty=tl.bfloat16)
    K_0: tl.extern_shared_array((64, 64), ty=tl.bfloat16)
    K_1: tl.extern_shared_array((64, 64), ty=tl.bfloat16)
    V_0: tl.extern_shared_array((64, 64), ty=tl.bfloat16)
    V_1: tl.extern_shared_array((64, 64), ty=tl.bfloat16)
    D_O_0: tl.extern_shared_array((64, 64), ty=tl.float32)
    D_O_1: tl.extern_shared_array((64, 64), ty=tl.float32)
    dA_0:   tl.extern_shared_array((64, 64), ty=tl.float32)
    dA_1:   tl.extern_shared_array((64, 64), ty=tl.float32)

    def p1_load_tile(ptr, start, mask, sh_0, sh_1):
        load_tile_sh(ptr, base_offset, start, 0, mask, stride_s, stride_d, sh_0)
        load_tile_sh(ptr, base_offset, start, 64, mask, stride_s, stride_d, sh_1)

    load_tile_sh(Q_ptr, base_offset, i_start, 0, s_mask_i, stride_s, stride_d, Q_0)
    load_tile_sh(Q_ptr, base_offset, i_start, 64, s_mask_i, stride_s, stride_d, Q_1)
    load_tile_sh(O_ptr, base_offset, i_start, 0, s_mask_i, stride_s, stride_d, O_0)
    load_tile_sh(O_ptr, base_offset, i_start, 64, s_mask_i, stride_s, stride_d, O_1)
    load_tile_sh(dO_ptr, base_offset, i_start, 0, s_mask_i, stride_s, stride_d, dO_0)
    load_tile_sh(dO_ptr, base_offset, i_start, 64, s_mask_i, stride_s, stride_d, dO_1)
    
    D_O_0_tmp = tl.zeros((64, 64), tl.float32)
    D_O_1_tmp = tl.zeros((64, 64), tl.float32)
    for i in range(64):
        if s_mask_i[i]:
            for j in range(64):
                D_O_0_tmp[i, j] = O_0[i, j] * dO_0[i, j]
                D_O_1_tmp[i, j] = O_1[i, j] * dO_1[i, j]
                
    store_tile_sh(D_O_0_tmp, D_O_0, 0, 0, s_off_i < 64)
    store_tile_sh(D_O_1_tmp, D_O_1, 0, 0, s_off_i < 64)
    
    L_expanded = tl.load(L_ptr + base_offset_L + i_start + s_off_i, mask=s_mask_i, other=0.0)
    
    acc_dQ_0 = tl.zeros((64, 64), tl.float32)
    acc_dQ_1 = tl.zeros((64, 64), tl.float32)
    
    for j_tile in range(i_tile + 1):
        j_start = j_tile * 64
        s_mask_j = (j_start + s_off_j) < S
        
        load_tile_sh(K_ptr, base_offset, j_start, 0, s_mask_j, stride_s, stride_d, K_0)
        load_tile_sh(K_ptr, base_offset, j_start, 64, s_mask_j, stride_s, stride_d, K_1)
        load_tile_sh(V_ptr, base_offset, j_start, 0, s_mask_j, stride_s, stride_d, V_0)
        load_tile_sh(V_ptr, base_offset, j_start, 64, s_mask_j, stride_s, stride_d, V_1)
        
        S_val = tl.zeros((64, 64), tl.float32)
        S_val = gemm_64x64x64(S_val, Q_0, K_0, trans_B=True)
        S_val = gemm_64x64x64(S_val, Q_1, K_1, trans_B=True)
        S_val = S_val * scale
        
        mask_curr = ((i_start + s_off_i[:, None]) >= (j_start + s_off_j[None, :])) & s_mask_i[:, None] & s_mask_j[None, :]
        A_curr = tl.where(mask_curr, tl.exp(S_val - L_expanded[:, None]), 0.0)
        
        store_tile_sh(A_curr, K_0, 0, 0, s_off_i < 64)
        
        dP_curr = tl.zeros((64, 64), tl.float32)
        dP_curr = gemm_64x64x64(dP_curr, dO_0, V_0, trans_B=True)
        dP_curr = gemm_64x64x64(dP_curr, dO_1, V_1, trans_B=True)
        
        store_tile_sh(dP_curr, V_0, 0, 0, s_off_i < 64)
        
        dA_curr = tl.zeros((64, 64), tl.float32)
        for i in range(64):
            if s_mask_i[i]:
                for j in range(64):
                    dA_curr[i, j] = K_0[i, j] * (V_0[i, j] - D_O_0[i, j])
        
        store_tile_sh(dA_curr, dA_0, 0, 0, s_off_i < 64)
        
        dA_curr_1 = tl.zeros((64, 64), tl.float32)
        for i in range(64):
            if s_mask_i[i]:
                for j in range(64):
                    dA_curr_1[i, j] = K_1[i, j] * (V_1[i, j] - D_O_1[i, j])
                    
        store_tile_sh(dA_curr_1, dA_1, 0, 0, s_off_i < 64)
        
        acc_dQ_0 = gemm_64x64x64(acc_dQ_0, dA_0, K_0, trans_B=False)
        acc_dQ_1 = gemm_64x64x64(acc_dQ_1, dA_1, K_1, trans_B=False)
        
    acc_dQ_0 = acc_dQ_0 * scale
    acc_dQ_1 = acc_dQ_1 * scale
    
    store_tile(acc_dQ_0, dQ_ptr, base_offset, i_start, 0, s_mask_i, stride_s, stride_d)
    store_tile(acc_dQ_1, dQ_ptr, base_offset, i_start, 64, s_mask_i, stride_s, stride_d)


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
    
    K_0: tl.extern_shared_array((64, 64), ty=tl.bfloat16)
    K_1: tl.extern_shared_array((64, 64), ty=tl.bfloat16)
    V_0: tl.extern_shared_array((64, 64), ty=tl.bfloat16)
    V_1: tl.extern_shared_array((64, 64), ty=tl.bfloat16)
    Q_0: tl.extern_shared_array((64, 64), ty=tl.bfloat16)
    Q_1: tl.extern_shared_array((64, 64), ty=tl.bfloat16)
    O_0: tl.extern_shared_array((64, 64), ty=tl.bfloat16)
    O_1: tl.extern_shared_array((64, 64), ty=tl.bfloat16)
    dO_0: tl.extern_shared_array((64, 64), ty=tl.bfloat16)
    dO_1: tl.extern_shared_array((64, 64), ty=tl.bfloat16)
    D_O_0: tl.extern_shared_array((64, 64), ty=tl.float32)
    D_O_1: tl.extern_shared_array((64, 64), ty=tl.float32)
    dA_0:   tl.extern_shared_array((64, 64), ty=tl.float32)
    dA_1:   tl.extern_shared_array((64, 64), ty=tl.float32)
    A_0:    tl.extern_shared_array((64, 64), ty=tl.float32)
    A_1:    tl.extern_shared_array((64, 64), ty=tl.float32)

    def p2_load_tile(ptr, start, mask, sh_0, sh_1):
        load_tile_sh(ptr, base_offset, start, 0, mask, stride_s, stride_d, sh_0)
        load_tile_sh(ptr, base_offset, start, 64, mask, stride_s, stride_d, sh_1)

    load_tile_sh(K_ptr, base_offset, j_start, 0, s_mask_j, stride_s, stride_d, K_0)
    load_tile_sh(K_ptr, base_offset, j_start, 64, s_mask_j, stride_s, stride_d, K_1)
    load_tile_sh(V_ptr, base_offset, j_start, 0, s_mask_j, stride_s, stride_d, V_0)
    load_tile_sh(V_ptr, base_offset, j_start, 64, s_mask_j, stride_s, stride_d, V_1)
    
    acc_dK_0 = tl.zeros((64, 64), tl.float32)
    acc_dK_1 = tl.zeros((64, 64), tl.float32)
    acc_dV_0 = tl.zeros((64, 64), tl.float32)
    acc_dV_1 = tl.zeros((64, 64), tl.float32)
    
    num_tiles = (S + 64 - 1) // 64
    
    for i_tile in range(num_tiles - 1, j_tile, -1):
        i_start = i_tile * 64
        s_mask_i = (i_start + s_off_i) < S
        
        load_tile_sh(Q_ptr, base_offset, i_start, 0, s_mask_i, stride_s, stride_d, Q_0)
        load_tile_sh(Q_ptr, base_offset, i_start, 64, s_mask_i, stride_s, stride_d, Q_1)
        load_tile_sh(O_ptr, base_offset, i_start, 0, s_mask_i, stride_s, stride_d, O_0)
        load_tile_sh(O_ptr, base_offset, i_start, 64, s_mask_i, stride_s, stride_d, O_1)
        load_tile_sh(dO_ptr, base_offset, i_start, 0, s_mask_i, stride_s, stride_d, dO_0)
        load_tile_sh(dO_ptr, base_offset, i_start, 64, s_mask_i, stride_s, stride_d, dO_1)
        
        D_O_0_tmp = tl.zeros((64, 64), tl.float32)
        D_O_1_tmp = tl.zeros((64, 64), tl.float32)
        for i in range(64):
            if s_mask_i[i]:
                for j in range(64):
                    D_O_0_tmp[i, j] = O_0[i, j] * dO_0[i, j]
                    D_O_1_tmp[i, j] = O_1[i, j] * dO_1[i, j]
        
        store_tile_sh(D_O_0_tmp, D_O_0, 0, 0, s_off_i < 64)
        store_tile_sh(D_O_1_tmp, D_O_1, 0, 0, s_off_i < 64)
        
        L_expanded = tl.load(L_ptr + base_offset_L + i_start + s_off_i, mask=s_mask_i, other=0.0)
        
        S_val = tl.zeros((64, 64), tl.float32)
        S_val = gemm_64x64x64(S_val, Q_0, K_0, trans_B=True)
        S_val = gemm_64x64x64(S_val, Q_1, K_1, trans_B=True)
        S_val = S_val * scale
        
        mask_prev = ((i_start + s_off_i[:, None]) >= (j_start + s_off_j[None, :])) & s_mask_i[:, None] & s_mask_j[None, :]
        A_prev = tl.where(mask_prev, tl.exp(S_val - L_expanded[:, None]), 0.0)
        
        store_tile_sh(A_prev, A_0, 0, 0, s_off_i < 64)
        
        dP_prev = tl.zeros((64, 64), tl.float32)
        dP_prev = gemm_64x64x64(dP_prev, dO_0, V_0, trans_B=True)
        dP_prev = gemm_64x64x64(dP_prev, dO_1, V_1, trans_B=True)
        
        store_tile_sh(dP_prev, V_0, 0, 0, s_off_i < 64)
        
        dA_prev = tl.zeros((64, 64), tl.float32)
        for i in range(64):
            if s_mask_i[i]:
                for j in range(64):
                    dA_prev[i, j] = A_0[i, j] * (V_0[i, j] - D_O_0[i, j])
                    
        store_tile_sh(dA_prev, dA_0, 0, 0, s_off_i < 64)
        
        dA_prev_1 = tl.zeros((64, 64), tl.float32)
        for i in range(64):
            if s_mask_i[i]:
                for j in range(64):
                    dA_prev_1[i, j] = A_1[i, j] * (V_1[i, j] - D_O_1[i, j])
                    
        store_tile_sh(dA_prev_1, dA_1, 0, 0, s_off_i < 64)
        
        acc_dK_0 = gemm_64x64x64(acc_dK_0, dA_0, Q_0, trans_B=True)
        acc_dK_1 = gemm_64x64x64(acc_dK_1, dA_1, Q_1, trans_B=True)
        
        acc_dV_0 = gemm_64x64x64(acc_dV_0, A_0, dO_0, trans_B=True)
        acc_dV_1 = gemm_64x64x64(acc_dV_1, A_1, dO_1, trans_B=True)
    
    if j_tile < num_tiles:
        curr_start_q = min(j_tile * 64, S - 1)
        s_mask_i = (curr_start_q + s_off_i) < S
        
        load_tile_sh(Q_ptr, base_offset, curr_start_q, 0, s_mask_i, stride_s, stride_d, Q_0)
        load_tile_sh(Q_ptr, base_offset, curr_start_q, 64, s_mask_i, stride_s, stride_d, Q_1)
        load_tile_sh(O_ptr, base_offset, curr_start_q, 0, s_mask_i, stride_s, stride_d, O_0)
        load_tile_sh(O_ptr, base_offset, curr_start_q, 64, s_mask_i, stride_s, stride_d, O_1)
        load_tile_sh(dO_ptr, base_offset, curr_start_q, 0, s_mask_i, stride_s, stride_d, dO_0)
        load_tile_sh(dO_ptr, base_offset, curr_start_q, 64, s_mask_i, stride_s, stride_d, dO_1)
        
        D_O_0_tmp = tl.zeros((64, 64), tl.float32)
        D_O_1_tmp = tl.zeros((64, 64), tl.float32)
        for i in range(64):
            if s_mask_i[i]:
                for j in range(64):
                    D_O_0_tmp[i, j] = O_0[i, j] * dO_0[i, j]
                    D_O_1_tmp[i, j] = O_1[i, j] * dO_1[i, j]
                    
        store_tile_sh(D_O_0_tmp, D_O_0, 0, 0, s_off_i < 64)
        store_tile_sh(D_O_1_tmp, D_O_1, 0, 0, s_off_i < 64)
        
        L_expanded = tl.load(L_ptr + base_offset_L + curr_start_q + s_off_i, mask=s_mask_i, other=0.0)
        
        S_val = tl.zeros((64, 64), tl.float32)
        S_val = gemm_64x64x64(S_val, Q_0, K_0, trans_B=True)
        S_val = gemm_64x64x64(S_val, Q_1, K_1, trans_B=True)
        S_val = S_val * scale
        
        mask_curr = ((curr_start_q + s_off_i[:, None]) >= (j_start + s_off_j[None, :])) & s_mask_i[:, None] & s_mask_j[None, :]
        A_curr = tl.where(mask_curr, tl.exp(S_val - L_expanded[:, None]), 0.0)
        
        store_tile_sh(A_curr, A_0, 0, 0, s_off_i < 64)
        
        dP_curr = tl.zeros((64, 64), tl.float32)
        dP_curr = gemm_64x64x64(dP_curr, dO_0, V_0, trans_B=True)
        dP_curr = gemm_64x64x64(dP_curr, dO_1, V_1, trans_B=True)
        
        store_tile_sh(dP_curr, V_0, 0, 0, s_off_i < 64)
        
        dA_curr = tl.zeros((64, 64), tl.float32)
        for i in range(64):
            if s_mask_i[i]:
                for j in range(64):
                    dA_curr[i, j] = A_0[i, j] * (V_0[i, j] - D_O_0[i, j])
                    
        store_tile_sh(dA_curr, dA_0, 0, 0, s_off_i < 64)
        
        dA_curr_1 = tl.zeros((64, 64), tl.float32)
        for i in range(64):
            if s_mask_i[i]:
                for j in range(64):
                    dA_curr_1[i, j] = A_1[i, j] * (V_1[i, j] - D_O_1[i, j])
                    
        store_tile_sh(dA_curr_1, dA_1, 0, 0, s_off_i < 64)
        
        acc_dK_0 = gemm_64x64x64(acc_dK_0, dA_0, Q_0, trans_B=True)
        acc_dK_1 = gemm_64x64x64(acc_dK_1, dA_1, Q_1, trans_B=True)
        
        acc_dV_0 = gemm_64x64x64(acc_dV_0, A_0, dO_0, trans_B=True)
        acc_dV_1 = gemm_64x64x64(acc_dV_1, A_1, dO_1, trans_B=True)
        
    acc_dK_0 = acc_dK_0 * scale
    acc_dK_1 = acc_dK_1 * scale
    
    store_tile(acc_dK_0, dK_ptr, base_offset, j_start, 0, s_mask_j, stride_s, stride_d)
    store_tile(acc_dK_1, dK_ptr, base_offset, j_start, 64, s_mask_j, stride_s, stride_d)
    store_tile(acc_dV_0, dV_ptr, base_offset, j_start, 0, s_mask_j, stride_s, stride_d)
    store_tile(acc_dV_1, dV_ptr, base_offset, j_start, 64, s_mask_j, stride_s, stride_d)


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