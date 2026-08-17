import torch
import triton
import triton.language as tl
import math

B = 4
H = 48
d = 128

@triton.jit
def load_tile_sh(ptr, start, mask, sh_0, sh_1):
    off = ptr + start * 128
    for i in range(64):
        if mask[i]:
            row_ptr = off + i * 128
            for j in range(64):
                sh_0[i, j] = tl.load(row_ptr + j)
                sh_1[i, j] = tl.load(row_ptr + 64 + j)

@triton.jit
def load_tile_sh_transposed(ptr, start, mask, sh_0_T, sh_1_T):
    off = ptr + start * 128
    for i in range(64):
        if mask[i]:
            row_ptr = off + i * 128
            for j in range(64):
                sh_0_T[j, i] = tl.load(row_ptr + j)
                sh_1_T[j, i] = tl.load(row_ptr + 64 + j)

@triton.jit
def store_tile(acc, ptr, start, mask):
    for i in range(64):
        if mask[i]:
            for j in range(64):
                ptr[start + i, j] = acc[i, j]
                ptr[start + i, j + 64] = acc_1[i, j]

@triton.jit
def load_col_sh(arr, col_idx):
    col = arr[:, col_idx]
    return col

@triton.jit
def off(b, h, start, ptr):
    return ptr + b * (H * S * d) + h * (S * d) + start * d

@triton.jit
def off_L(b, h, start):
    return b * (H * S) + h * S + start

def p1_load_tile(ptr, start, mask, sh_0, sh_1):
    load_tile_sh(ptr, start, mask, sh_0, sh_1)

def p2_load_tile(ptr, start, mask, sh_0, sh_1):
    load_tile_sh(ptr, start, mask, sh_0, sh_1)

@triton.jit
def bwd_kernel(
    Q, K, V, O, dO, L,
    dQ, dK, dV,
    S, scale,
):
    NUM_SMS = 132
    sm_id = tl.program_id(0)
    h_id = sm_id % H
    b_id = (sm_id // H) % B
    t_id = sm_id // (H * B)
    
    i_tile = t_id
    j_tile = t_id
    
    s_off_i = tl.arange(0, 64)
    s_off_j = tl.arange(0, 64)
    
    i_start = i_tile * 64
    j_start = j_tile * 64
    
    s_mask_i = (i_start + s_off_i) < S
    s_mask_j = (j_start + s_off_j) < S
    
    L_expanded = tl.load(L + off_L(b_id, h_id, i_start + s_off_i), mask=s_mask_i, other=0.0)
    
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
    Q_0_T: tl.extern_shared_array((64, 64), ty=tl.bfloat16)
    Q_1_T: tl.extern_shared_array((64, 64), ty=tl.bfloat16)
    K_0_T: tl.extern_shared_array((64, 64), ty=tl.bfloat16)
    K_1_T: tl.extern_shared_array((64, 64), ty=tl.bfloat16)
    V_0_T: tl.extern_shared_array((64, 64), ty=tl.bfloat16)
    V_1_T: tl.extern_shared_array((64, 64), ty=tl.bfloat16)
    dO_0_T: tl.extern_shared_array((64, 64), ty=tl.bfloat16)
    dO_1_T: tl.extern_shared_array((64, 64), ty=tl.bfloat16)
    
    dP_sh:   tl.extern_shared_array((64, 64), ty=tl.float32)
    dA_sh_0: tl.extern_shared_array((64, 64), ty=tl.float32)
    dA_sh_1: tl.extern_shared_array((64, 64), ty=tl.float32)
    dA_T_sh_0: tl.extern_shared_array((64, 64), ty=tl.float32)
    dA_T_sh_1: tl.extern_shared_array((64, 64), ty=tl.float32)
    A_T_sh_0:  tl.extern_shared_array((64, 64), ty=tl.float32)
    A_T_sh_1:  tl.extern_shared_array((64, 64), ty=tl.float32)
    D_O_sh:    tl.extern_shared_array((64, 64), ty=tl.float32)

    # ---------------------- Phase 1 ----------------------
    acc_dQ_0 = tl.zeros((64, 64), tl.float32)
    acc_dQ_1 = tl.zeros((64, 64), tl.float32)
    
    ptr_Q = off(b_id, h_id, i_start, Q)
    ptr_O = off(b_id, h_id, i_start, O)
    ptr_dO = off(b_id, h_id, i_start, dO)
    
    p1_load_tile(ptr_Q, i_start, s_mask_i, Q_0, Q_1)
    p1_load_tile(ptr_O, i_start, s_mask_i, O_0, O_1)
    p1_load_tile(ptr_dO, i_start, s_mask_i, dO_0, dO_1)
    load_tile_sh_transposed(ptr_Q, i_start, s_mask_i, Q_0_T, Q_1_T)
    load_tile_sh_transposed(ptr_dO, i_start, s_mask_i, dO_0_T, dO_1_T)
    
    D_O = tl.zeros((64, 64), tl.float32)
    for k_idx in range(64):
        o0 = load_col_sh(Q_0, k_idx)
        do0_T = load_col_sh(dO_0_T, k_idx)
        D_O = D_O + tl.dot(o0, do0_T.T)
        
        o1 = load_col_sh(Q_1, k_idx)
        do1_T = load_col_sh(dO_1_T, k_idx)
        D_O = D_O + tl.dot(o1, do1_T.T)
    
    store_tile(D_O.T, D_O_sh, 0, s_off_i < 64)
    
    num_tiles = (S + 64 - 1) // 64
    step = max(1, num_tiles // 132)
    
    for j_tile in range(0, i_tile + 1):
        j_start = j_tile * 64
        s_mask_j = (j_start + s_off_j) < S
        
        ptr_K = off(b_id, h_id, j_start, K)
        ptr_V = off(b_id, h_id, j_start, V)
        
        p1_load_tile(ptr_K, j_start, s_mask_j, K_0, K_1)
        p1_load_tile(ptr_V, j_start, s_mask_j, V_0, V_1)
        load_tile_sh_transposed(ptr_K, j_start, s_mask_j, K_0_T, K_1_T)
        load_tile_sh_transposed(ptr_V, j_start, s_mask_j, V_0_T, V_1_T)
        
        S_val = tl.zeros((64, 64), tl.float32)
        for k_idx in range(64):
            q0_T = load_col_sh(Q_0_T, k_idx)
            k0 = load_col_sh(K_0, k_idx)
            S_val = S_val + tl.dot(q0_T, k0.T)
            
            q1_T = load_col_sh(Q_1_T, k_idx)
            k1 = load_col_sh(K_1, k_idx)
            S_val = S_val + tl.dot(q1_T, k1.T)
            
        S_val = S_val * scale
        
        mask_curr = ((i_start + s_off_i[:, None]) >= (j_start + s_off_j[None, :])) & s_mask_i[:, None] & s_mask_j[None, :]
        A_curr = tl.where(mask_curr, tl.exp(S_val - L_expanded[:, None]), 0.0)
        
        store_tile(A_curr.T, A_T_sh_0, 0, s_off_i < 64)
        
        dP = tl.zeros((64, 64), tl.float32)
        for k_idx in range(64):
            do0_T = load_col_sh(dO_0_T, k_idx)
            v0 = load_col_sh(V_0, k_idx)
            dP = dP + tl.dot(do0_T, v0.T)
            
            do1_T = load_col_sh(dO_1_T, k_idx)
            v1 = load_col_sh(V_1, k_idx)
            dP = dP + tl.dot(do1_T, v1.T)
            
        store_tile(dP, dP_sh, 0, s_off_i < 64)
        
        load_tile_sh(dP_sh, dA_sh_0, 0, s_off_i < 64)
        load_tile_sh_transposed(D_O_sh, dA_sh_1, 0, s_off_i < 64)
        
        dA_curr = dA_sh_0 * A_curr - dA_sh_1
        
        store_tile(dA_curr, dA_sh_0, 0, s_off_i < 64)
        store_tile(dA_curr.T, dA_T_sh_0, 0, s_off_i < 64)
        
        for k_idx in range(64):
            da0 = load_col_sh(dA_sh_0, k_idx)
            k0 = load_col_sh(K_0, k_idx)
            acc_dQ_0 = acc_dQ_0 + tl.dot(da0, k0.T)
            
            da1 = load_col_sh(dA_sh_1, k_idx)
            k1 = load_col_sh(K_1, k_idx)
            acc_dQ_1 = acc_dQ_1 + tl.dot(da1, k1.T)
            
        acc_dQ_0 = acc_dQ_0 * scale
        acc_dQ_1 = acc_dQ_1 * scale
        
    ptr_dQ = off(b_id, h_id, i_start, dQ)
    store_tile(acc_dQ_0, ptr_dQ, i_start, s_mask_i)


    # ---------------------- Phase 2 ----------------------
    acc_dK_0 = tl.zeros((64, 64), tl.float32)
    acc_dK_1 = tl.zeros((64, 64), tl.float32)
    acc_dV_0 = tl.zeros((64, 64), tl.float32)
    acc_dV_1 = tl.zeros((64, 64), tl.float32)
    
    ptr_K = off(b_id, h_id, j_start, K)
    ptr_V = off(b_id, h_id, j_start, V)
    
    p2_load_tile(ptr_K, j_start, s_mask_j, K_0, K_1)
    p2_load_tile(ptr_V, j_start, s_mask_j, V_0, V_1)
    load_tile_sh_transposed(ptr_K, j_start, s_mask_j, K_0_T, K_1_T)
    load_tile_sh_transposed(ptr_V, j_start, s_mask_j, V_0_T, V_1_T)
    
    for i_tile in range(j_tile, num_tiles, step):
        i_start = i_tile * 64
        s_mask_i = (i_start + s_off_i) < S
        
        ptr_Q = off(b_id, h_id, i_start, Q)
        ptr_O = off(b_id, h_id, i_start, O)
        ptr_dO = off(b_id, h_id, i_start, dO)
        
        p2_load_tile(ptr_Q, i_start, s_mask_i, Q_0, Q_1)
        p2_load_tile(ptr_O, i_start, s_mask_i, O_0, O_1)
        p2_load_tile(ptr_dO, i_start, s_mask_i, dO_0, dO_1)
        load_tile_sh_transposed(ptr_Q, i_start, s_mask_i, Q_0_T, Q_1_T)
        load_tile_sh_transposed(ptr_dO, i_start, s_mask_i, dO_0_T, dO_1_T)
        
        D_O = tl.zeros((64, 64), tl.float32)
        for k_idx in range(64):
            o0 = load_col_sh(Q_0, k_idx)
            do0_T = load_col_sh(dO_0_T, k_idx)
            D_O = D_O + tl.dot(o0, do0_T.T)
            
            o1 = load_col_sh(Q_1, k_idx)
            do1_T = load_col_sh(dO_1_T, k_idx)
            D_O = D_O + tl.dot(o1, do1_T.T)
            
        store_tile(D_O.T, D_O_sh, 0, s_off_i < 64)
        
        L_expanded = tl.load(L + off_L(b_id, h_id, i_start + s_off_i), mask=s_mask_i, other=0.0)
        
        S_val = tl.zeros((64, 64), tl.float32)
        for k_idx in range(64):
            q0_T = load_col_sh(Q_0_T, k_idx)
            k0 = load_col_sh(K_0, k_idx)
            S_val = S_val + tl.dot(q0_T, k0.T)
            
            q1_T = load_col_sh(Q_1_T, k_idx)
            k1 = load_col_sh(K_1, k_idx)
            S_val = S_val + tl.dot(q1_T, k1.T)
            
        S_val = S_val * scale
        
        mask_curr = ((i_start + s_off_i[:, None]) >= (j_start + s_off_j[None, :])) & s_mask_i[:, None] & s_mask_j[None, :]
        A_curr = tl.where(mask_curr, tl.exp(S_val - L_expanded[:, None]), 0.0)
        
        store_tile(A_curr.T, A_T_sh_0, 0, s_off_i < 64)
        
        dP = tl.zeros((64, 64), tl.float32)
        for k_idx in range(64):
            do0_T = load_col_sh(dO_0_T, k_idx)
            v0 = load_col_sh(V_0, k_idx)
            dP = dP + tl.dot(do0_T, v0.T)
            
            do1_T = load_col_sh(dO_1_T, k_idx)
            v1 = load_col_sh(V_1, k_idx)
            dP = dP + tl.dot(do1_T, v1.T)
            
        store_tile(dP, dP_sh, 0, s_off_i < 64)
        
        load_tile_sh(dP_sh, dA_sh_0, 0, s_off_i < 64)
        load_tile_sh_transposed(D_O_sh, dA_sh_1, 0, s_off_i < 64)
        
        dA_curr = dA_sh_0 * A_curr - dA_sh_1
        
        store_tile(dA_curr.T, dA_T_sh_0, 0, s_off_i < 64)
        
        for k_idx in range(64):
            da_T0 = load_col_sh(dA_T_sh_0, k_idx)
            q0 = load_col_sh(Q_0, k_idx)
            acc_dK_0 = acc_dK_0 + tl.dot(da_T0, q0.T)
            
            da_T1 = load_col_sh(dA_T_sh_1, k_idx)
            q1 = load_col_sh(Q_1, k_idx)
            acc_dK_1 = acc_dK_1 + tl.dot(da_T1, q1.T)
            
            a_T0 = load_col_sh(A_T_sh_0, k_idx)
            do0 = load_col_sh(dO_0, k_idx)
            acc_dV_0 = acc_dV_0 + tl.dot(a_T0, do0.T)
            
            a_T1 = load_col_sh(A_T_sh_1, k_idx)
            do1 = load_col_sh(dO_1, k_idx)
            acc_dV_1 = acc_dV_1 + tl.dot(a_T1, do1.T)
            
        acc_dK_0 = acc_dK_0 * scale
        acc_dK_1 = acc_dK_1 * scale
        
    ptr_dK = off(b_id, h_id, j_start, dK)
    ptr_dV = off(b_id, h_id, j_start, dV)
    store_tile(acc_dK_0, ptr_dK, j_start, s_mask_j)
    store_tile(acc_dV_0, ptr_dV, j_start, s_mask_j)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    S = Q.shape[2]
    
    dQ = dQ.to(torch.bfloat16)
    dK = dK.to(torch.bfloat16)
    dV = dV.to(torch.bfloat16)
    
    scale = 1.0 / math.sqrt(d)
    NUM_SMS = 132
    num_tiles = (S + 64 - 1) // 64
    total_tiles = num_tiles * B * H
    
    grid = lambda META: (B * H * min(NUM_SMS, total_tiles),)
    
    bwd_kernel[grid](
        Q, K, V, O, dO, L,
        dQ, dK, dV,
        S, scale,
        num_warps=4,
        num_stages=2,
    )