import torch
import triton
import triton.language as tl
import math

B = 4
H = 48
d = 128
head_dim = 64
num_k_tiles = d // head_dim

@triton.jit
def off_Q(b, h, row):
    global H, S, d
    return b * (H * S * d) + h * (S * d) + row * d

@triton.jit
def off_K(b, h, row):
    global H, S, d
    return b * (H * S * d) + h * (S * d) + row * d

@triton.jit
def off_V(b, h, row):
    global H, S, d
    return b * (H * S * d) + h * (S * d) + row * d

@triton.jit
def off_O(b, h, row):
    global H, S, d
    return b * (H * S * d) + h * (S * d) + row * d

@triton.jit
def off_dO(b, h, row):
    global H, S, d
    return b * (H * S * d) + h * (S * d) + row * d

@triton.jit
def off_dK(b, h, row):
    global H, S, d
    return b * (H * S * d) + h * (S * d) + row * d

@triton.jit
def off_dV(b, h, row):
    global H, S, d
    return b * (H * S * d) + h * (S * d) + row * d

@triton.jit
def off_dP_T(b, h, row, col):
    global H, S
    return b * (H * S * S) + h * (S * S) + row * S + col

@triton.jit
def off_L(b, h, row):
    global H, S
    return b * (H * S) + h * S + row


# ---------------------- Phase 1 ----------------------
@triton.jit
def phase_1(b, h, i_tile, dQ_ptr, dP_T_ptr, Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr):
    global H, S, d, head_dim, num_k_tiles
    
    Q_0: tl.extern_shared_array((head_dim, head_dim), ty=tl.bfloat16)
    Q_1: tl.extern_shared_array((head_dim, head_dim), ty=tl.bfloat16)
    O_0: tl.extern_shared_array((head_dim, head_dim), ty=tl.bfloat16)
    O_1: tl.extern_shared_array((head_dim, head_dim), ty=tl.bfloat16)
    dO_0: tl.extern_shared_array((head_dim, head_dim), ty=tl.bfloat16)
    dO_1: tl.extern_shared_array((head_dim, head_dim), ty=tl.bfloat16)
    K_0_prev: tl.extern_shared_array((head_dim, head_dim), ty=tl.bfloat16)
    K_1_prev: tl.extern_shared_array((head_dim, head_dim), ty=tl.bfloat16)
    K_0_curr: tl.extern_shared_array((head_dim, head_dim), ty=tl.bfloat16)
    K_1_curr: tl.extern_shared_array((head_dim, head_dim), ty=tl.bfloat16)
    V_0:     tl.extern_shared_array((head_dim, head_dim), ty=tl.bfloat16)
    V_1:     tl.extern_shared_array((head_dim, head_dim), ty=tl.bfloat16)
    dP_store: tl.extern_shared_array((head_dim, head_dim), ty=tl.float32)

    s_idx_prev = head_dim * tl.arange(0, head_dim)
    s_idx_curr = head_dim * tl.arange(0, head_dim)
    
    curr_start = min(i_tile * head_dim, S - 1)
    L_expanded = tl.load(L_ptr + off_L(b, h, curr_start + tl.arange(0, head_dim)))
    
    Q_0_ptr = Q_ptr + off_Q(b, h, curr_start + s_idx_prev)
    Q_1_ptr = Q_ptr + off_Q(b, h, curr_start + s_idx_prev) + head_dim
    O_0_ptr = O_ptr + off_O(b, h, curr_start + s_idx_prev)
    O_1_ptr = O_ptr + off_O(b, h, curr_start + s_idx_prev) + head_dim
    dO_0_ptr = dO_ptr + off_dO(b, h, curr_start + s_idx_prev)
    dO_1_ptr = dO_ptr + off_dO(b, h, curr_start + s_idx_prev) + head_dim

    Q_0 = tl.load(Q_0_ptr)
    Q_1 = tl.load(Q_1_ptr)
    O_0 = tl.load(O_0_ptr)
    O_1 = tl.load(O_1_ptr)
    dO_0 = tl.load(dO_0_ptr)
    dO_1 = tl.load(dO_1_ptr)

    # Compute dP for current Q tile against all K tiles
    dP_store = dP_store * 0.0
    for k_tile in range(num_k_tiles):
        V_0 = tl.load(V_ptr + off_V(b, h, curr_start + s_idx_prev))
        V_1 = tl.load(V_ptr + off_V(b, h, curr_start + s_idx_prev + head_dim))
        dP_store = dP_store + dO_0 @ V_0.T + dO_1 @ V_1.T
    
    i_tile_start = i_tile * head_dim
    j_tile_start = i_tile * head_dim
    tl.store(dP_T_ptr + off_dP_T(b, h, j_tile_start + s_idx_prev, i_tile_start + s_idx_curr), dP_store)
    
    acc_dQ_0 = 0.0
    acc_dQ_1 = 0.0
    
    # Causal masking guarantees that we only care about j <= i
    num_tiles = (S + head_dim - 1) // head_dim
    for j_tile in range(min(i_tile + 1, num_tiles)):
        prev_start = min(j_tile * head_dim, S - 1)
        
        K_0_prev_ptr = K_ptr + off_K(b, h, prev_start + s_idx_prev)
        K_1_prev_ptr = K_ptr + off_K(b, h, prev_start + s_idx_prev) + head_dim
        K_0_prev = tl.load(K_0_prev_ptr)
        K_1_prev = tl.load(K_1_prev_ptr)
        
        S_val_prev = Q_0 @ K_0_prev.T + Q_1 @ K_1_prev.T
        
        mask_prev = (curr_start + s_idx_prev[:, None]) >= (prev_start + s_idx_prev[None, :])
        S_val_prev = S_val_prev / scale
        A_prev = tl.where(mask_prev, tl.exp(S_val_prev - L_expanded[None, :]), 0.0)
        
        dP_prev = tl.load(dP_T_ptr + off_dP_T(b, h, prev_start + s_idx_prev, i_tile_start + s_idx_curr))
        
        D_A_prev_0 = A_prev[:, 0:head_dim] * dP_prev[:, 0:head_dim]
        D_A_prev_1 = A_prev[:, head_dim:head_dim*2] * dP_prev[:, head_dim:head_dim*2]
        
        acc_dQ_0 = acc_dQ_0 + D_A_prev_0 @ K_0_prev
        acc_dQ_1 = acc_dQ_1 + D_A_prev_1 @ K_1_prev
        
        if j_tile == i_tile:
            mask_curr = (curr_start + s_idx_prev[:, None]) >= (curr_start + s_idx_prev[None, :])
            K_0_curr = K_0_prev * mask_curr
            K_1_curr = K_1_prev * mask_curr
            
            S_val_curr = Q_0 @ K_0_curr.T + Q_1 @ K_1_curr.T
            S_val_curr = S_val_curr / scale
            A_curr = tl.where(mask_curr, tl.exp(S_val_curr - L_expanded[None, :]), 0.0)
            
            dP_curr = dP_prev
            D_A_curr_0 = A_curr[:, 0:head_dim] * dP_curr[:, 0:head_dim]
            D_A_curr_1 = A_curr[:, head_dim:head_dim*2] * dP_curr[:, head_dim:head_dim*2]
            
            acc_dQ_0 = acc_dQ_0 + D_A_curr_0 @ K_0_curr
            acc_dQ_1 = acc_dQ_1 + D_A_curr_1 @ K_1_curr
    
    acc_dQ_0 = acc_dQ_0 / scale
    acc_dQ_1 = acc_dQ_1 / scale
    
    dQ_0_ptr = dQ_ptr + off_dQ(b, h, curr_start + s_idx_prev)
    dQ_1_ptr = dQ_ptr + off_dQ(b, h, curr_start + s_idx_prev) + head_dim
    tl.store(dQ_0_ptr, acc_dQ_0)
    tl.store(dQ_1_ptr, acc_dQ_1)


# ---------------------- Phase 2 ----------------------
@triton.jit
def phase_2(b, h, j_tile, dK_ptr, dV_ptr, K_ptr, V_ptr, Q_ptr, dO_ptr, dP_T_ptr, L_ptr):
    global H, S, d, head_dim, num_k_tiles
    
    K_0_curr: tl.extern_shared_array((head_dim, head_dim), ty=tl.bfloat16)
    K_1_curr: tl.extern_shared_array((head_dim, head_dim), ty=tl.bfloat16)
    V_0_curr: tl.extern_shared_array((head_dim, head_dim), ty=tl.bfloat16)
    V_1_curr: tl.extern_shared_array((head_dim, head_dim), ty=tl.bfloat16)
    Q_0_prev: tl.extern_shared_array((head_dim, head_dim), ty=tl.bfloat16)
    Q_1_prev: tl.extern_shared_array((head_dim, head_dim), ty=tl.bfloat16)
    Q_0_curr: tl.extern_shared_array((head_dim, head_dim), ty=tl.bfloat16)
    Q_1_curr: tl.extern_shared_array((head_dim, head_dim), ty=tl.bfloat16)
    dO_0_prev: tl.extern_shared_array((head_dim, head_dim), ty=tl.bfloat16)
    dO_1_prev: tl.extern_shared_array((head_dim, head_dim), ty=tl.bfloat16)
    dO_0_curr: tl.extern_shared_array((head_dim, head_dim), ty=tl.bfloat16)
    dO_1_curr: tl.extern_shared_array((head_dim, head_dim), ty=tl.bfloat16)
    D_A_prev_T_0: tl.extern_shared_array((head_dim, head_dim), ty=tl.float32)
    D_A_prev_T_1: tl.extern_shared_array((head_dim, head_dim), ty=tl.float32)
    D_A_curr_T_0: tl.extern_shared_array((head_dim, head_dim), ty=tl.float32)
    D_A_curr_T_1: tl.extern_shared_array((head_dim, head_dim), ty=tl.float32)
    A_prev_T_0:   tl.extern_shared_array((head_dim, head_dim), ty=tl.float32)
    A_prev_T_1:   tl.extern_shared_array((head_dim, head_dim), ty=tl.float32)
    A_curr_T_0:   tl.extern_shared_array((head_dim, head_dim), ty=tl.float32)
    A_curr_T_1:   tl.extern_shared_array((head_dim, head_dim), ty=tl.float32)
    S_val_prev:   tl.extern_shared_array((head_dim, head_dim), ty=tl.float32)
    S_val_curr:   tl.extern_shared_array((head_dim, head_dim), ty=tl.float32)
    dP_prev_T:    tl.extern_shared_array((head_dim, head_dim), ty=tl.float32)
    dP_curr_T:    tl.extern_shared_array((head_dim, head_dim), ty=tl.float32)
    mask_prev:    tl.extern_shared_array((head_dim, head_dim), ty=tl.int1)
    mask_curr:    tl.extern_shared_array((head_dim, head_dim), ty=tl.int1)
    k_idx_prev:   tl.extern_shared_array((head_dim,), ty=tl.int32)
    k_idx_curr:   tl.extern_shared_array((head_dim,), ty=tl.int32)
    s_idx_prev:   tl.extern_shared_array((head_dim,), ty=tl.int32)
    s_idx_curr:   tl.extern_shared_array((head_dim,), ty=tl.int32)
    
    s_idx_prev = head_dim * tl.arange(0, head_dim)
    s_idx_curr = head_dim * tl.arange(0, head_dim)
    
    curr_start = min(j_tile * head_dim, S - 1)
    
    K_0_curr_ptr = K_ptr + off_K(b, h, curr_start + s_idx_prev)
    K_1_curr_ptr = K_ptr + off_K(b, h, curr_start + s_idx_prev) + head_dim
    V_0_curr_ptr = V_ptr + off_V(b, h, curr_start + s_idx_prev)
    V_1_curr_ptr = V_ptr + off_V(b, h, curr_start + s_idx_prev) + head_dim
    
    K_0_curr = tl.load(K_0_curr_ptr)
    K_1_curr = tl.load(K_1_curr_ptr)
    V_0_curr = tl.load(V_0_curr_ptr)
    V_1_curr = tl.load(V_1_curr_ptr)
    
    acc_dK_0 = 0.0
    acc_dK_1 = 0.0
    acc_dV_0 = 0.0
    acc_dV_1 = 0.0
    
    num_tiles = (S + head_dim - 1) // head_dim
    for i_tile in range(num_tiles - 1, j_tile, -1):
        prev_start = min(i_tile * head_dim, S - 1)
        
        Q_0_prev_ptr = Q_ptr + off_Q(b, h, prev_start + s_idx_prev)
        Q_1_prev_ptr = Q_ptr + off_Q(b, h, prev_start + s_idx_prev) + head_dim
        dO_0_prev_ptr = dO_ptr + off_dO(b, h, prev_start + s_idx_prev)
        dO_1_prev_ptr = dO_ptr + off_dO(b, h, prev_start + s_idx_prev) + head_dim
        
        Q_0_prev = tl.load(Q_0_prev_ptr)
        Q_1_prev = tl.load(Q_1_prev_ptr)
        dO_0_prev = tl.load(dO_0_prev_ptr)
        dO_1_prev = tl.load(dO_1_prev_ptr)
        
        L_prev = tl.load(L_ptr + off_L(b, h, prev_start + tl.arange(0, head_dim)))
        
        S_val_prev = Q_0_prev @ K_0_curr.T + Q_1_prev @ K_1_curr.T
        
        mask_prev = (prev_start + s_idx_prev[:, None]) >= (curr_start + s_idx_prev[None, :])
        S_val_prev = S_val_prev / scale
        A_prev = tl.where(mask_prev, tl.exp(S_val_prev - L_prev[None, :]), 0.0)
        
        dP_prev_T = tl.load(dP_T_ptr + off_dP_T(b, h, curr_start + s_idx_prev, prev_start + s_idx_curr))
        
        D_A_prev_0 = A_prev[:, 0:head_dim] * dP_prev_T[:, 0:head_dim]
        D_A_prev_1 = A_prev[:, head_dim:head_dim*2] * dP_prev_T[:, head_dim:head_dim*2]
        
        D_A_prev_T_0 = D_A_prev_0.T
        D_A_prev_T_1 = D_A_prev_1.T
        
        acc_dK_0 = acc_dK_0 + D_A_prev_T_0 @ Q_0_prev
        acc_dK_1 = acc_dK_1 + D_A_prev_T_1 @ Q_1_prev
        
        A_prev_T_0 = A_prev[:, 0:head_dim].T
        A_prev_T_1 = A_prev[:, head_dim:head_dim*2].T
        acc_dV_0 = acc_dV_0 + A_prev_T_0 @ dO_0_prev
        acc_dV_1 = acc_dV_1 + A_prev_T_1 @ dO_1_prev
    
    if j_tile < num_tiles:
        curr_start_q = min(j_tile * head_dim, S - 1)
        Q_0_curr_ptr = Q_ptr + off_Q(b, h, curr_start_q + s_idx_prev)
        Q_1_curr_ptr = Q_ptr + off_Q(b, h, curr_start_q + s_idx_prev) + head_dim
        dO_0_curr_ptr = dO_ptr + off_dO(b, h, curr_start_q + s_idx_prev)
        dO_1_curr_ptr = dO_ptr + off_dO(b, h, curr_start_q + s_idx_prev) + head_dim
        
        Q_0_curr = tl.load(Q_0_curr_ptr)
        Q_1_curr = tl.load(Q_1_curr_ptr)
        dO_0_curr = tl.load(dO_0_curr_ptr)
        dO_1_curr = tl.load(dO_1_curr_ptr)
        
        L_curr = tl.load(L_ptr + off_L(b, h, curr_start_q + tl.arange(0, head_dim)))
        
        S_val_curr = Q_0_curr @ K_0_curr.T + Q_1_curr @ K_1_curr.T
        
        mask_curr = (curr_start_q + s_idx_prev[:, None]) >= (curr_start + s_idx_prev[None, :])
        S_val_curr = S_val_curr / scale
        A_curr = tl.where(mask_curr, tl.exp(S_val_curr - L_curr[None, :]), 0.0)
        
        dP_curr_T = tl.load(dP_T_ptr + off_dP_T(b, h, curr_start + s_idx_prev, curr_start_q + s_idx_curr))
        
        D_A_curr_0 = A_curr[:, 0:head_dim] * dP_curr_T[:, 0:head_dim]
        D_A_curr_1 = A_curr[:, head_dim:head_dim*2] * dP_curr_T[:, head_dim:head_dim*2]
        
        D_A_curr_T_0 = D_A_curr_0.T
        D_A_curr_T_1 = D_A_curr_1.T
        
        Q_0_curr_valid = tl.where(mask_curr, Q_0_curr, 0.0)
        Q_1_curr_valid = tl.where(mask_curr, Q_1_curr, 0.0)
        
        acc_dK_0 = acc_dK_0 + D_A_curr_T_0 @ Q_0_curr_valid
        acc_dK_1 = acc_dK_1 + D_A_curr_T_1 @ Q_1_curr_valid
        
        A_curr_T_0 = A_curr[:, 0:head_dim].T
        A_curr_T_1 = A_curr[:, head_dim:head_dim*2].T
        dO_0_curr_valid = tl.where(mask_curr, dO_0_curr, 0.0)
        dO_1_curr_valid = tl.where(mask_curr, dO_1_curr, 0.0)
        
        acc_dV_0 = acc_dV_0 + A_curr_T_0 @ dO_0_curr_valid
        acc_dV_1 = acc_dV_1 + A_curr_T_1 @ dO_1_curr_valid
    
    acc_dK_0 = acc_dK_0 / scale
    acc_dK_1 = acc_dK_1 / scale
    
    dK_0_ptr = dK_ptr + off_dK(b, h, curr_start + s_idx_prev)
    dK_1_ptr = dK_ptr + off_dK(b, h, curr_start + s_idx_prev) + head_dim
    tl.store(dK_0_ptr, acc_dK_0)
    tl.store(dK_1_ptr, acc_dK_1)
    
    dV_0_ptr = dV_ptr + off_dV(b, h, curr_start + s_idx_prev)
    dV_1_ptr = dV_ptr + off_dV(b, h, curr_start + s_idx_prev) + head_dim
    tl.store(dV_0_ptr, acc_dV_0)
    tl.store(dV_1_ptr, acc_dV_1)


def run(Q, K, V, O, dO, L):
    global S
    S = Q.shape[2]
    
    dQ = torch.empty([B, H, S, d] + [S, S], device=Q.device, dtype=torch.bfloat16)
    dK = torch.empty_like(K)
    dV = torch.empty_like(V)
    
    dQ_ptr = dQ.data_ptr()
    dP_T_ptr = dQ_ptr + (B * H * S * d) 
    
    scale = 1.0 / math.sqrt(128)
    
    grid_1 = ((S + head_dim - 1) // head_dim, B, H)
    phase_1[grid_1](
        dQ_ptr, dP_T_ptr, Q, K, V, O, dO, L,
        scale=scale
    )
    
    grid_2 = ((S + head_dim - 1) // head_dim, B, H)
    phase_2[grid_2](
        dK, dV, K, V, Q, dO, dP_T_ptr, L,
        scale=scale
    )
    
    return dQ[:, :, :, :, 0, 0], dK, dV