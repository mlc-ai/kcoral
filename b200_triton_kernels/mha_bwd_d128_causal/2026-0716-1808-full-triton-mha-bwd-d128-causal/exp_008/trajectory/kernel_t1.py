import torch
import triton
import triton.language as tl
import math

triton.config.default_max_bytes = 128 * 128 * 4


@triton.jit
def load_2d_tile(base, bh, row_start, col_start, S, max_row, max_col):
    row_idx = row_start + tl.arange(0, max_row)
    col_idx = col_start + tl.arange(0, max_col)
    
    tile_base = base + bh * (S * 128) + row_idx[:, None] * 128 + col_idx[None, :] * 1
    
    mask_row = row_idx[:, None] < S
    
    my_tile = tl.load(tile_base, mask=mask_row, other=0.0)
    return my_tile


@triton.jit
def bwd_dQ_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dQ_ptr,
    S, scale, num_blocks,
):
    bh = tl.program_id(1)
    q_blk = tl.program_id(0)
    
    q_rows = q_blk * 128 + tl.arange(0, 128)
    
    Q_block_0 = load_2d_tile(Q_ptr, bh, q_rows, 0, S, 128, 64)
    Q_block_1 = load_2d_tile(Q_ptr, bh, q_rows, 64, S, 128, 64)
    
    O_block_0 = load_2d_tile(O_ptr, bh, q_rows, 0, S, 128, 64)
    O_block_1 = load_2d_tile(O_ptr, bh, q_rows, 64, S, 128, 64)
    
    dO_block_0 = load_2d_tile(dO_ptr, bh, q_rows, 0, S, 128, 64)
    dO_block_1 = load_2d_tile(dO_ptr, bh, q_rows, 64, S, 128, 64)
    
    D_val = 0.0
    for r in range(128):
        for c in range(64):
            D_val += O_block_0[r, c] * dO_block_0[r, c]
            D_val += O_block_1[r, c] * dO_block_1[r, c]
            
    L_ptr_flat = L_ptr + bh * S + q_rows
    L_val = tl.load(L_ptr_flat, mask=q_rows < S, other=0.0)
    
    dQ_0 = 0.0
    dQ_1 = 0.0
    
    for k_blk in range(0, q_blk + 1):
        k_rows = k_blk * 128 + tl.arange(0, 128)
        
        K_block_0 = load_2d_tile(K_ptr, bh, k_rows, 0, S, 128, 64)
        K_block_1 = load_2d_tile(K_ptr, bh, k_rows, 64, S, 128, 64)
        
        V_block_0 = load_2d_tile(V_ptr, bh, k_rows, 0, S, 128, 64)
        V_block_1 = load_2d_tile(V_ptr, bh, k_rows, 64, S, 128, 64)
        
        S_mat = Q_block_0 @ K_block_0.T + Q_block_1 @ K_block_1.T
        
        d_P = dO_block_0 @ V_block_0.T + dO_block_1 @ V_block_1.T
        
        S_valid = S_mat * scale - L_val
        
        P_ij = 0.0
        if q_rows >= k_rows:
            P_ij = tl.exp(S_valid)
            
        dS_unscaled = d_P - D_val
        dS = P_ij * dS_unscaled * scale
        
        dQ_0 += dS @ K_block_0
        dQ_1 += dS @ K_block_1
        
    if q_rows < S:
        ptr_0 = dQ_ptr + bh * S * 128 + q_rows * 128 + tl.arange(0, 64)
        ptr_1 = dQ_ptr + bh * S * 128 + q_rows * 128 + 64 + tl.arange(0, 64)
        
        store_0 = dQ_0.to(tl.bfloat16)
        store_1 = dQ_1.to(tl.bfloat16)
        
        tl.store(ptr_0, store_0)
        tl.store(ptr_1, store_1)


@triton.jit
def bwd_dKV_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dK_ptr, dV_ptr,
    S, scale, num_blocks,
):
    bh = tl.program_id(1)
    k_blk = tl.program_id(0)
    
    k_rows = k_blk * 128 + tl.arange(0, 128)
    
    K_block_0 = load_2d_tile(K_ptr, bh, k_rows, 0, S, 128, 64)
    K_block_1 = load_2d_tile(K_ptr, bh, k_rows, 64, S, 128, 64)
    
    V_block_0 = load_2d_tile(V_ptr, bh, k_rows, 0, S, 128, 64)
    V_block_1 = load_2d_tile(V_ptr, bh, k_rows, 64, S, 128, 64)
    
    d_K_0 = 0.0
    d_K_1 = 0.0
    d_V_0 = 0.0
    d_V_1 = 0.0
    
    for q_blk in range(k_blk, num_blocks):
        q_rows = q_blk * 128 + tl.arange(0, 128)
        
        Q_block_0 = load_2d_tile(Q_ptr, bh, q_rows, 0, S, 128, 64)
        Q_block_1 = load_2d_tile(Q_ptr, bh, q_rows, 64, S, 128, 64)
        
        O_block_0 = load_2d_tile(O_ptr, bh, q_rows, 0, S, 128, 64)
        O_block_1 = load_2d_tile(O_ptr, bh, q_rows, 64, S, 128, 64)
        
        dO_block_0 = load_2d_tile(dO_ptr, bh, q_rows, 0, S, 128, 64)
        dO_block_1 = load_2d_tile(dO_ptr, bh, q_rows, 64, S, 128, 64)
        
        D_val = 0.0
        for r in range(128):
            for c in range(64):
                D_val += O_block_0[r, c] * dO_block_0[r, c]
                D_val += O_block_1[r, c] * dO_block_1[r, c]
                
        L_ptr_flat = L_ptr + bh * S + q_rows
        L_val = tl.load(L_ptr_flat, mask=q_rows < S, other=0.0)
        
        S_mat = Q_block_0 @ K_block_0.T + Q_block_1 @ K_block_1.T
        
        d_P = dO_block_0 @ V_block_0.T + dO_block_1 @ V_block_1.T
        
        S_valid = S_mat * scale - L_val
        
        P_ij = 0.0
        if q_rows >= k_rows:
            P_ij = tl.exp(S_valid)
            
        P_T = P_ij
            
        dS_unscaled = d_P - D_val
        dS = P_ij * dS_unscaled * scale
        dS_T = dS
        
        d_V_0 += P_T * dO_block_0
        d_V_1 += P_T * dO_block_1
        
        d_K_0 += dS_T * Q_block_0
        d_K_1 += dS_T * Q_block_1
        
    if k_rows < S:
        ptr_K0 = dK_ptr + bh * S * 128 + k_rows * 128 + tl.arange(0, 64)
        ptr_K1 = dK_ptr + bh * S * 128 + k_rows * 128 + 64 + tl.arange(0, 64)
        ptr_V0 = dV_ptr + bh * S * 128 + k_rows * 128 + tl.arange(0, 64)
        ptr_V1 = dV_ptr + bh * S * 128 + k_rows * 128 + 64 + tl.arange(0, 64)
        
        store_K0 = d_K_0.to(tl.bfloat16)
        store_K1 = d_K_1.to(tl.bfloat16)
        store_V0 = d_V_0.to(tl.bfloat16)
        store_V1 = d_V_1.to(tl.bfloat16)
        
        tl.store(ptr_K0, store_K0)
        tl.store(ptr_K1, store_K1)
        tl.store(ptr_V0, store_V0)
        tl.store(ptr_V1, store_V1)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    device = Q.device
    
    scale = 1.0 / math.sqrt(d)
    
    num_blocks = (S + 127) // 128
    grid = (num_blocks, B * H)
    
    bwd_dKV_kernel[grid](
        Q_ptr=Q, K_ptr=K, V_ptr=V, O_ptr=O, dO_ptr=dO, L_ptr=L,
        dK_ptr=dK, dV_ptr=dV, S=S, scale=scale, num_blocks=num_blocks
    )
    
    bwd_dQ_kernel[grid](
        Q_ptr=Q, K_ptr=K, V_ptr=V, O_ptr=O, dO_ptr=dO, L_ptr=L,
        dQ_ptr=dQ, S=S, scale=scale, num_blocks=num_blocks
    )