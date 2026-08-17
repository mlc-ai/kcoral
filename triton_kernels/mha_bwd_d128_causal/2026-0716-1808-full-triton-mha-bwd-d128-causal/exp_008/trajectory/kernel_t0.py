import torch
import triton
import triton.language as tl


@triton.jit
def load_2d_tile_sliced(base, bh, row_start, col_start, S, max_row, max_col):
    row_idx = row_start + tl.arange(0, max_row)
    col_idx = col_start + tl.arange(0, max_col)
    
    tile_base = base + bh * (S * 128) + row_idx[:, None] * 128 + col_idx[None, :] * 1
    
    mask_row = row_idx[:, None] < S
    
    my_tile = tl.load(tile_base, mask=mask_row, other=0.0)
    return my_tile


@triton.jit
def dl_dot(A, B, acc=None, out_dtype=tl.float32):
    if acc is None:
        return tl.dot(A, B, out_dtype=out_dtype)
    return tl.dot(A, B, acc=acc, out_dtype=out_dtype)


@triton.jit
def bwd_dQ_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dQ_ptr,
    S, scale, num_blocks,
):
    bh = tl.program_id(1)
    q_blk = tl.program_id(0)
    
    q_rows = q_blk * 128 + tl.arange(0, 128)
    row_mask = q_rows < S
    
    Q_block_0 = load_2d_tile_sliced(Q_ptr, bh, q_rows, 0, S, 128, 64)
    Q_block_1 = load_2d_tile_sliced(Q_ptr, bh, q_rows, 64, S, 128, 64)
    
    O_block_0 = load_2d_tile_sliced(O_ptr, bh, q_rows, 0, S, 128, 64)
    O_block_1 = load_2d_tile_sliced(O_ptr, bh, q_rows, 64, S, 128, 64)
    
    dO_block_0 = load_2d_tile_sliced(dO_ptr, bh, q_rows, 0, S, 128, 64)
    dO_block_1 = load_2d_tile_sliced(dO_ptr, bh, q_rows, 64, S, 128, 64)
    
    D_half_0 = 0.0
    D_half_1 = 0.0
    for c in range(0, 64):
        D_half_0 += O_block_0[0, c] * dO_block_0[0, c]
        D_half_1 += O_block_1[0, c] * dO_block_1[0, c]
    D_val = D_half_0 + D_half_1
    
    dQ_0 = 0.0
    dQ_1 = 0.0
    
    L_val = 0.0
    
    for k_blk in range(0, q_blk + 1):
        k_rows = k_blk * 128 + tl.arange(0, 128)
        
        K_block_0 = load_2d_tile_sliced(K_ptr, bh, k_rows, 0, S, 128, 64)
        K_block_1 = load_2d_tile_sliced(K_ptr, bh, k_rows, 64, S, 128, 64)
        
        V_block_0 = load_2d_tile_sliced(V_ptr, bh, k_rows, 0, S, 128, 64)
        V_block_1 = load_2d_tile_sliced(V_ptr, bh, k_rows, 64, S, 128, 64)
        
        S = dl_dot(Q_block_0, K_block_0.T, acc=S, out_dtype=tl.float32)
        S = dl_dot(Q_block_1, K_block_1.T, acc=S, out_dtype=tl.float32)
        
        d_P = dl_dot(dO_block_0, V_block_0.T, acc=d_P, out_dtype=tl.float32)
        d_P = dl_dot(dO_block_1, V_block_1.T, acc=d_P, out_dtype=tl.float32)
        
        S_valid = S * scale - L_val
        
        P_ij = 0.0
        if q_rows >= k_rows:
            P_ij = exp(S_valid)
            
        dS_unscaled = d_P - D_val
        dS = dS_unscaled * P_ij * scale
        
        dQ_0 += dS * K_block_0
        dQ_1 += dS * K_block_1
        
    if row_mask:
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
    row_mask = k_rows < S
    
    K_block_0 = load_2d_tile_sliced(K_ptr, bh, k_rows, 0, S, 128, 64)
    K_block_1 = load_2d_tile_sliced(K_ptr, bh, k_rows, 64, S, 128, 64)
    
    V_block_0 = load_2d_tile_sliced(V_ptr, bh, k_rows, 0, S, 128, 64)
    V_block_1 = load_2d_tile_sliced(V_ptr, bh, k_rows, 64, S, 128, 64)
    
    d_K_0 = 0.0
    d_K_1 = 0.0
    d_V_0 = 0.0
    d_V_1 = 0.0
    
    for q_blk in range(k_blk, num_blocks):
        q_rows = q_blk * 128 + tl.arange(0, 128)
        
        Q_block_0 = load_2d_tile_sliced(Q_ptr, bh, q_rows, 0, S, 128, 64)
        Q_block_1 = load_2d_tile_sliced(Q_ptr, bh, q_rows, 64, S, 128, 64)
        
        O_block_0 = load_2d_tile_sliced(O_ptr, bh, q_rows, 0, S, 128, 64)
        O_block_1 = load_2d_tile_sliced(O_ptr, bh, q_rows, 64, S, 128, 64)
        
        dO_block_0 = load_2d_tile_sliced(dO_ptr, bh, q_rows, 0, S, 128, 64)
        dO_block_1 = load_2d_tile_sliced(dO_ptr, bh, q_rows, 64, S, 128, 64)
        
        D_half_0 = 0.0
        D_half_1 = 0.0
        for c in range(0, 64):
            D_half_0 += O_block_0[0, c] * dO_block_0[0, c]
            D_half_1 += O_block_1[0, c] * dO_block_1[0, c]
        D_val = D_half_0 + D_half_1
        
        L_val = 0.0
        
        S = dl_dot(Q_block_0, K_block_0.T, acc=S, out_dtype=tl.float32)
        S = dl_dot(Q_block_1, K_block_1.T, acc=S, out_dtype=tl.float32)
        
        d_P = dl_dot(dO_block_0, V_block_0.T, acc=d_P, out_dtype=tl.float32)
        d_P = dl_dot(dO_block_1, V_block_1.T, acc=d_P, out_dtype=tl.float32)
        
        S_valid = S * scale - L_val
        
        P_ij = 0.0
        if q_rows >= k_rows:
            P_ij = exp(S_valid)
            
        P_T = P_ij
        dS_unscaled = d_P - D_val
        dS = dS_unscaled * P_ij * scale
        dS_T = dS
        
        d_V_0 += P_T * dO_block_0
        d_V_1 += P_T * dO_block_1
        
        d_K_0 += dS_T * Q_block_0
        d_K_1 += dS_T * Q_block_1
        
    if row_mask:
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
    
    D = torch.empty((B * H, S), dtype=torch.float32, device=device)
    
    grid_D = ((S + 127) // 128, B * H)
    O_and_D_kernel[grid_D](O_ptr=O, dO_ptr=dO, D_ptr=D, S=S, stride_b=128, stride_s=128*128)
    
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