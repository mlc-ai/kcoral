import torch
import triton
import triton.language as tl
import math


@triton.jit
def load_subtile_16x16(ptr, bh, row_start, col_start, S):
    row_idx = row_start + tl.arange(0, 16)
    col_idx = col_start + tl.arange(0, 16)
    tile_ptr = ptr + bh * (S * 128) + row_idx[:, None] * 128 + col_idx[None, :]
    mask_row = row_idx[:, None] < S
    return tl.load(tile_ptr, mask=mask_row, other=0.0)


@triton.jit
def store_subtile_16x16(ptr, bh, row_start, col_start, my_tile, S):
    row_idx = row_start + tl.arange(0, 16)
    col_idx = col_start + tl.arange(0, 16)
    tile_ptr = ptr + bh * (S * 128) + row_idx[:, None] * 128 + col_idx[None, :]
    mask_row = row_idx[:, None] < S
    tl.store(tile_ptr, my_tile, mask=mask_row)


@triton.jit
def bwd_dQ_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dQ_ptr,
    S, scale, num_blocks,
):
    bh = tl.program_id(1)
    q_blk = tl.program_id(0)
    q_offset = q_blk * 128
    
    Q0 = [[None for _ in range(4)] for _ in range(8)]
    Q1 = [[None for _ in range(4)] for _ in range(8)]
    O0 = [[None for _ in range(4)] for _ in range(8)]
    O1 = [[None for _ in range(4)] for _ in range(8)]
    dO0 = [[None for _ in range(4)] for _ in range(8)]
    dO1 = [[None for _ in range(4)] for _ in range(8)]
    
    for idx_0 in range(8):
        for idx_1 in range(2):
            for idx_2 in range(4):
                row_s = q_offset + idx_0 * 16
                col_s = (idx_1 * 64) + idx_2 * 16
                
                if idx_1 == 0:
                    Q0[idx_0][idx_2] = load_subtile_16x16(Q_ptr, bh, row_s, col_s, S)
                    O0[idx_0][idx_2] = load_subtile_16x16(O_ptr, bh, row_s, col_s, S)
                    dO0[idx_0][idx_2] = load_subtile_16x16(dO_ptr, bh, row_s, col_s, S)
                else:
                    Q1[idx_0][idx_2] = load_subtile_16x16(Q_ptr, bh, row_s, col_s, S)
                    O1[idx_0][idx_2] = load_subtile_16x16(O_ptr, bh, row_s, col_s, S)
                    dO1[idx_0][idx_2] = load_subtile_16x16(dO_ptr, bh, row_s, col_s, S)
                    
    D_16 = [None] * 8
    for i in range(8):
        d_sum = 0.0
        for k in range(4):
            d_sum += tl.sum(O0[i][k] * dO0[i][k], -1, keep_dims=True)
            d_sum += tl.sum(O1[i][k] * dO1[i][k], -1, keep_dims=True)
        D_16[i] = d_sum
        
    q_rows = q_offset + tl.arange(0, 128)
    L = tl.load(L_ptr + bh * S + q_rows, mask=q_rows < S, other=0.0)
    
    L_16 = [None] * 8
    for i in range(8):
        L_16[i] = L[i * 16:(i + 1) * 16]
        
    dQ0 = [None] * 8
    dQ1 = [None] * 8
    for i in range(8):
        dQ0[i] = [0.0] * 4
        dQ1[i] = [0.0] * 4
        
    K0 = [[None for _ in range(4)] for _ in range(8)]
    K1 = [[None for _ in range(4)] for _ in range(8)]
    V0 = [[None for _ in range(4)] for _ in range(8)]
    V1 = [[None for _ in range(4)] for _ in range(8)]
    
    for k_blk in range(0, q_blk + 1):
        k_offset = k_blk * 128
        
        for idx_0 in range(8):
            for idx_1 in range(2):
                for idx_2 in range(4):
                    row_s = k_offset + idx_0 * 16
                    col_s = (idx_1 * 64) + idx_2 * 16
                    
                    if idx_1 == 0:
                        K0[idx_0][idx_2] = load_subtile_16x16(K_ptr, bh, row_s, col_s, S)
                        V0[idx_0][idx_2] = load_subtile_16x16(V_ptr, bh, row_s, col_s, S)
                    else:
                        K1[idx_0][idx_2] = load_subtile_16x16(K_ptr, bh, row_s, col_s, S)
                        V1[idx_0][idx_2] = load_subtile_16x16(V_ptr, bh, row_s, col_s, S)
                        
        K0_transposed = [[None for _ in range(4)] for _ in range(8)]
        for j in range(8):
            for k in range(4):
                K0_transposed[j][k] = K0[j][k].T
                
        K1_transposed = [[None for _ in range(4)] for _ in range(8)]
        for j in range(8):
            for k in range(4):
                K1_transposed[j][k] = K1[j][k].T
                
        V0_transposed = [[None for _ in range(4)] for _ in range(8)]
        for j in range(8):
            for k in range(4):
                V0_transposed[j][k] = V0[j][k].T
                
        V1_transposed = [[None for _ in range(4)] for _ in range(8)]
        for j in range(8):
            for k in range(4):
                V1_transposed[j][k] = V1[j][k].T
        
        S_tiles = [[None for _ in range(8)] for _ in range(8)]
        for i in range(8):
            for j in range(8):
                s = 0.0
                for k in range(4):
                    s = tl.dot(Q0[i][k], K0_transposed[j][k], acc=s)
                    s = tl.dot(Q1[i][k], K1_transposed[j][k], acc=s)
                S_tiles[i][j] = s
                
        dP_tiles = [[None for _ in range(8)] for _ in range(8)]
        for i in range(8):
            for j in range(8):
                dp = 0.0
                for k in range(4):
                    dp = tl.dot(dO0[i][k], V0_transposed[j][k], acc=dp)
                    dp = tl.dot(dO1[i][k], V1_transposed[j][k], acc=dp)
                dP_tiles[i][j] = dp
                
        dS_tiles = [[None for _ in range(8)] for _ in range(8)]
        
        q_rows_16 = q_offset + tl.arange(0, 128).reshape(8, 16)
        k_rows_16 = k_offset + tl.arange(0, 128).reshape(8, 16)
        
        for i in range(8):
            for j in range(8):
                mask = (q_rows_16[i, :] >= k_rows_16[:, j])
                
                s_val = S_tiles[i][j] * scale - L_16[i, None]
                p_ij = tl.where(mask, tl.exp(s_val), 0.0)
                    
                ds = p_ij * (dP_tiles[i][j] - D_16[i]) * scale
                
                dS_tiles[i][j] = ds
                
        for i in range(8):
            for k in range(4):
                dq = 0.0
                for j in range(8):
                    dq = tl.dot(dS_tiles[i][j], K0[j][k], acc=dq)
                    dq = tl.dot(dS_tiles[i][j], K1[j][k], acc=dq)
                dQ0[i][k] = dq
                dQ1[i][k] = dq # Wait, dQ1 computes exactly the same thing here?? 
                               # Ah, dQ1 should accumulate over j using K1. 
                               # But I assigned the exact same loop variable `dq` to both dQ0 and dQ1. 
                               # I need separate accumulators!
                               
                dq1 = 0.0
                for j in range(8):
                    dq1 = tl.dot(dS_tiles[i][j], K1[j][k], acc=dq1)
                dQ1[i][k] = dq1
                              
    for i in range(8):
        for k in range(4):
            row_s = q_offset + i * 16
            col_s0 = k * 16
            col_s1 = 64 + k * 16
            
            store_subtile_16x16(dQ_ptr, bh, row_s, col_s0, dQ0[i][k].to(tl.bfloat16), S)
            store_subtile_16x16(dQ_ptr, bh, row_s, col_s1, dQ1[i][k].to(tl.bfloat16), S)


@triton.jit
def bwd_dKV_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dK_ptr, dV_ptr,
    S, scale, num_blocks,
):
    bh = tl.program_id(1)
    k_blk = tl.program_id(0)
    k_offset = k_blk * 128
    
    K0 = [[None for _ in range(4)] for _ in range(8)]
    K1 = [[None for _ in range(4)] for _ in range(8)]
    V0 = [[None for _ in range(4)] for _ in range(8)]
    V1 = [[None for _ in range(4)] for _ in range(8)]
    
    for idx_0 in range(8):
        for idx_1 in range(2):
            for idx_2 in range(4):
                row_s = k_offset + idx_0 * 16
                col_s = (idx_1 * 64) + idx_2 * 16
                
                if idx_1 == 0:
                    K0[idx_0][idx_2] = load_subtile_16x16(K_ptr, bh, row_s, col_s, S)
                    V0[idx_0][idx_2] = load_subtile_16x16(V_ptr, bh, row_s, col_s, S)
                else:
                    K1[idx_0][idx_2] = load_subtile_16x16(K_ptr, bh, row_s, col_s, S)
                    V1[idx_0][idx_2] = load_subtile_16x16(V_ptr, bh, row_s, col_s, S)
                    
    dK0 = [[0.0 for _ in range(4)] for _ in range(8)]
    dK1 = [[0.0 for _ in range(4)] for _ in range(8)]
    dV0 = [[0.0 for _ in range(4)] for _ in range(8)]
    dV1 = [[0.0 for _ in range(4)] for _ in range(8)]
    
    for q_blk in range(k_blk, num_blocks):
        q_offset = q_blk * 128
        
        Q0 = [[None for _ in range(4)] for _ in range(8)]
        Q1 = [[None for _ in range(4)] for _ in range(8)]
        O0 = [[None for _ in range(4)] for _ in range(8)]
        O1 = [[None for _ in range(4)] for _ in range(8)]
        dO0 = [[None for _ in range(4)] for _ in range(8)]
        dO1 = [[None for _ in range(4)] for _ in range(8)]
        
        for idx_0 in range(8):
            for idx_1 in range(2):
                for idx_2 in range(4):
                    row_s = q_offset + idx_0 * 16
                    col_s = (idx_1 * 64) + idx_2 * 16
                    
                    if idx_1 == 0:
                        Q0[idx_0][idx_2] = load_subtile_16x16(Q_ptr, bh, row_s, col_s, S)
                        O0[idx_0][idx_2] = load_subtile_16x16(O_ptr, bh, row_s, col_s, S)
                        dO0[idx_0][idx_2] = load_subtile_16x16(dO_ptr, bh, row_s, col_s, S)
                    else:
                        Q1[idx_0][idx_2] = load_subtile_16x16(Q_ptr, bh, row_s, col_s, S)
                        O1[idx_0][idx_2] = load_subtile_16x16(O_ptr, bh, row_s, col_s, S)
                        dO1[idx_0][idx_2] = load_subtile_16x16(dO_ptr, bh, row_s, col_s, S)
                        
        D_16 = [None] * 8
        for i in range(8):
            d_sum = 0.0
            for k in range(4):
                d_sum += tl.sum(O0[i][k] * dO0[i][k], -1, keep_dims=True)
                d_sum += tl.sum(O1[i][k] * dO1[i][k], -1, keep_dims=True)
            D_16[i] = d_sum
            
        q_rows = q_offset + tl.arange(0, 128)
        L = tl.load(L_ptr + bh * S + q_rows, mask=q_rows < S, other=0.0)
        
        L_16 = [None] * 8
        for i in range(8):
            L_16[i] = L[i * 16:(i + 1) * 16]
            
        K0_transposed = [[None for _ in range(4)] for _ in range(8)]
        for j in range(8):
            for k in range(4):
                K0_transposed[j][k] = K0[j][k].T
                
        K1_transposed = [[None for _ in range(4)] for _ in range(8)]
        for j in range(8):
            for k in range(4):
                K1_transposed[j][k] = K1[j][k].T
                
        V0_transposed = [[None for _ in range(4)] for _ in range(8)]
        for j in range(8):
            for k in range(4):
                V0_transposed[j][k] = V0[j][k].T
                
        V1_transposed = [[None for _ in range(4)] for _ in range(8)]
        for j in range(8):
            for k in range(4):
                V1_transposed[j][k] = V1[j][k].T
                
        S_tiles = [[None for _ in range(8)] for _ in range(8)]
        for i in range(8):
            for j in range(8):
                s = 0.0
                for k in range(4):
                    s = tl.dot(Q0[i][k], K0_transposed[j][k], acc=s)
                    s = tl.dot(Q1[i][k], K1_transposed[j][k], acc=s)
                S_tiles[i][j] = s
                
        dP_tiles = [[None for _ in range(8)] for _ in range(8)]
        for i in range(8):
            for j in range(8):
                dp = 0.0
                for k in range(4):
                    dp = tl.dot(dO0[i][k], V0_transposed[j][k], acc=dp)
                    dp = tl.dot(dO1[i][k], V1_transposed[j][k], acc=dp)
                dP_tiles[i][j] = dp
                
        dS_tiles = [[None for _ in range(8)] for _ in range(8)]
        P_tiles = [[None for _ in range(8)] for _ in range(8)]
        
        q_rows_16 = q_offset + tl.arange(0, 128).reshape(8, 16)
        k_rows_16 = k_offset + tl.arange(0, 128).reshape(8, 16)
        
        for i in range(8):
            for j in range(8):
                mask = (q_rows_16[i, :] >= k_rows_16[:, j])
                
                s_val = S_tiles[i][j] * scale - L_16[i, None]
                p_ij = tl.where(mask, tl.exp(s_val), 0.0)
                P_tiles[i][j] = p_ij
                    
                ds = p_ij * (dP_tiles[i][j] - D_16[i]) * scale
                dS_tiles[i][j] = ds
                
        for j in range(8):
            for k in range(4):
                dk0 = 0.0
                dk1 = 0.0
                dv0 = 0.0
                dv1 = 0.0
                for i in range(8):
                    ds_t = dS_tiles[i][j].T
                    p_t = P_tiles[i][j].T
                    
                    dk0 = tl.dot(ds_t, Q0[i][k], acc=dk0)
                    dk1 = tl.dot(ds_t, Q1[i][k], acc=dk1)
                    
                    dv0 = tl.dot(p_t, dO0[i][k], acc=dv0)
                    dv1 = tl.dot(p_t, dO1[i][k], acc=dv1)
                    
                dK0[j][k] = dk0
                dK1[j][k] = dk1
                dV0[j][k] = dv0
                dV1[j][k] = dv1
                
    for j in range(8):
        for k in range(4):
            row_s = k_offset + j * 16
            col_s0 = k * 16
            col_s1 = 64 + k * 16
            
            store_subtile_16x16(dK_ptr, bh, row_s, col_s0, dK0[j][k].to(tl.bfloat16), S)
            store_subtile_16x16(dK_ptr, bh, row_s, col_s1, dK1[j][k].to(tl.bfloat16), S)
            store_subtile_16x16(dV_ptr, bh, row_s, col_s0, dV0[j][k].to(tl.bfloat16), S)
            store_subtile_16x16(dV_ptr, bh, row_s, col_s1, dV1[j][k].to(tl.bfloat16), S)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    
    scale = 1.0 / math.sqrt(d)
    
    num_blocks = triton.cdiv(S, 128)
    grid = (num_blocks, B * H)
    
    bwd_dKV_kernel[grid](
        Q_ptr=Q, K_ptr=K, V_ptr=V, O_ptr=O, dO_ptr=dO, L_ptr=L,
        dK_ptr=dK, dV_ptr=dV, S=S, scale=scale, num_blocks=num_blocks,
        num_stages=1, num_warps=4
    )
    
    bwd_dQ_kernel[grid](
        Q_ptr=Q, K_ptr=K, V_ptr=V, O_ptr=O, dO_ptr=dO, L_ptr=L,
        dQ_ptr=dQ, S=S, scale=scale, num_blocks=num_blocks,
        num_stages=1, num_warps=4
    )