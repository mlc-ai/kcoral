import torch
import triton
import triton.language as tl
import math


@triton.jit
def load_2d_tile(ptr, bh, row_start, col_start, S):
    row_idx = row_start + tl.arange(0, 128)
    col_idx = col_start + tl.arange(0, 64)
    tile_ptr = ptr + bh * (S * 128) + row_idx[:, None] * 128 + col_idx[None, :]
    mask_row = row_idx[:, None] < S
    return tl.load(tile_ptr, mask=mask_row, other=0.0, padding_option="zero")


@triton.jit
def bwd_dQ_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dQ_ptr,
    S, scale, num_blocks,
):
    bh = tl.program_id(1)
    q_blk = tl.program_id(0)
    q_offset = q_blk * 128
    
    Q0 = load_2d_tile(Q_ptr, bh, q_offset, 0, S)
    Q1 = load_2d_tile(Q_ptr, bh, q_offset, 64, S)
    O0 = load_2d_tile(O_ptr, bh, q_offset, 0, S)
    O1 = load_2d_tile(O_ptr, bh, q_offset, 64, S)
    dO0 = load_2d_tile(dO_ptr, bh, q_offset, 0, S)
    dO1 = load_2d_tile(dO_ptr, bh, q_offset, 64, S)
    
    D = tl.sum(O0 * dO0, -1, keep_dims=True) + tl.sum(O1 * dO1, -1, keep_dims=True)
    
    q_rows = q_offset + tl.arange(0, 128)
    L = tl.load(L_ptr + bh * S + q_rows, mask=q_rows < S, other=0.0)
    
    dQ0 = 0.0
    dQ1 = 0.0
    
    for k_blk in range(0, q_blk + 1):
        k_offset = k_blk * 128
        
        K0 = load_2d_tile(K_ptr, bh, k_offset, 0, S)
        K1 = load_2d_tile(K_ptr, bh, k_offset, 64, S)
        
        V0 = load_2d_tile(V_ptr, bh, k_offset, 0, S)
        V1 = load_2d_tile(V_ptr, bh, k_offset, 64, S)
        
        S_mat = tl.dot(Q0, tl.trans(K0, (1, 0)))
        S_mat = tl.dot(Q1, tl.trans(K1, (1, 0)), acc=S_mat)
        
        d_P = tl.dot(dO0, tl.trans(V0, (1, 0)))
        d_P = tl.dot(dO1, tl.trans(V1, (1, 0)), acc=d_P)
        
        k_cols = k_offset + tl.arange(0, 128)
        mask = (q_rows[:, None] >= k_cols[None, :])
        
        S_valid = S_mat * scale - L[:, None]
        P_ij = tl.where(mask, tl.exp(S_valid), 0.0)
            
        dS = P_ij * (d_P - D) * scale
        
        dQ0 = tl.dot(dS, K0, acc=dQ0)
        dQ1 = tl.dot(dS, K1, acc=dQ1)
        
    ptr_Q0 = dQ_ptr + bh * (S * 128) + q_rows[:, None] * 128 + tl.arange(0, 64)[None, :]
    ptr_Q1 = dQ_ptr + bh * (S * 128) + q_rows[:, None] * 128 + 64 + tl.arange(0, 64)[None, :]
    
    tl.store(ptr_Q0, dQ0.to(tl.bfloat16), mask=q_rows[:, None] < S)
    tl.store(ptr_Q1, dQ1.to(tl.bfloat16), mask=q_rows[:, None] < S)


@triton.jit
def bwd_dKV_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dK_ptr, dV_ptr,
    S, scale, num_blocks,
):
    bh = tl.program_id(1)
    k_blk = tl.program_id(0)
    k_offset = k_blk * 128
    
    k_rows = k_offset + tl.arange(0, 128)
    
    K0 = load_2d_tile(K_ptr, bh, k_offset, 0, S)
    K1 = load_2d_tile(K_ptr, bh, k_offset, 64, S)
    
    V0 = load_2d_tile(V_ptr, bh, k_offset, 0, S)
    V1 = load_2d_tile(V_ptr, bh, k_offset, 64, S)
    
    d_K0 = 0.0
    d_K1 = 0.0
    d_V0 = 0.0
    d_V1 = 0.0
    
    for q_blk in range(k_blk, num_blocks):
        q_offset = q_blk * 128
        
        Q0 = load_2d_tile(Q_ptr, bh, q_offset, 0, S)
        Q1 = load_2d_tile(Q_ptr, bh, q_offset, 64, S)
        
        O0 = load_2d_tile(O_ptr, bh, q_offset, 0, S)
        O1 = load_2d_tile(O_ptr, bh, q_offset, 64, S)
        
        dO0 = load_2d_tile(dO_ptr, bh, q_offset, 0, S)
        dO1 = load_2d_tile(dO_ptr, bh, q_offset, 64, S)
        
        D = tl.sum(O0 * dO0, -1, keep_dims=True) + tl.sum(O1 * dO1, -1, keep_dims=True)
        
        q_rows = q_offset + tl.arange(0, 128)
        L = tl.load(L_ptr + bh * S + q_rows, mask=q_rows < S, other=0.0)
        
        S_mat = tl.dot(Q0, tl.trans(K0, (1, 0)))
        S_mat = tl.dot(Q1, tl.trans(K1, (1, 0)), acc=S_mat)
        
        d_P = tl.dot(dO0, tl.trans(V0, (1, 0)))
        d_P = tl.dot(dO1, tl.trans(V1, (1, 0)), acc=d_P)
        
        k_cols = k_offset + tl.arange(0, 128)
        mask = (q_rows[:, None] >= k_cols[None, :])
        
        S_valid = S_mat * scale - L[:, None]
        P_ij = tl.where(mask, tl.exp(S_valid), 0.0)
            
        dS = P_ij * (d_P - D) * scale
        
        d_K0 = tl.dot(tl.trans(dS, (1, 0)), Q0, acc=d_K0)
        d_K1 = tl.dot(tl.trans(dS, (1, 0)), Q1, acc=d_K1)
        
        d_V0 = tl.dot(tl.trans(P_ij, (1, 0)), dO0, acc=d_V0)
        d_V1 = tl.dot(tl.trans(P_ij, (1, 0)), dO1, acc=d_V1)
        
    ptr_K0 = dK_ptr + bh * (S * 128) + k_rows[:, None] * 128 + tl.arange(0, 64)[None, :]
    ptr_K1 = dK_ptr + bh * (S * 128) + k_rows[:, None] * 128 + 64 + tl.arange(0, 64)[None, :]
    ptr_V0 = dV_ptr + bh * (S * 128) + k_rows[:, None] * 128 + tl.arange(0, 64)[None, :]
    ptr_V1 = dV_ptr + bh * (S * 128) + k_rows[:, None] * 128 + 64 + tl.arange(0, 64)[None, :]
    
    tl.store(ptr_K0, d_K0.to(tl.bfloat16), mask=k_rows[:, None] < S)
    tl.store(ptr_K1, d_K1.to(tl.bfloat16), mask=k_rows[:, None] < S)
    tl.store(ptr_V0, d_V0.to(tl.bfloat16), mask=k_rows[:, None] < S)
    tl.store(ptr_V1, d_V1.to(tl.bfloat16), mask=k_rows[:, None] < S)


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
        num_stages=2, num_warps=4
    )