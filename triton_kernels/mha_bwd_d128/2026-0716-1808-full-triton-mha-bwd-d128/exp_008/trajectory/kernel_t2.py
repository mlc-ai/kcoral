import math
import torch
import triton
import triton.language as tl


@triton.jit
def load_tile_2d(base_ptr, row_offset, col_offset, valid_rows, valid_cols, stride_row, stride_col):
    row_idx = row_offset + tl.arange(0, 64)[:, None]
    col_idx = col_offset + tl.arange(0, 64)[None, :]
    
    ptr = base_ptr + row_idx * stride_row + col_idx * stride_col
    mask = (row_idx < valid_rows[:, None]) & (col_idx < valid_cols[None, :])
    tile = tl.load(ptr, mask=mask, other=0.0)
    return tile.to(tl.float32)


@triton.jit
def dKdV_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
    dK_ptr, dV_ptr,
    S,
    SCALE: tl.constexpr,
):
    pid_i = tl.program_id(0)
    bh = tl.program_id(1)
    
    i_start = pid_i * 64
    
    stride_s = 128
    stride_b_h = S * 128
    
    K_0 = load_tile_2d(K_ptr, bh * S + i_start, 0, bh * S + S, 128, stride_s, 1)
    K_1 = load_tile_2d(K_ptr, bh * S + i_start, 64, bh * S + S, 128, stride_s, 1)
    V_0 = load_tile_2d(V_ptr, bh * S + i_start, 0, bh * S + S, 128, stride_s, 1)
    V_1 = load_tile_2d(V_ptr, bh * S + i_start, 64, bh * S + S, 128, stride_s, 1)
    
    dK_0 = tl.zeros((64, 64), tl.float32)
    dK_1 = tl.zeros((64, 64), tl.float32)
    dV_0 = tl.zeros((64, 64), tl.float32)
    dV_1 = tl.zeros((64, 64), tl.float32)
    
    for j_start in range(0, S, 64):
        Q_0 = load_tile_2d(Q_ptr, bh * S + j_start, 0, bh * S + S, 128, stride_s, 1)
        Q_1 = load_tile_2d(Q_ptr, bh * S + j_start, 64, bh * S + S, 128, stride_s, 1)
        O_0 = load_tile_2d(O_ptr, bh * S + j_start, 0, bh * S + S, 128, stride_s, 1)
        O_1 = load_tile_2d(O_ptr, bh * S + j_start, 64, bh * S + S, 128, stride_s, 1)
        dO_0 = load_tile_2d(dO_ptr, bh * S + j_start, 0, bh * S + S, 128, stride_s, 1)
        dO_1 = load_tile_2d(dO_ptr, bh * S + j_start, 64, bh * S + S, 128, stride_s, 1)
        
        D = tl.sum(O_0 * dO_0 + O_1 * dO_1, axis=1)
        row_idx = j_start + tl.arange(0, 64)
        D = tl.where(row_idx[:, None] < S, D[:, None], 0.0)
        
        S_acc = tl.zeros((64, 64), tl.float32)
        S_acc = tl.dot(Q_0, K_0.T, S_acc)
        S_acc = tl.dot(Q_1, K_1.T, S_acc)
        
        dP_acc = tl.zeros((64, 64), tl.float32)
        dP_acc = tl.dot(dO_0, V_0.T, dP_acc)
        dP_acc = tl.dot(dO_1, V_1.T, dP_acc)
        
        l_ptr_j = L_ptr + bh * S + j_start + tl.arange(0, 64)
        L_j = tl.load(l_ptr_j, mask=(j_start + tl.arange(0, 64)) < S, other=float('inf'))
        
        P = tl.exp(S_acc * SCALE - L_j[:, None])
        
        dS = P * (dP_acc - D) * SCALE
        
        P_T = P.T
        dS_T = dS.T
        
        dV_0 = tl.dot(P_T, dO_0, dV_0)
        dV_1 = tl.dot(P_T, dO_1, dV_1)
        
        dK_0 = tl.dot(dS_T, Q_0, dK_0)
        dK_1 = tl.dot(dS_T, Q_1, dK_1)
        
    row_idx = i_start + tl.arange(0, 64)
    col_idx_0 = tl.arange(0, 64)
    ptr_k0 = dK_ptr + bh * stride_b_h + row_idx[:, None] * stride_s + col_idx_0[None, :]
    tl.store(ptr_k0, dK_0.to(tl.bfloat16), mask=(row_idx[:, None] < S) & (col_idx_0[None, :] < 128))
    
    col_idx_1 = 64 + tl.arange(0, 64)
    ptr_k1 = dK_ptr + bh * stride_b_h + row_idx[:, None] * stride_s + col_idx_1[None, :]
    tl.store(ptr_k1, dK_1.to(tl.bfloat16), mask=(row_idx[:, None] < S) & (col_idx_1[None, :] < 128))
    
    ptr_v0 = dV_ptr + bh * stride_b_h + row_idx[:, None] * stride_s + col_idx_0[None, :]
    tl.store(ptr_v0, dV_0.to(tl.bfloat16), mask=(row_idx[:, None] < S) & (col_idx_0[None, :] < 128))
    
    ptr_v1 = dV_ptr + bh * stride_b_h + row_idx[:, None] * stride_s + col_idx_1[None, :]
    tl.store(ptr_v1, dV_1.to(tl.bfloat16), mask=(row_idx[:, None] < S) & (col_idx_1[None, :] < 128))


@triton.jit
def dQ_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
    dQ_ptr,
    S,
    SCALE: tl.constexpr,
):
    pid_i = tl.program_id(0)
    bh = tl.program_id(1)
    
    i_start = pid_i * 64
    
    stride_s = 128
    stride_b_h = S * 128
    
    Q_0 = load_tile_2d(Q_ptr, bh * S + i_start, 0, bh * S + S, 128, stride_s, 1)
    Q_1 = load_tile_2d(Q_ptr, bh * S + i_start, 64, bh * S + S, 128, stride_s, 1)
    
    O_0 = load_tile_2d(O_ptr, bh * S + i_start, 0, bh * S + S, 128, stride_s, 1)
    O_1 = load_tile_2d(O_ptr, bh * S + i_start, 64, bh * S + S, 128, stride_s, 1)
    
    dO_0 = load_tile_2d(dO_ptr, bh * S + i_start, 0, bh * S + S, 128, stride_s, 1)
    dO_1 = load_tile_2d(dO_ptr, bh * S + i_start, 64, bh * S + S, 128, stride_s, 1)
    
    D = tl.sum(O_0 * dO_0 + O_1 * dO_1, axis=1)
    row_idx = i_start + tl.arange(0, 64)
    D = tl.where(row_idx < S, D, 0.0)
    
    l_ptr_i = L_ptr + bh * S + i_start + tl.arange(0, 64)
    L_i = tl.load(l_ptr_i, mask=(i_start + tl.arange(0, 64)) < S, other=float('inf'))
    
    dQ_0 = tl.zeros((64, 64), tl.float32)
    dQ_1 = tl.zeros((64, 64), tl.float32)
    
    for j_start in range(0, S, 64):
        K_0 = load_tile_2d(K_ptr, bh * S + j_start, 0, bh * S + S, 128, stride_s, 1)
        K_1 = load_tile_2d(K_ptr, bh * S + j_start, 64, bh * S + S, 128, stride_s, 1)
        
        V_0 = load_tile_2d(V_ptr, bh * S + j_start, 0, bh * S + S, 128, stride_s, 1)
        V_1 = load_tile_2d(V_ptr, bh * S + j_start, 64, bh * S + S, 128, stride_s, 1)
        
        S_acc = tl.zeros((64, 64), tl.float32)
        S_acc = tl.dot(Q_0, K_0.T, S_acc)
        S_acc = tl.dot(Q_1, K_1.T, S_acc)
        
        dP_acc = tl.zeros((64, 64), tl.float32)
        dP_acc = tl.dot(dO_0, V_0.T, dP_acc)
        dP_acc = tl.dot(dO_1, V_1.T, dP_acc)
        
        P = tl.exp(S_acc * SCALE - L_i[:, None])
        
        dS = P * (dP_acc - D[:, None]) * SCALE
        
        dQ_0 = tl.dot(dS, K_0, dQ_0)
        dQ_1 = tl.dot(dS, K_1, dQ_1)
        
    row_idx = i_start + tl.arange(0, 64)
    col_idx_0 = tl.arange(0, 64)
    ptr_q0 = dQ_ptr + bh * stride_b_h + row_idx[:, None] * stride_s + col_idx_0[None, :]
    tl.store(ptr_q0, dQ_0.to(tl.bfloat16), mask=(row_idx[:, None] < S) & (col_idx_0[None, :] < 128))
    
    col_idx_1 = 64 + tl.arange(0, 64)
    ptr_q1 = dQ_ptr + bh * stride_b_h + row_idx[:, None] * stride_s + col_idx_1[None, :]
    tl.store(ptr_q1, dQ_1.to(tl.bfloat16), mask=(row_idx[:, None] < S) & (col_idx_1[None, :] < 128))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute backward pass for Multi-Head Attention."""
    B, H, S, d = Q.shape
    
    scale = 1.0 / math.sqrt(d)
    
    dQ = torch.zeros(dQ.shape, device=dQ.device, dtype=torch.bfloat16)
    dK = torch.zeros(dK.shape, device=dK.device, dtype=torch.bfloat16)
    dV = torch.zeros(dV.shape, device=dV.device, dtype=torch.bfloat16)
    
    num_blocks = triton.cdiv(S, 64)
    grid = (num_blocks, B * H)
    
    dKdV_kernel[grid](
        Q.data_ptr(), K.data_ptr(), V.data_ptr(), O.data_ptr(), dO.data_ptr(), L.data_ptr(),
        dK.data_ptr(), dV.data_ptr(),
        S,
        SCALE=scale,
        num_warps=4, num_stages=3
    )
    
    dQ_kernel[grid](
        Q.data_ptr(), K.data_ptr(), V.data_ptr(), O.data_ptr(), dO.data_ptr(), L.data_ptr(),
        dQ.data_ptr(),
        S,
        SCALE=scale,
        num_warps=4, num_stages=3
    )