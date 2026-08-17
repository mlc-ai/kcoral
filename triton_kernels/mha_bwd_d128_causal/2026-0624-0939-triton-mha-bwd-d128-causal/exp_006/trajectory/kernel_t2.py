import math

import torch
import triton
import triton.language as tl


@triton.jit
def _preprocess_kernel(O_ptr, dO_ptr, D_ptr, S_len, H, STRIDE_B: tl.constexpr, STRIDE_H: tl.constexpr, STRIDE_S: tl.constexpr):
    b = tl.program_id(2)
    h = tl.program_id(1)
    s_idx = tl.program_id(0)
    
    if s_idx < S_len:
        off = b * STRIDE_B + h * STRIDE_H + s_idx * STRIDE_S
        do_vals = tl.load(dO_ptr + off + tl.arange(0, 128))
        o_vals = tl.load(O_ptr + off + tl.arange(0, 128))
        d_val = tl.sum(do_vals.to(tl.float32) * o_vals.to(tl.float32))
        tl.store(D_ptr + (b * H + h) * S_len + s_idx, d_val)


@triton.jit
def _dkdv_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, D_ptr, dK_ptr, dV_ptr,
    S_len, H, tau,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    j = tl.program_id(0)
    h_idx = tl.program_id(1)
    b_idx = tl.program_id(2)
    num_blocks = tl.cdiv(S_len, BLOCK_M)
    
    row_idx = tl.arange(0, BLOCK_N)
    col_idx = tl.arange(0, BLOCK_N)
    
    base_offset = b_idx * (H * S_len * 128) + h_idx * (S_len * 128)
    
    k_off_left = base_offset + j * 128 * BLOCK_N + 0
    k_off_right = base_offset + j * 128 * BLOCK_N + 64 * 128
    
    K_tile_l = tl.load(K_ptr + k_off_left + row_idx[:, None] * 128 + col_idx[None, :],
                       mask=(j * BLOCK_N + row_idx)[:, None] < S_len, other=0.0)
    K_tile_r = tl.load(K_ptr + k_off_right + row_idx[:, None] * 128 + col_idx[None, :],
                       mask=(j * BLOCK_N + row_idx)[:, None] < S_len, other=0.0)
    
    v_off_left = base_offset + j * 128 * BLOCK_N + 0
    v_off_right = base_offset + j * 128 * BLOCK_N + 64 * 128
    
    V_tile_l = tl.load(V_ptr + v_off_left + row_idx[:, None] * 128 + col_idx[None, :],
                       mask=(j * BLOCK_N + row_idx)[:, None] < S_len, other=0.0)
    V_tile_r = tl.load(V_ptr + v_off_right + row_idx[:, None] * 128 + col_idx[None, :],
                       mask=(j * BLOCK_N + row_idx)[:, None] < S_len, other=0.0)
    
    dK_acc_l = tl.zeros((BLOCK_N, BLOCK_N), tl.float32)
    dK_acc_r = tl.zeros((BLOCK_N, BLOCK_N), tl.float32)
    dV_acc_l = tl.zeros((BLOCK_N, BLOCK_N), tl.float32)
    dV_acc_r = tl.zeros((BLOCK_N, BLOCK_N), tl.float32)
    
    for i in range(j, num_blocks):
        q_off_left = base_offset + i * 128 * BLOCK_M + 0
        q_off_right = base_offset + i * 128 * BLOCK_M + 64 * 128
        
        Q_tile_l = tl.load(Q_ptr + q_off_left + row_idx[:, None] * 128 + col_idx[None, :],
                           mask=(i * BLOCK_M + row_idx)[:, None] < S_len, other=0.0)
        Q_tile_r = tl.load(Q_ptr + q_off_right + row_idx[:, None] * 128 + col_idx[None, :],
                           mask=(i * BLOCK_M + row_idx)[:, None] < S_len, other=0.0)
        
        do_off_left = base_offset + i * 128 * BLOCK_M + 0
        do_off_right = base_offset + i * 128 * BLOCK_M + 64 * 128
        
        dO_tile_l = tl.load(dO_ptr + do_off_left + row_idx[:, None] * 128 + col_idx[None, :],
                            mask=(i * BLOCK_M + row_idx)[:, None] < S_len, other=0.0)
        dO_tile_r = tl.load(dO_ptr + do_off_right + row_idx[:, None] * 128 + col_idx[None, :],
                            mask=(i * BLOCK_M + row_idx)[:, None] < S_len, other=0.0)
        
        abs_i = i * BLOCK_M + row_idx
        L_vec = tl.load(L_ptr + (b_idx * H + h_idx) * S_len + abs_i, mask=(abs_i < S_len), other=1e20)
        D_vec = tl.load(D_ptr + (b_idx * H + h_idx) * S_len + abs_i, mask=(abs_i < S_len), other=0.0)
        
        L_tile = L_vec[:, None]
        D_tile = D_vec[:, None]
        
        S = (Q_tile_l @ K_tile_l.T + Q_tile_r @ K_tile_r.T) * tau
        
        abs_j_base = j * BLOCK_N + col_idx
        mask = (abs_i[:, None] >= abs_j_base[None, :]) & (abs_i[:, None] < S_len) & (abs_j_base[None, :] < S_len)
        
        P = tl.exp(S - L_tile)
        P = P * mask
        
        dP = dO_tile_l @ V_tile_l.T + dO_tile_r @ V_tile_r.T
        
        dS = P * (dP - D_tile) * tau
        dS = dS * mask
        
        dV_acc_l = dV_acc_l + P.T @ dO_tile_l
        dV_acc_r = dV_acc_r + P.T @ dO_tile_r
        
        dK_acc_l = dK_acc_l + dS.T @ Q_tile_l
        dK_acc_r = dK_acc_r + dS.T @ Q_tile_r
        
    out_off_k = b_idx * (H * S_len * 128) + h_idx * (S_len * 128) + j * 128 * BLOCK_N
    
    dk_ptrs_l = out_off_k + row_idx[:, None] * 128 + col_idx[None, :]
    dk_mask = (j * BLOCK_N + row_idx) < S_len
    tl.store(dK_ptr + dk_ptrs_l, dK_acc_l.to(tl.bfloat16), mask=dk_mask[:, None])
    
    dk_ptrs_r = out_off_k + 64 * 128 + row_idx[:, None] * 128 + col_idx[None, :]
    tl.store(dK_ptr + dk_ptrs_r, dK_acc_r.to(tl.bfloat16), mask=dk_mask[:, None])
    
    dv_ptrs_l = out_off_k + row_idx[:, None] * 128 + col_idx[None, :]
    tl.store(dV_ptr + dv_ptrs_l, dV_acc_l.to(tl.bfloat16), mask=dk_mask[:, None])
    
    dv_ptrs_r = out_off_k + 64 * 128 + row_idx[:, None] * 128 + col_idx[None, :]
    tl.store(dV_ptr + dv_ptrs_r, dV_acc_r.to(tl.bfloat16), mask=dk_mask[:, None])


@triton.jit
def _dq_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, D_ptr, dQ_ptr,
    S_len, H, tau,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    i = tl.program_id(0)
    h_idx = tl.program_id(1)
    b_idx = tl.program_id(2)
    num_blocks = tl.cdiv(S_len, BLOCK_N)
    
    row_idx = tl.arange(0, BLOCK_M)
    col_idx = tl.arange(0, BLOCK_N)
    
    base_offset = b_idx * (H * S_len * 128) + h_idx * (S_len * 128)
    
    q_off_left = base_offset + i * 128 * BLOCK_M + 0
    q_off_right = base_offset + i * 128 * BLOCK_M + 64 * 128
    
    Q_tile_l = tl.load(Q_ptr + q_off_left + row_idx[:, None] * 128 + col_idx[None, :],
                       mask=(i * BLOCK_M + row_idx)[:, None] < S_len, other=0.0)
    Q_tile_r = tl.load(Q_ptr + q_off_right + row_idx[:, None] * 128 + col_idx[None, :],
                       mask=(i * BLOCK_M + row_idx)[:, None] < S_len, other=0.0)
    
    do_off_left = base_offset + i * 128 * BLOCK_M + 0
    do_off_right = base_offset + i * 128 * BLOCK_M + 64 * 128
    
    dO_tile_l = tl.load(dO_ptr + do_off_left + row_idx[:, None] * 128 + col_idx[None, :],
                        mask=(i * BLOCK_M + row_idx)[:, None] < S_len, other=0.0)
    dO_tile_r = tl.load(dO_ptr + do_off_right + row_idx[:, None] * 128 + col_idx[None, :],
                        mask=(i * BLOCK_M + row_idx)[:, None] < S_len, other=0.0)
    
    abs_i = i * BLOCK_M + row_idx
    L_vec = tl.load(L_ptr + (b_idx * H + h_idx) * S_len + abs_i, mask=(abs_i < S_len), other=1e20)
    D_vec = tl.load(D_ptr + (b_idx * H + h_idx) * S_len + abs_i, mask=(abs_i < S_len), other=0.0)
    
    L_tile = L_vec[:, None]
    D_tile = D_vec[:, None]
    
    dQ_acc_l = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    dQ_acc_r = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    
    for j_inner in range(0, i + 1):
        k_off_left = base_offset + j_inner * 128 * BLOCK_N + 0
        k_off_right = base_offset + j_inner * 128 * BLOCK_N + 64 * 128
        
        K_tile_l = tl.load(K_ptr + k_off_left + row_idx[:, None] * 128 + col_idx[None, :],
                           mask=(j_inner * BLOCK_N + row_idx)[:, None] < S_len, other=0.0)
        K_tile_r = tl.load(K_ptr + k_off_right + row_idx[:, None] * 128 + col_idx[None, :],
                           mask=(j_inner * BLOCK_N + row_idx)[:, None] < S_len, other=0.0)
        
        v_off_left = base_offset + j_inner * 128 * BLOCK_N + 0
        v_off_right = base_offset + j_inner * 128 * BLOCK_N + 64 * 128
        
        V_tile_l = tl.load(V_ptr + v_off_left + row_idx[:, None] * 128 + col_idx[None, :],
                           mask=(j_inner * BLOCK_N + row_idx)[:, None] < S_len, other=0.0)
        V_tile_r = tl.load(V_ptr + v_off_right + row_idx[:, None] * 128 + col_idx[None, :],
                           mask=(j_inner * BLOCK_N + row_idx)[:, None] < S_len, other=0.0)
        
        S = (Q_tile_l @ K_tile_l.T + Q_tile_r @ K_tile_r.T) * tau
        
        abs_j_base = j_inner * BLOCK_N + col_idx
        mask = (abs_i[:, None] >= abs_j_base[None, :]) & (abs_i[:, None] < S_len) & (abs_j_base[None, :] < S_len)
        
        P = tl.exp(S - L_tile)
        P = P * mask
        
        dP = dO_tile_l @ V_tile_l.T + dO_tile_r @ V_tile_r.T
        
        dS = P * (dP - D_tile) * tau
        dS = dS * mask
        
        dQ_acc_l = dQ_acc_l + dS @ K_tile_l
        dQ_acc_r = dQ_acc_r + dS @ K_tile_r
        
    out_off_q = b_idx * (H * S_len * 128) + h_idx * (S_len * 128) + i * 128 * BLOCK_M
    
    dq_ptrs_l = out_off_q + row_idx[:, None] * 128 + col_idx[None, :]
    dq_mask = (i * BLOCK_M + row_idx) < S_len
    tl.store(dQ_ptr + dq_ptrs_l, dQ_acc_l.to(tl.bfloat16), mask=dq_mask[:, None])
    
    dq_ptrs_r = out_off_q + 64 * 128 + row_idx[:, None] * 128 + col_idx[None, :]
    tl.store(dQ_ptr + dq_ptrs_r, dQ_acc_r.to(tl.bfloat16), mask=dq_mask[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    
    B, H, S_len, d = Q.shape
    tau = 1.0 / math.sqrt(d)
    
    D = torch.empty((B, H, S_len), dtype=torch.float32, device=Q.device)
    
    grid_pre = (S_len, H, B)
    _preprocess_kernel[grid_pre](O, dO, D, S_len, H, H * S_len * 128, S_len * 128, 128)
    
    BLOCK_M = 64
    BLOCK_N = 64
    
    T_r = triton.cdiv(S_len, BLOCK_M)
    grid = (T_r, H, B)
    
    _dkdv_kernel[grid](
        Q, K, V, dO, L, D, dK, dV,
        S_len, H, tau,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=8, num_stages=2
    )
    
    _dq_kernel[grid](
        Q, K, V, dO, L, D, dQ,
        S_len, H, tau,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=8, num_stages=2
    )