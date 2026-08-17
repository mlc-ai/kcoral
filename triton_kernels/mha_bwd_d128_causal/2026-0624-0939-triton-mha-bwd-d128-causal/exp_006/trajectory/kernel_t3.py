import math

import torch
import triton
import triton.language as tl


@triton.jit
def _dkdv_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, O_ptr, L_ptr, dK_ptr, dV_ptr,
    S_len, H, tau,
    BLOCK_M: tl.constexpr,
):
    j = tl.program_id(0)
    h_idx = tl.program_id(1)
    b_idx = tl.program_id(2)
    num_blocks = tl.cdiv(S_len, BLOCK_M)
    
    row_idx = tl.arange(0, BLOCK_M)
    col_idx = tl.arange(0, BLOCK_M)
    col_idx_64 = tl.arange(0, 64)
    
    stride_b = H * S_len * 128
    stride_h = S_len * 128
    
    k_off_left = b_idx * stride_b + h_idx * stride_h + j * 128 * BLOCK_M + 0
    k_off_right = b_idx * stride_b + h_idx * stride_h + j * 128 * BLOCK_M + 64
    
    K_l = tl.load(K_ptr + k_off_left + row_idx[:, None] * 128 + col_idx[None, :],
                  mask=(j * BLOCK_M + row_idx)[:, None] < S_len, other=0.0)
    K_r = tl.load(K_ptr + k_off_right + row_idx[:, None] * 128 + col_idx[None, :],
                  mask=(j * BLOCK_M + row_idx)[:, None] < S_len, other=0.0)
    
    v_off_left = b_idx * stride_b + h_idx * stride_h + j * 128 * BLOCK_M + 0
    v_off_right = b_idx * stride_b + h_idx * stride_h + j * 128 * BLOCK_M + 64
    
    V_l = tl.load(V_ptr + v_off_left + row_idx[:, None] * 128 + col_idx[None, :],
                  mask=(j * BLOCK_M + row_idx)[:, None] < S_len, other=0.0)
    V_r = tl.load(V_ptr + v_off_right + row_idx[:, None] * 128 + col_idx[None, :],
                  mask=(j * BLOCK_M + row_idx)[:, None] < S_len, other=0.0)
    
    dK_acc_l = tl.zeros((BLOCK_M, 64), tl.float32)
    dK_acc_r = tl.zeros((BLOCK_M, 64), tl.float32)
    dV_acc_l = tl.zeros((BLOCK_M, 64), tl.float32)
    dV_acc_r = tl.zeros((BLOCK_M, 64), tl.float32)
    
    abs_j = j * BLOCK_M + col_idx
    
    for i in range(j, num_blocks):
        q_off_left = b_idx * stride_b + h_idx * stride_h + i * 128 * BLOCK_M + 0
        q_off_right = b_idx * stride_b + h_idx * stride_h + i * 128 * BLOCK_M + 64
        
        Q_l = tl.load(Q_ptr + q_off_left + row_idx[:, None] * 128 + col_idx[None, :],
                      mask=(i * BLOCK_M + row_idx)[:, None] < S_len, other=0.0)
        Q_r = tl.load(Q_ptr + q_off_right + row_idx[:, None] * 128 + col_idx[None, :],
                      mask=(i * BLOCK_M + row_idx)[:, None] < S_len, other=0.0)
        
        do_off_left = b_idx * stride_b + h_idx * stride_h + i * 128 * BLOCK_M + 0
        do_off_right = b_idx * stride_b + h_idx * stride_h + i * 128 * BLOCK_M + 64
        
        dO_l = tl.load(dO_ptr + do_off_left + row_idx[:, None] * 128 + col_idx[None, :],
                       mask=(i * BLOCK_M + row_idx)[:, None] < S_len, other=0.0)
        dO_r = tl.load(dO_ptr + do_off_right + row_idx[:, None] * 128 + col_idx[None, :],
                       mask=(i * BLOCK_M + row_idx)[:, None] < S_len, other=0.0)
        
        o_off_left = b_idx * stride_b + h_idx * stride_h + i * 128 * BLOCK_M + 0
        o_off_right = b_idx * stride_b + h_idx * stride_h + i * 128 * BLOCK_M + 64
        
        O_l = tl.load(O_ptr + o_off_left + row_idx[:, None] * 128 + col_idx[None, :],
                      mask=(i * BLOCK_M + row_idx)[:, None] < S_len, other=0.0)
        O_r = tl.load(O_ptr + o_off_right + row_idx[:, None] * 128 + col_idx[None, :],
                      mask=(i * BLOCK_M + row_idx)[:, None] < S_len, other=0.0)
        
        abs_i = i * BLOCK_M + row_idx
        L_stride_b = H * S_len
        L_stride_h = S_len
        L_vec = tl.load(L_ptr + b_idx * L_stride_b + h_idx * L_stride_h + abs_i, mask=(abs_i < S_len), other=1e20)
        
        D_val = tl.sum(dO_l * O_l + dO_r * O_r, axis=1)
        
        L_tile = L_vec[:, None]
        D_tile = D_val[:, None]
        
        S = (Q_l @ K_l.T + Q_r @ K_r.T) * tau
        
        mask = (abs_i[:, None] >= abs_j[None, :]) & (abs_i[:, None] < S_len) & (abs_j[None, :] < S_len)
        
        P = tl.exp(S - L_tile)
        P = P * mask
        
        dP = dO_l @ V_l.T + dO_r @ V_r.T
        
        dS = P * (dP - D_tile) * tau
        dS = dS * mask
        
        dV_acc_l = dV_acc_l + P.T @ dO_l
        dV_acc_r = dV_acc_r + P.T @ dO_r
        
        dK_acc_l = dK_acc_l + dS.T @ Q_l
        dK_acc_r = dK_acc_r + dS.T @ Q_r
        
    out_off_k = b_idx * stride_b + h_idx * stride_h + j * 128 * BLOCK_M
    
    dk_ptrs_l = out_off_k + row_idx[:, None] * 128 + col_idx_64[None, :]
    dk_mask = (j * BLOCK_M + row_idx) < S_len
    tl.store(dK_ptr + dk_ptrs_l, dK_acc_l.to(tl.bfloat16), mask=dk_mask[:, None])
    
    dk_ptrs_r = out_off_k + 64 + row_idx[:, None] * 128 + col_idx_64[None, :]
    tl.store(dK_ptr + dk_ptrs_r, dK_acc_r.to(tl.bfloat16), mask=dk_mask[:, None])
    
    dv_ptrs_l = out_off_k + row_idx[:, None] * 128 + col_idx_64[None, :]
    tl.store(dV_ptr + dv_ptrs_l, dV_acc_l.to(tl.bfloat16), mask=dk_mask[:, None])
    
    dv_ptrs_r = out_off_k + 64 + row_idx[:, None] * 128 + col_idx_64[None, :]
    tl.store(dV_ptr + dv_ptrs_r, dV_acc_r.to(tl.bfloat16), mask=dk_mask[:, None])


@triton.jit
def _dq_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, O_ptr, L_ptr, dQ_ptr,
    S_len, H, tau,
    BLOCK_M: tl.constexpr,
):
    i = tl.program_id(0)
    h_idx = tl.program_id(1)
    b_idx = tl.program_id(2)
    num_blocks = tl.cdiv(S_len, BLOCK_M)
    
    row_idx = tl.arange(0, BLOCK_M)
    col_idx = tl.arange(0, BLOCK_M)
    col_idx_64 = tl.arange(0, 64)
    
    stride_b = H * S_len * 128
    stride_h = S_len * 128
    
    q_off_left = b_idx * stride_b + h_idx * stride_h + i * 128 * BLOCK_M + 0
    q_off_right = b_idx * stride_b + h_idx * stride_h + i * 128 * BLOCK_M + 64
    
    Q_l = tl.load(Q_ptr + q_off_left + row_idx[:, None] * 128 + col_idx[None, :],
                  mask=(i * BLOCK_M + row_idx)[:, None] < S_len, other=0.0)
    Q_r = tl.load(Q_ptr + q_off_right + row_idx[:, None] * 128 + col_idx[None, :],
                  mask=(i * BLOCK_M + row_idx)[:, None] < S_len, other=0.0)
    
    do_off_left = b_idx * stride_b + h_idx * stride_h + i * 128 * BLOCK_M + 0
    do_off_right = b_idx * stride_b + h_idx * stride_h + i * 128 * BLOCK_M + 64
    
    dO_l = tl.load(dO_ptr + do_off_left + row_idx[:, None] * 128 + col_idx[None, :],
                   mask=(i * BLOCK_M + row_idx)[:, None] < S_len, other=0.0)
    dO_r = tl.load(dO_ptr + do_off_right + row_idx[:, None] * 128 + col_idx[None, :],
                   mask=(i * BLOCK_M + row_idx)[:, None] < S_len, other=0.0)
    
    o_off_left = b_idx * stride_b + h_idx * stride_h + i * 128 * BLOCK_M + 0
    o_off_right = b_idx * stride_b + h_idx * stride_h + i * 128 * BLOCK_M + 64
    
    O_l = tl.load(O_ptr + o_off_left + row_idx[:, None] * 128 + col_idx[None, :],
                  mask=(i * BLOCK_M + row_idx)[:, None] < S_len, other=0.0)
    O_r = tl.load(O_ptr + o_off_right + row_idx[:, None] * 128 + col_idx[None, :],
                  mask=(i * BLOCK_M + row_idx)[:, None] < S_len, other=0.0)
    
    abs_i = i * BLOCK_M + row_idx
    L_stride_b = H * S_len
    L_stride_h = S_len
    L_vec = tl.load(L_ptr + b_idx * L_stride_b + h_idx * L_stride_h + abs_i, mask=(abs_i < S_len), other=1e20)
    
    D_val = tl.sum(dO_l * O_l + dO_r * O_r, axis=1)
    
    L_tile = L_vec[:, None]
    D_tile = D_val[:, None]
    
    dQ_acc_l = tl.zeros((BLOCK_M, 64), tl.float32)
    dQ_acc_r = tl.zeros((BLOCK_M, 64), tl.float32)
    
    for j_inner in range(0, i + 1):
        k_off_left = b_idx * stride_b + h_idx * stride_h + j_inner * 128 * BLOCK_M + 0
        k_off_right = b_idx * stride_b + h_idx * stride_h + j_inner * 128 * BLOCK_M + 64
        
        K_l = tl.load(K_ptr + k_off_left + row_idx[:, None] * 128 + col_idx[None, :],
                      mask=(j_inner * BLOCK_M + row_idx)[:, None] < S_len, other=0.0)
        K_r = tl.load(K_ptr + k_off_right + row_idx[:, None] * 128 + col_idx[None, :],
                      mask=(j_inner * BLOCK_M + row_idx)[:, None] < S_len, other=0.0)
        
        v_off_left = b_idx * stride_b + h_idx * stride_h + j_inner * 128 * BLOCK_M + 0
        v_off_right = b_idx * stride_b + h_idx * stride_h + j_inner * 128 * BLOCK_M + 64
        
        V_l = tl.load(V_ptr + v_off_left + row_idx[:, None] * 128 + col_idx[None, :],
                      mask=(j_inner * BLOCK_M + row_idx)[:, None] < S_len, other=0.0)
        V_r = tl.load(V_ptr + v_off_right + row_idx[:, None] * 128 + col_idx[None, :],
                      mask=(j_inner * BLOCK_M + row_idx)[:, None] < S_len, other=0.0)
        
        S = (Q_l @ K_l.T + Q_r @ K_r.T) * tau
        
        abs_j = j_inner * BLOCK_M + col_idx
        mask = (abs_i[:, None] >= abs_j[None, :]) & (abs_i[:, None] < S_len) & (abs_j[None, :] < S_len)
        
        P = tl.exp(S - L_tile)
        P = P * mask
        
        dP = dO_l @ V_l.T + dO_r @ V_r.T
        
        dS = P * (dP - D_tile) * tau
        dS = dS * mask
        
        dQ_acc_l = dQ_acc_l + dS @ K_l
        dQ_acc_r = dQ_acc_r + dS @ K_r
        
    out_off_q = b_idx * stride_b + h_idx * stride_h + i * 128 * BLOCK_M
    
    dq_ptrs_l = out_off_q + row_idx[:, None] * 128 + col_idx_64[None, :]
    dq_mask = (i * BLOCK_M + row_idx) < S_len
    tl.store(dQ_ptr + dq_ptrs_l, dQ_acc_l.to(tl.bfloat16), mask=dq_mask[:, None])
    
    dq_ptrs_r = out_off_q + 64 + row_idx[:, None] * 128 + col_idx_64[None, :]
    tl.store(dQ_ptr + dq_ptrs_r, dQ_acc_r.to(tl.bfloat16), mask=dq_mask[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    
    B, H, S_len, d = Q.shape
    tau = 1.0 / math.sqrt(d)
    
    BLOCK_M = 64
    
    T_r = triton.cdiv(S_len, BLOCK_M)
    grid = (T_r, H, B)
    
    _dkdv_kernel[grid](
        Q, K, V, dO, O, L, dK, dV,
        S_len, H, tau,
        BLOCK_M=BLOCK_M,
        num_warps=8, num_stages=2
    )
    
    _dq_kernel[grid](
        Q, K, V, dO, O, L, dQ,
        S_len, H, tau,
        BLOCK_M=BLOCK_M,
        num_warps=8, num_stages=2
    )