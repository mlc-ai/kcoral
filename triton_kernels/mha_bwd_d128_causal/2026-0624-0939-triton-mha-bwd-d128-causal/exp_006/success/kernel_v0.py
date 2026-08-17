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
    
    k_rows = tl.arange(0, BLOCK_M)
    col_idx_64 = tl.arange(0, 64)
    
    base = (b_idx * H + h_idx) * S_len * 128
    
    K_l = tl.load(K_ptr + base + j * 128 * BLOCK_M + k_rows[:, None] * 128 + col_idx_64[None, :],
                  mask=(j * BLOCK_M + k_rows)[:, None] < S_len, other=0.0).to(tl.float32)
    K_r = tl.load(K_ptr + base + j * 128 * BLOCK_M + 64 + k_rows[:, None] * 128 + col_idx_64[None, :],
                  mask=(j * BLOCK_M + k_rows)[:, None] < S_len, other=0.0).to(tl.float32)
    V_l = tl.load(V_ptr + base + j * 128 * BLOCK_M + k_rows[:, None] * 128 + col_idx_64[None, :],
                  mask=(j * BLOCK_M + k_rows)[:, None] < S_len, other=0.0).to(tl.float32)
    V_r = tl.load(V_ptr + base + j * 128 * BLOCK_M + 64 + k_rows[:, None] * 128 + col_idx_64[None, :],
                  mask=(j * BLOCK_M + k_rows)[:, None] < S_len, other=0.0).to(tl.float32)
    
    dK_acc_l = tl.zeros((BLOCK_M, 64), tl.float32)
    dK_acc_r = tl.zeros((BLOCK_M, 64), tl.float32)
    dV_acc_l = tl.zeros((BLOCK_M, 64), tl.float32)
    dV_acc_r = tl.zeros((BLOCK_M, 64), tl.float32)
    
    abs_j = j * BLOCK_M + k_rows
    
    for i in range(j, num_blocks):
        q_rows = tl.arange(0, BLOCK_M)
        
        Q_l = tl.load(Q_ptr + base + i * 128 * BLOCK_M + q_rows[:, None] * 128 + col_idx_64[None, :],
                      mask=(i * BLOCK_M + q_rows)[:, None] < S_len, other=0.0).to(tl.float32)
        Q_r = tl.load(Q_ptr + base + i * 128 * BLOCK_M + 64 + q_rows[:, None] * 128 + col_idx_64[None, :],
                      mask=(i * BLOCK_M + q_rows)[:, None] < S_len, other=0.0).to(tl.float32)
        dO_l = tl.load(dO_ptr + base + i * 128 * BLOCK_M + q_rows[:, None] * 128 + col_idx_64[None, :],
                       mask=(i * BLOCK_M + q_rows)[:, None] < S_len, other=0.0).to(tl.float32)
        dO_r = tl.load(dO_ptr + base + i * 128 * BLOCK_M + 64 + q_rows[:, None] * 128 + col_idx_64[None, :],
                       mask=(i * BLOCK_M + q_rows)[:, None] < S_len, other=0.0).to(tl.float32)
        O_l = tl.load(O_ptr + base + i * 128 * BLOCK_M + q_rows[:, None] * 128 + col_idx_64[None, :],
                      mask=(i * BLOCK_M + q_rows)[:, None] < S_len, other=0.0).to(tl.float32)
        O_r = tl.load(O_ptr + base + i * 128 * BLOCK_M + 64 + q_rows[:, None] * 128 + col_idx_64[None, :],
                      mask=(i * BLOCK_M + q_rows)[:, None] < S_len, other=0.0).to(tl.float32)
        
        abs_i = i * BLOCK_M + q_rows
        L_vec = tl.load(L_ptr + (b_idx * H + h_idx) * S_len + abs_i, mask=(abs_i < S_len), other=1e20)
        
        D_val = tl.sum(dO_l * O_l + dO_r * O_r, axis=1)
        
        S = (tl.dot(Q_l, K_l.T) + tl.dot(Q_r, K_r.T)) * tau
        
        mask = (abs_i[:, None] >= abs_j[None, :]) & (abs_i[:, None] < S_len) & (abs_j[None, :] < S_len)
        
        P = tl.exp(S - L_vec[:, None])
        P = P * mask
        
        dP = tl.dot(dO_l, V_l.T) + tl.dot(dO_r, V_r.T)
        
        dS = P * (dP - D_val[:, None]) * tau
        dS = dS * mask
        
        dV_acc_l = tl.dot(P.T, dO_l, dV_acc_l)
        dV_acc_r = tl.dot(P.T, dO_r, dV_acc_r)
        
        dK_acc_l = tl.dot(dS.T, Q_l, dK_acc_l)
        dK_acc_r = tl.dot(dS.T, Q_r, dK_acc_r)
        
    out_off_k = base + j * 128 * BLOCK_M
    dk_ptrs_l = out_off_k + k_rows[:, None] * 128 + col_idx_64[None, :]
    dk_mask = (j * BLOCK_M + k_rows) < S_len
    tl.store(dK_ptr + dk_ptrs_l, dK_acc_l.to(tl.bfloat16), mask=dk_mask[:, None])
    
    dk_ptrs_r = out_off_k + 64 + k_rows[:, None] * 128 + col_idx_64[None, :]
    tl.store(dK_ptr + dk_ptrs_r, dK_acc_r.to(tl.bfloat16), mask=dk_mask[:, None])
    
    dv_ptrs_l = out_off_k + k_rows[:, None] * 128 + col_idx_64[None, :]
    tl.store(dV_ptr + dv_ptrs_l, dV_acc_l.to(tl.bfloat16), mask=dk_mask[:, None])
    
    dv_ptrs_r = out_off_k + 64 + k_rows[:, None] * 128 + col_idx_64[None, :]
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
    
    q_rows = tl.arange(0, BLOCK_M)
    col_idx_64 = tl.arange(0, 64)
    
    base = (b_idx * H + h_idx) * S_len * 128
    
    Q_l = tl.load(Q_ptr + base + i * 128 * BLOCK_M + q_rows[:, None] * 128 + col_idx_64[None, :],
                  mask=(i * BLOCK_M + q_rows)[:, None] < S_len, other=0.0).to(tl.float32)
    Q_r = tl.load(Q_ptr + base + i * 128 * BLOCK_M + 64 + q_rows[:, None] * 128 + col_idx_64[None, :],
                  mask=(i * BLOCK_M + q_rows)[:, None] < S_len, other=0.0).to(tl.float32)
    dO_l = tl.load(dO_ptr + base + i * 128 * BLOCK_M + q_rows[:, None] * 128 + col_idx_64[None, :],
                   mask=(i * BLOCK_M + q_rows)[:, None] < S_len, other=0.0).to(tl.float32)
    dO_r = tl.load(dO_ptr + base + i * 128 * BLOCK_M + 64 + q_rows[:, None] * 128 + col_idx_64[None, :],
                   mask=(i * BLOCK_M + q_rows)[:, None] < S_len, other=0.0).to(tl.float32)
    O_l = tl.load(O_ptr + base + i * 128 * BLOCK_M + q_rows[:, None] * 128 + col_idx_64[None, :],
                  mask=(i * BLOCK_M + q_rows)[:, None] < S_len, other=0.0).to(tl.float32)
    O_r = tl.load(O_ptr + base + i * 128 * BLOCK_M + 64 + q_rows[:, None] * 128 + col_idx_64[None, :],
                  mask=(i * BLOCK_M + q_rows)[:, None] < S_len, other=0.0).to(tl.float32)
    
    abs_i = i * BLOCK_M + q_rows
    L_vec = tl.load(L_ptr + (b_idx * H + h_idx) * S_len + abs_i, mask=(abs_i < S_len), other=1e20)
    
    D_val = tl.sum(dO_l * O_l + dO_r * O_r, axis=1)
    
    dQ_acc_l = tl.zeros((BLOCK_M, 64), tl.float32)
    dQ_acc_r = tl.zeros((BLOCK_M, 64), tl.float32)
    
    for j_inner in range(0, i + 1):
        k_rows = tl.arange(0, BLOCK_M)
        
        K_l = tl.load(K_ptr + base + j_inner * 128 * BLOCK_M + k_rows[:, None] * 128 + col_idx_64[None, :],
                      mask=(j_inner * BLOCK_M + k_rows)[:, None] < S_len, other=0.0).to(tl.float32)
        K_r = tl.load(K_ptr + base + j_inner * 128 * BLOCK_M + 64 + k_rows[:, None] * 128 + col_idx_64[None, :],
                      mask=(j_inner * BLOCK_M + k_rows)[:, None] < S_len, other=0.0).to(tl.float32)
        V_l = tl.load(V_ptr + base + j_inner * 128 * BLOCK_M + k_rows[:, None] * 128 + col_idx_64[None, :],
                      mask=(j_inner * BLOCK_M + k_rows)[:, None] < S_len, other=0.0).to(tl.float32)
        V_r = tl.load(V_ptr + base + j_inner * 128 * BLOCK_M + 64 + k_rows[:, None] * 128 + col_idx_64[None, :],
                      mask=(j_inner * BLOCK_M + k_rows)[:, None] < S_len, other=0.0).to(tl.float32)
        
        S = (tl.dot(Q_l, K_l.T) + tl.dot(Q_r, K_r.T)) * tau
        
        abs_j = j_inner * BLOCK_M + k_rows
        mask = (abs_i[:, None] >= abs_j[None, :]) & (abs_i[:, None] < S_len) & (abs_j[None, :] < S_len)
        
        P = tl.exp(S - L_vec[:, None])
        P = P * mask
        
        dP = tl.dot(dO_l, V_l.T) + tl.dot(dO_r, V_r.T)
        
        dS = P * (dP - D_val[:, None]) * tau
        dS = dS * mask
        
        dQ_acc_l = tl.dot(dS, K_l, dQ_acc_l)
        dQ_acc_r = tl.dot(dS, K_r, dQ_acc_r)
        
    out_off_q = base + i * 128 * BLOCK_M
    dq_ptrs_l = out_off_q + q_rows[:, None] * 128 + col_idx_64[None, :]
    dq_mask = (i * BLOCK_M + q_rows) < S_len
    tl.store(dQ_ptr + dq_ptrs_l, dQ_acc_l.to(tl.bfloat16), mask=dq_mask[:, None])
    
    dq_ptrs_r = out_off_q + 64 + q_rows[:, None] * 128 + col_idx_64[None, :]
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