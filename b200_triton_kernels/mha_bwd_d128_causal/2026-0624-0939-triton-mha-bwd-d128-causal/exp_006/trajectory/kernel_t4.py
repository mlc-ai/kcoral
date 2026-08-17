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
    
    stride_b = H * S_len * 128
    stride_h = S_len * 128
    
    k_off = b_idx * stride_b + h_idx * stride_h + j * 128 * BLOCK_M
    K_tile = tl.load(K_ptr + k_off + row_idx[:, None] * 128 + col_idx[None, :],
                     mask=(j * BLOCK_M + row_idx)[:, None] < S_len, other=0.0)
    
    v_off = b_idx * stride_b + h_idx * stride_h + j * 128 * BLOCK_M
    V_tile = tl.load(V_ptr + v_off + row_idx[:, None] * 128 + col_idx[None, :],
                     mask=(j * BLOCK_M + row_idx)[:, None] < S_len, other=0.0)
    
    dK_acc = tl.zeros((BLOCK_M, BLOCK_M), tl.float32)
    dV_acc = tl.zeros((BLOCK_M, BLOCK_M), tl.float32)
    
    abs_j = j * BLOCK_M + col_idx
    
    for i in range(j, num_blocks):
        q_off = b_idx * stride_b + h_idx * stride_h + i * 128 * BLOCK_M
        Q_tile = tl.load(Q_ptr + q_off + row_idx[:, None] * 128 + col_idx[None, :],
                         mask=(i * BLOCK_M + row_idx)[:, None] < S_len, other=0.0)
        
        do_off = b_idx * stride_b + h_idx * stride_h + i * 128 * BLOCK_M
        dO_tile = tl.load(dO_ptr + do_off + row_idx[:, None] * 128 + col_idx[None, :],
                          mask=(i * BLOCK_M + row_idx)[:, None] < S_len, other=0.0)
        
        o_off = b_idx * stride_b + h_idx * stride_h + i * 128 * BLOCK_M
        O_tile = tl.load(O_ptr + o_off + row_idx[:, None] * 128 + col_idx[None, :],
                         mask=(i * BLOCK_M + row_idx)[:, None] < S_len, other=0.0)
        
        abs_i = i * BLOCK_M + row_idx
        L_stride_b = H * S_len
        L_stride_h = S_len
        L_vec = tl.load(L_ptr + b_idx * L_stride_b + h_idx * L_stride_h + abs_i, mask=(abs_i < S_len), other=1e20)
        
        D_val = tl.sum(dO_tile * O_tile, axis=1)
        
        L_tile = L_vec[:, None]
        D_tile = D_val[:, None]
        
        S = tl.dot(Q_tile, K_tile.T) * tau
        
        if i == j:
            mask = (abs_i[:, None] >= abs_j[None, :]) & (abs_i[:, None] < S_len) & (abs_j[None, :] < S_len)
        else:
            mask = (abs_i[:, None] < S_len) & (abs_j[None, :] < S_len)
        
        P = tl.exp(S - L_tile)
        P = P * mask
        
        dP = tl.dot(dO_tile, V_tile.T)
        
        dS = P * (dP - D_tile) * tau
        dS = dS * mask
        
        dV_acc = tl.dot(P.T, dO_tile, dV_acc)
        dK_acc = tl.dot(dS.T, Q_tile, dK_acc)
        
    out_off_k = b_idx * stride_b + h_idx * stride_h + j * 128 * BLOCK_M
    
    dk_ptrs = out_off_k + row_idx[:, None] * 128 + col_idx[None, :]
    dk_mask = (j * BLOCK_M + row_idx) < S_len
    tl.store(dK_ptr + dk_ptrs, dK_acc.to(tl.bfloat16), mask=dk_mask[:, None])
    
    dv_ptrs = out_off_k + row_idx[:, None] * 128 + col_idx[None, :]
    tl.store(dV_ptr + dv_ptrs, dV_acc.to(tl.bfloat16), mask=dk_mask[:, None])


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
    
    stride_b = H * S_len * 128
    stride_h = S_len * 128
    
    q_off = b_idx * stride_b + h_idx * stride_h + i * 128 * BLOCK_M
    Q_tile = tl.load(Q_ptr + q_off + row_idx[:, None] * 128 + col_idx[None, :],
                     mask=(i * BLOCK_M + row_idx)[:, None] < S_len, other=0.0)
    
    do_off = b_idx * stride_b + h_idx * stride_h + i * 128 * BLOCK_M
    dO_tile = tl.load(dO_ptr + do_off + row_idx[:, None] * 128 + col_idx[None, :],
                      mask=(i * BLOCK_M + row_idx)[:, None] < S_len, other=0.0)
    
    o_off = b_idx * stride_b + h_idx * stride_h + i * 128 * BLOCK_M
    O_tile = tl.load(O_ptr + o_off + row_idx[:, None] * 128 + col_idx[None, :],
                     mask=(i * BLOCK_M + row_idx)[:, None] < S_len, other=0.0)
    
    abs_i = i * BLOCK_M + row_idx
    L_stride_b = H * S_len
    L_stride_h = S_len
    L_vec = tl.load(L_ptr + b_idx * L_stride_b + h_idx * L_stride_h + abs_i, mask=(abs_i < S_len), other=1e20)
    
    D_val = tl.sum(dO_tile * O_tile, axis=1)
    
    L_tile = L_vec[:, None]
    D_tile = D_val[:, None]
    
    dQ_acc = tl.zeros((BLOCK_M, BLOCK_M), tl.float32)
    
    for j_inner in range(0, i + 1):
        k_off = b_idx * stride_b + h_idx * stride_h + j_inner * 128 * BLOCK_M
        K_tile = tl.load(K_ptr + k_off + row_idx[:, None] * 128 + col_idx[None, :],
                         mask=(j_inner * BLOCK_M + row_idx)[:, None] < S_len, other=0.0)
        
        v_off = b_idx * stride_b + h_idx * stride_h + j_inner * 128 * BLOCK_M
        V_tile = tl.load(V_ptr + v_off + row_idx[:, None] * 128 + col_idx[None, :],
                         mask=(j_inner * BLOCK_M + row_idx)[:, None] < S_len, other=0.0)
        
        S = tl.dot(Q_tile, K_tile.T) * tau
        
        abs_j = j_inner * BLOCK_M + col_idx
        if i == j_inner:
            mask = (abs_i[:, None] >= abs_j[None, :]) & (abs_i[:, None] < S_len) & (abs_j[None, :] < S_len)
        else:
            mask = (abs_i[:, None] < S_len) & (abs_j[None, :] < S_len)
        
        P = tl.exp(S - L_tile)
        P = P * mask
        
        dP = tl.dot(dO_tile, V_tile.T)
        
        dS = P * (dP - D_tile) * tau
        dS = dS * mask
        
        dQ_acc = tl.dot(dS, K_tile, dQ_acc)
        
    out_off_q = b_idx * stride_b + h_idx * stride_h + i * 128 * BLOCK_M
    
    dq_ptrs = out_off_q + row_idx[:, None] * 128 + col_idx[None, :]
    dq_mask = (i * BLOCK_M + row_idx) < S_len
    tl.store(dQ_ptr + dq_ptrs, dQ_acc.to(tl.bfloat16), mask=dq_mask[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    
    B, H, S_len, d = Q.shape
    tau = 1.0 / math.sqrt(d)
    
    BLOCK_M = 128
    
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