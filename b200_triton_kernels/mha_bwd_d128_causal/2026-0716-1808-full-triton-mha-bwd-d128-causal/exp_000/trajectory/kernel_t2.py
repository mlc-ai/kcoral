import torch
import triton
import triton.language as tl
import math


@triton.jit
def load_bf16_block(base_ptr, row, col, S):
    mask = ((row[:, None] < S) & (col[None, :] < 128))
    ptrs = base_ptr + row[:, None] * 128 + col[None, :]
    return tl.load(ptrs, mask=mask, other=0.0)


@triton.jit
def store_bf16_block(base_ptr, row, col, value, S):
    mask = ((row[:, None] < S) & (col[None, :] < 128))
    ptrs = base_ptr + row[:, None] * 128 + col[None, :]
    tl.store(ptrs, value.to(tl.bfloat16), mask=mask)


@triton.jit
def mha_bwd_dQ_kernel(
    Q, K, V, O, dO, L, dQ, S
):
    bh = tl.program_id(0)
    pid_s = tl.program_id(1)
    
    r_base = pid_s * 64
    
    Q_bh = Q + bh * S * 128
    K_bh = K + bh * S * 128
    V_bh = V + bh * S * 128
    O_bh = O + bh * S * 128
    dO_bh = dO + bh * S * 128
    L_bh = L + bh * S
    dQ_bh = dQ + bh * S * 128
    
    scale_factor = 1.0 / math.sqrt(128)
    
    dQ_acc0 = tl.zeros((64, 64), dtype=tl.float32)
    dQ_acc1 = tl.zeros((64, 64), dtype=tl.float32)
    
    r_idx = tl.arange(0, 64)
    c_idx = tl.arange(0, 64)
    
    Q_tile_0 = load_bf16_block(Q_bh, r_base + r_idx, c_idx, S)
    Q_tile_1 = load_bf16_block(Q_bh, r_base + r_idx, c_idx + 64, S)
    O_tile_0 = load_bf16_block(O_bh, r_base + r_idx, c_idx, S)
    O_tile_1 = load_bf16_block(O_bh, r_base + r_idx, c_idx + 64, S)
    dO_tile_0 = load_bf16_block(dO_bh, r_base + r_idx, c_idx, S)
    dO_tile_1 = load_bf16_block(dO_bh, r_base + r_idx, c_idx + 64, S)
    
    for k_base in range(0, min(r_base + 64, S), 64):
        K_tile_0 = load_bf16_block(K_bh, k_base + r_idx, c_idx, S)
        K_tile_1 = load_bf16_block(K_bh, k_base + r_idx, c_idx + 64, S)
        V_tile_0 = load_bf16_block(V_bh, k_base + r_idx, c_idx, S)
        V_tile_1 = load_bf16_block(V_bh, k_base + r_idx, c_idx + 64, S)
        
        S_h = (tl.dot(Q_tile_0, K_tile_0.T) + tl.dot(Q_tile_1, K_tile_1.T)) / scale_factor
        
        L_h = L_bh[r_base + r_idx][:, None]
        P_h = tl.math.exp(S_h - L_h)
        
        mask_causal = (r_base + r_idx[:, None]) >= (k_base + c_idx[None, :])
        mask_h = mask_causal & ((r_base + r_idx[:, None]) < S) & ((k_base + c_idx[None, :]) < S)
        P_h = P_h * mask_h
        
        dP_h = (tl.dot(dO_tile_0, V_tile_0.T) + tl.dot(dO_tile_1, V_tile_1.T))
        O_sum_exp_h = (dO_tile_0 * O_tile_0 + dO_tile_1 * O_tile_1).sum(axis=1)[:, None]
        D_S_h = P_h * (dP_h - O_sum_exp_h)
        
        dQ_acc0 += tl.dot(D_S_h, K_tile_0) / scale_factor
        dQ_acc1 += tl.dot(D_S_h, K_tile_1) / scale_factor
        
    store_bf16_block(dQ_bh, r_base + r_idx, c_idx, dQ_acc0, S)
    store_bf16_block(dQ_bh, r_base + r_idx, c_idx + 64, dQ_acc1, S)


@triton.jit
def mha_bwd_dK_kernel(
    Q, K, V, O, dO, L, dK, S
):
    bh = tl.program_id(0)
    pid_s = tl.program_id(1)
    
    k_base = pid_s * 64
    
    Q_bh = Q + bh * S * 128
    K_bh = K + bh * S * 128
    V_bh = V + bh * S * 128
    O_bh = O + bh * S * 128
    dO_bh = dO + bh * S * 128
    L_bh = L + bh * S
    dK_bh = dK + bh * S * 128
    
    scale_factor = 1.0 / math.sqrt(128)
    
    dK_acc0 = tl.zeros((64, 64), dtype=tl.float32)
    dK_acc1 = tl.zeros((64, 64), dtype=tl.float32)
    
    r_idx = tl.arange(0, 64)
    c_idx = tl.arange(0, 64)
    
    K_tile_0 = load_bf16_block(K_bh, k_base + r_idx, c_idx, S)
    K_tile_1 = load_bf16_block(K_bh, k_base + r_idx, c_idx + 64, S)
    
    for r_base in range(k_base, S, 64):
        Q_tile_0 = load_bf16_block(Q_bh, r_base + r_idx, c_idx, S)
        Q_tile_1 = load_bf16_block(Q_bh, r_base + r_idx, c_idx + 64, S)
        O_tile_0 = load_bf16_block(O_bh, r_base + r_idx, c_idx, S)
        O_tile_1 = load_bf16_block(O_bh, r_base + r_idx, c_idx + 64, S)
        dO_tile_0 = load_bf16_block(dO_bh, r_base + r_idx, c_idx, S)
        dO_tile_1 = load_bf16_block(dO_bh, r_base + r_idx, c_idx + 64, S)
        
        S_h = (tl.dot(Q_tile_0, K_tile_0.T) + tl.dot(Q_tile_1, K_tile_1.T)) / scale_factor
        
        L_h = L_bh[r_base + r_idx][:, None]
        P_h = tl.math.exp(S_h - L_h)
        
        mask_causal = (r_base + r_idx[:, None]) >= (k_base + c_idx[None, :])
        mask_h = mask_causal & ((r_base + r_idx[:, None]) < S) & ((k_base + c_idx[None, :]) < S)
        P_h = P_h * mask_h
        
        dP_h = (tl.dot(dO_tile_0, V_tile_0.T) + tl.dot(dO_tile_1, V_tile_1.T))
        O_sum_exp_h = (dO_tile_0 * O_tile_0 + dO_tile_1 * O_tile_1).sum(axis=1)[:, None]
        D_S_h = P_h * (dP_h - O_sum_exp_h)
        
        D_S_h_T = D_S_h.T
        dK_acc0 += tl.dot(D_S_h_T, Q_tile_0) / scale_factor
        dK_acc1 += tl.dot(D_S_h_T, Q_tile_1) / scale_factor
        
    store_bf16_block(dK_bh, k_base + r_idx, c_idx, dK_acc0, S)
    store_bf16_block(dK_bh, k_base + r_idx, c_idx + 64, dK_acc1, S)


@triton.jit
def mha_bwd_dV_kernel(
    Q, K, V, O, dO, L, dV, S
):
    bh = tl.program_id(0)
    pid_s = tl.program_id(1)
    
    k_base = pid_s * 64
    
    Q_bh = Q + bh * S * 128
    K_bh = K + bh * S * 128
    O_bh = O + bh * S * 128
    dO_bh = dO + bh * S * 128
    L_bh = L + bh * S
    dV_bh = dV + bh * S * 128
    
    scale_factor = 1.0 / math.sqrt(128)
    
    dV_acc0 = tl.zeros((64, 64), dtype=tl.float32)
    dV_acc1 = tl.zeros((64, 64), dtype=tl.float32)
    
    r_idx = tl.arange(0, 64)
    c_idx = tl.arange(0, 64)
    
    K_tile_0 = load_bf16_block(K_bh, k_base + r_idx, c_idx, S)
    K_tile_1 = load_bf16_block(K_bh, k_base + r_idx, c_idx + 64, S)
    
    for r_base in range(k_base, S, 64):
        Q_tile_0 = load_bf16_block(Q_bh, r_base + r_idx, c_idx, S)
        Q_tile_1 = load_bf16_block(Q_bh, r_base + r_idx, c_idx + 64, S)
        dO_tile_0 = load_bf16_block(dO_bh, r_base + r_idx, c_idx, S)
        dO_tile_1 = load_bf16_block(dO_bh, r_base + r_idx, c_idx + 64, S)
        
        S_h = (tl.dot(Q_tile_0, K_tile_0.T) + tl.dot(Q_tile_1, K_tile_1.T)) / scale_factor
        
        L_h = L_bh[r_base + r_idx][:, None]
        P_h = tl.math.exp(S_h - L_h)
        
        mask_causal = (r_base + r_idx[:, None]) >= (k_base + c_idx[None, :])
        mask_h = mask_causal & ((r_base + r_idx[:, None]) < S) & ((k_base + c_idx[None, :]) < S)
        P_h = P_h * mask_h
        
        P_h_T = P_h.T
        dV_acc0 += tl.dot(P_h_T, dO_tile_0)
        dV_acc1 += tl.dot(P_h_T, dO_tile_1)
        
    store_bf16_block(dV_bh, k_base + r_idx, c_idx, dV_acc0, S)
    store_bf16_block(dV_bh, k_base + r_idx, c_idx + 64, dV_acc1, S)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    b, h, s, d = Q.shape
    S = s
    grid = (b * h, triton.cdiv(S, 64))
    
    mha_bwd_dQ_kernel[grid](Q, K, V, O, dO, L, dQ, S, num_warps=8)
    mha_bwd_dK_kernel[grid](Q, K, V, O, dO, L, dK, S, num_warps=8)
    mha_bwd_dV_kernel[grid](Q, K, V, O, dO, L, dV, S, num_warps=8)