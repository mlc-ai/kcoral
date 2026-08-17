import torch
import triton
import triton.language as tl
import math


@triton.jit
def load_bf16_block(base_ptr, row, col, mask):
    x = base_ptr + row[:, None] * 128 + col[None, :]
    return tl.load(x, mask=mask, other=0.0)


@triton.jit
def store_bf16_block(base_ptr, row, col, value, mask):
    x = base_ptr + row[:, None] * 128 + col[None, :]
    tl.store(x, value.to(tl.bfloat16), mask=mask)


@triton.jit
def mha_bwd_dQ_kernel(
    Q, K, V, O, dO, L, dQ, S
):
    bh = tl.program_id(0)
    pid_s = tl.program_id(1)
    
    r_base = pid_s * 128
    
    Q_bh = Q + bh * S * 128
    K_bh = K + bh * S * 128
    V_bh = V + bh * S * 128
    O_bh = O + bh * S * 128
    dO_bh = dO + bh * S * 128
    L_bh = L + bh * S
    dQ_bh = dQ + bh * S * 128
    
    dQ_acc = tl.zeros((128, 128), dtype=tl.float32)
    
    r_idx = tl.arange(0, 128)
    c_idx = tl.arange(0, 128)
    d_idx = tl.arange(0, 128)
    
    mask_Q = ((r_base + r_idx[:, None]) < S)
    Q_tile = load_bf16_block(Q_bh, r_base + r_idx, d_idx, mask_Q)
    O_tile = load_bf16_block(O_bh, r_base + r_idx, d_idx, mask_Q)
    dO_tile = load_bf16_block(dO_bh, r_base + r_idx, d_idx, mask_Q)
    
    for k_base_inner in range(0, r_base + 128, 128):
        mask_K = ((k_base_inner + r_idx[:, None]) < S)
        K_tile = load_bf16_block(K_bh, k_base_inner + r_idx, d_idx, mask_K)
        V_tile = load_bf16_block(V_bh, k_base_inner + r_idx, d_idx, mask_K)
        
        K_tile_T = K_tile.T
        S_h = tl.dot(Q_tile, K_tile_T) / math.sqrt(128)
        
        L_h = L_bh[r_base + r_idx][:, None]
        P_h = tl.math.exp(S_h - L_h)
        
        mask_causal = (r_base + r_idx[:, None]) >= (k_base_inner + c_idx[None, :])
        mask_h = mask_causal & ((r_base + r_idx[:, None]) < S) & ((k_base_inner + c_idx[None, :]) < S)
        P_h = P_h * mask_h
        
        V_tile_T = V_tile.T
        dP_h = tl.dot(dO_tile, V_tile_T)
        
        O_sum_exp_h = (dO_tile * O_tile).sum(axis=1)[:, None]
        
        D_S_h = P_h * (dP_h - O_sum_exp_h)
        
        dQ_acc += tl.dot(D_S_h, K_tile) / math.sqrt(128)
    
    store_bf16_block(dQ_bh, r_base + r_idx, d_idx, dQ_acc, mask_Q)


@triton.jit
def mha_bwd_dK_kernel(
    Q, K, V, O, dO, L, dK, S
):
    bh = tl.program_id(0)
    pid_s = tl.program_id(1)
    
    k_base = pid_s * 128
    
    Q_bh = Q + bh * S * 128
    K_bh = K + bh * S * 128
    V_bh = V + bh * S * 128
    O_bh = O + bh * S * 128
    dO_bh = dO + bh * S * 128
    L_bh = L + bh * S
    dK_bh = dK + bh * S * 128
    
    dK_acc = tl.zeros((128, 128), dtype=tl.float32)
    
    r_idx = tl.arange(0, 128)
    c_idx = tl.arange(0, 128)
    d_idx = tl.arange(0, 128)
    
    mask_K = ((k_base + r_idx[:, None]) < S)
    K_tile = load_bf16_block(K_bh, k_base + r_idx, d_idx, mask_K)
    
    for r_base_inner in range(k_base, S, 128):
        mask_Q = ((r_base_inner + r_idx[:, None]) < S)
        Q_tile = load_bf16_block(Q_bh, r_base_inner + r_idx, d_idx, mask_Q)
        O_tile = load_bf16_block(O_bh, r_base_inner + r_idx, d_idx, mask_Q)
        dO_tile = load_bf16_block(dO_bh, r_base_inner + r_idx, d_idx, mask_Q)
        
        K_tile_T = K_tile.T
        S_h = tl.dot(Q_tile, K_tile_T) / math.sqrt(128)
        
        L_h = L_bh[r_base_inner + r_idx][:, None]
        P_h = tl.math.exp(S_h - L_h)
        
        mask_causal = (r_base_inner + r_idx[:, None]) >= (k_base + c_idx[None, :])
        mask_h = mask_causal & ((r_base_inner + r_idx[:, None]) < S) & ((k_base + c_idx[None, :]) < S)
        P_h = P_h * mask_h
        
        V_tile = load_bf16_block(V_bh, k_base + r_idx, d_idx, mask_K)
        V_tile_T = V_tile.T
        dP_h = tl.dot(dO_tile, V_tile_T)
        
        O_sum_exp_h = (dO_tile * O_tile).sum(axis=1)[:, None]
        
        D_S_h = P_h * (dP_h - O_sum_exp_h)
        
        D_S_h_T = D_S_h.T
        dK_acc += tl.dot(D_S_h_T, Q_tile) / math.sqrt(128)
    
    store_bf16_block(dK_bh, k_base + r_idx, d_idx, dK_acc, mask_K)


@triton.jit
def mha_bwd_dV_kernel(
    Q, K, V, O, dO, L, dV, S
):
    bh = tl.program_id(0)
    pid_s = tl.program_id(1)
    
    k_base = pid_s * 128
    
    Q_bh = Q + bh * S * 128
    K_bh = K + bh * S * 128
    O_bh = O + bh * S * 128
    dO_bh = dO + bh * S * 128
    L_bh = L + bh * S
    dV_bh = dV + bh * S * 128
    
    dV_acc = tl.zeros((128, 128), dtype=tl.float32)
    
    r_idx = tl.arange(0, 128)
    c_idx = tl.arange(0, 128)
    d_idx = tl.arange(0, 128)
    
    mask_K = ((k_base + r_idx[:, None]) < S)
    K_tile = load_bf16_block(K_bh, k_base + r_idx, d_idx, mask_K)
    
    for r_base_inner in range(k_base, S, 128):
        mask_Q = ((r_base_inner + r_idx[:, None]) < S)
        Q_tile = load_bf16_block(Q_bh, r_base_inner + r_idx, d_idx, mask_Q)
        dO_tile = load_bf16_block(dO_bh, r_base_inner + r_idx, d_idx, mask_Q)
        
        K_tile_T = K_tile.T
        S_h = tl.dot(Q_tile, K_tile_T) / math.sqrt(128)
        
        L_h = L_bh[r_base_inner + r_idx][:, None]
        P_h = tl.math.exp(S_h - L_h)
        
        mask_causal = (r_base_inner + r_idx[:, None]) >= (k_base + c_idx[None, :])
        mask_h = mask_causal & ((r_base_inner + r_idx[:, None]) < S) & ((k_base + c_idx[None, :]) < S)
        P_h = P_h * mask_h
        
        P_h_T = P_h.T
        dV_acc += tl.dot(P_h_T, dO_tile)
    
    store_bf16_block(dV_bh, k_base + r_idx, d_idx, dV_acc, mask_K)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    b, h, s, d = Q.shape
    S = s
    grid = (b * h, triton.cdiv(S, 128))
    
    mha_bwd_dQ_kernel[grid](Q, K, V, O, dO, L, dQ, S, num_warps=8)
    mha_bwd_dK_kernel[grid](Q, K, V, O, dO, L, dK, S, num_warps=8)
    mha_bwd_dV_kernel[grid](Q, K, V, O, dO, L, dV, S, num_warps=8)