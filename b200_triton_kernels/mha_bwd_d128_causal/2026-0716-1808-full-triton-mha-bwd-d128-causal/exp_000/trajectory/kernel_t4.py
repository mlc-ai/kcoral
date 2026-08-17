import torch
import triton
import triton.language as tl
import math
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def mha_bwd_dQ_kernel(
    Q_desc, K_desc, V_desc, O_desc, dO_desc, L, dQ_desc, S
):
    bh = tl.program_id(1)
    tile = tl.program_id(0)
    
    s_len = 128
    s_start = tile * s_len
    
    num_tiles = tl.cdiv(S, s_len)
    
    L_bh = L + bh * S
    
    scale = 1.0 / math.sqrt(128)
    
    r_idx = tl.arange(0, 128)
    c_idx = tl.arange(0, 128)
    
    Q_0 = Q_desc.load([bh * S + s_start, 0]).to(tl.float32)
    Q_1 = Q_desc.load([bh * S + s_start, 64]).to(tl.float32)
    O_0 = O_desc.load([bh * S + s_start, 0]).to(tl.float32)
    O_1 = O_desc.load([bh * S + s_start, 64]).to(tl.float32)
    dO_0 = dO_desc.load([bh * S + s_start, 0]).to(tl.float32)
    dO_1 = dO_desc.load([bh * S + s_start, 64]).to(tl.float32)
    
    dQ_0 = tl.zeros((128, 64), dtype=tl.float32)
    dQ_1 = tl.zeros((128, 64), dtype=tl.float32)
    
    max_k_tile = min(tile + 1, num_tiles)
    
    for k_tile in range(max_k_tile):
        k_start = k_tile * s_len
        
        K_0 = K_desc.load([bh * S + k_start, 0]).to(tl.float32)
        K_1 = K_desc.load([bh * S + k_start, 64]).to(tl.float32)
        V_0 = V_desc.load([bh * S + k_start, 0]).to(tl.float32)
        V_1 = V_desc.load([bh * S + k_start, 64]).to(tl.float32)
        
        S_h = (tl.dot(Q_0, K_0.T) + tl.dot(Q_1, K_1.T)) * scale
        
        L_h = tl.load(L_bh + s_start + r_idx, mask=(s_start + r_idx < S), other=0.0)
        P_h = tl.math.exp(S_h - L_h[:, None])
        
        global_r = s_start + r_idx[:, None]
        global_c = k_start + c_idx[None, :]
        mask_causal = global_r >= global_c
        mask_h = mask_causal & (global_r < S) & (global_c < S)
        P_h = P_h * mask_h
        
        dP_h = tl.dot(dO_0, V_0.T) + tl.dot(dO_1, V_1.T)
        
        O_sum_exp_h = (O_0 * dO_0 + O_1 * dO_1).sum(axis=1)[:, None]
        D_S_h = P_h * (dP_h - O_sum_exp_h)
        
        dQ_0 += tl.dot(D_S_h, K_0) * scale
        dQ_1 += tl.dot(D_S_h, K_1) * scale
        
    dQ_desc.store([bh * S + s_start, 0], dQ_0.to(tl.bfloat16))
    dQ_desc.store([bh * S + s_start, 64], dQ_1.to(tl.bfloat16))


@triton.jit
def mha_bwd_dK_kernel(
    Q_desc, K_desc, V_desc, O_desc, dO_desc, L, dK_desc, S
):
    bh = tl.program_id(1)
    tile = tl.program_id(0)
    
    s_len = 128
    k_start = tile * s_len
    
    num_tiles = tl.cdiv(S, s_len)
    
    L_bh = L + bh * S
    
    scale = 1.0 / math.sqrt(128)
    
    r_idx = tl.arange(0, 128)
    c_idx = tl.arange(0, 128)
    
    dK_0 = tl.zeros((128, 64), dtype=tl.float32)
    dK_1 = tl.zeros((128, 64), dtype=tl.float32)
    
    K_0 = K_desc.load([bh * S + k_start, 0]).to(tl.float32)
    K_1 = K_desc.load([bh * S + k_start, 64]).to(tl.float32)
    V_0 = V_desc.load([bh * S + k_start, 0]).to(tl.float32)
    V_1 = V_desc.load([bh * S + k_start, 64]).to(tl.float32)
    
    max_q_tile = min(num_tiles, S // s_len + 1)
    
    for q_tile in range(tile, max_q_tile):
        q_start = q_tile * s_len
        
        Q_0 = Q_desc.load([bh * S + q_start, 0]).to(tl.float32)
        Q_1 = Q_desc.load([bh * S + q_start, 64]).to(tl.float32)
        O_0 = O_desc.load([bh * S + q_start, 0]).to(tl.float32)
        O_1 = O_desc.load([bh * S + q_start, 64]).to(tl.float32)
        dO_0 = dO_desc.load([bh * S + q_start, 0]).to(tl.float32)
        dO_1 = dO_desc.load([bh * S + q_start, 64]).to(tl.float32)
        
        S_h = (tl.dot(Q_0, K_0.T) + tl.dot(Q_1, K_1.T)) * scale
        
        L_h = tl.load(L_bh + q_start + r_idx, mask=(q_start + r_idx < S), other=0.0)
        P_h = tl.math.exp(S_h - L_h[:, None])
        
        global_r = q_start + r_idx[:, None]
        global_c = k_start + c_idx[None, :]
        mask_causal = global_r >= global_c
        mask_h = mask_causal & (global_r < S) & (global_c < S)
        P_h = P_h * mask_h
        
        dP_h = tl.dot(dO_0, V_0.T) + tl.dot(dO_1, V_1.T)
        
        O_sum_exp_h = (O_0 * dO_0 + O_1 * dO_1).sum(axis=1)[:, None]
        D_S_h = P_h * (dP_h - O_sum_exp_h)
        
        D_S_h_T = D_S_h.T
        dK_0 += tl.dot(D_S_h_T, Q_0) * scale
        dK_1 += tl.dot(D_S_h_T, Q_1) * scale
        
    dK_desc.store([bh * S + k_start, 0], dK_0.to(tl.bfloat16))
    dK_desc.store([bh * S + k_start, 64], dK_1.to(tl.bfloat16))


@triton.jit
def mha_bwd_dV_kernel(
    Q_desc, K_desc, V_desc, O_desc, dO_desc, L, dV_desc, S
):
    bh = tl.program_id(1)
    tile = tl.program_id(0)
    
    s_len = 128
    k_start = tile * s_len
    
    num_tiles = tl.cdiv(S, s_len)
    
    L_bh = L + bh * S
    
    scale = 1.0 / math.sqrt(128)
    
    r_idx = tl.arange(0, 128)
    c_idx = tl.arange(0, 128)
    
    dV_0 = tl.zeros((128, 64), dtype=tl.float32)
    dV_1 = tl.zeros((128, 64), dtype=tl.float32)
    
    K_0 = K_desc.load([bh * S + k_start, 0]).to(tl.float32)
    K_1 = K_desc.load([bh * S + k_start, 64]).to(tl.float32)
    
    max_q_tile = min(num_tiles, S // s_len + 1)
    
    for q_tile in range(tile, max_q_tile):
        q_start = q_tile * s_len
        
        Q_0 = Q_desc.load([bh * S + q_start, 0]).to(tl.float32)
        Q_1 = Q_desc.load([bh * S + q_start, 64]).to(tl.float32)
        dO_0 = dO_desc.load([bh * S + q_start, 0]).to(tl.float32)
        dO_1 = dO_desc.load([bh * S + q_start, 64]).to(tl.float32)
        
        S_h = (tl.dot(Q_0, K_0.T) + tl.dot(Q_1, K_1.T)) * scale
        
        L_h = tl.load(L_bh + q_start + r_idx, mask=(q_start + r_idx < S), other=0.0)
        P_h = tl.math.exp(S_h - L_h[:, None])
        
        global_r = q_start + r_idx[:, None]
        global_c = k_start + c_idx[None, :]
        mask_causal = global_r >= global_c
        mask_h = mask_causal & (global_r < S) & (global_c < S)
        P_h = P_h * mask_h
        
        P_h_T = P_h.T
        dV_0 += tl.dot(P_h_T, dO_0)
        dV_1 += tl.dot(P_h_T, dO_1)
        
    dV_desc.store([bh * S + k_start, 0], dV_0.to(tl.bfloat16))
    dV_desc.store([bh * S + k_start, 64], dV_1.to(tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    b, h, s, d = Q.shape
    S = s
    
    BLOCK_M = 128
    BLOCK_K = 64
    
    Q_desc = TensorDescriptor.from_tensor(Q.reshape(b * h * S, d), [BLOCK_M, BLOCK_K])
    K_desc = TensorDescriptor.from_tensor(K.reshape(b * h * S, d), [BLOCK_M, BLOCK_K])
    V_desc = TensorDescriptor.from_tensor(V.reshape(b * h * S, d), [BLOCK_M, BLOCK_K])
    O_desc = TensorDescriptor.from_tensor(O.reshape(b * h * S, d), [BLOCK_M, BLOCK_K])
    dO_desc = TensorDescriptor.from_tensor(dO.reshape(b * h * S, d), [BLOCK_M, BLOCK_K])
    
    dQ_desc = TensorDescriptor.from_tensor(dQ.reshape(b * h * S, d), [BLOCK_M, BLOCK_K])
    dK_desc = TensorDescriptor.from_tensor(dK.reshape(b * h * S, d), [BLOCK_M, BLOCK_K])
    dV_desc = TensorDescriptor.from_tensor(dV.reshape(b * h * S, d), [BLOCK_M, BLOCK_K])
    
    num_tiles = triton.cdiv(S, 128)
    grid = (num_tiles, b * h)
    
    mha_bwd_dQ_kernel[grid](Q_desc, K_desc, V_desc, O_desc, dO_desc, L, dQ_desc, S, num_warps=8)
    mha_bwd_dK_kernel[grid](Q_desc, K_desc, V_desc, O_desc, dO_desc, L, dK_desc, S, num_warps=8)
    mha_bwd_dV_kernel[grid](Q_desc, K_desc, V_desc, O_desc, dO_desc, L, dV_desc, S, num_warps=8)