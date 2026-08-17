import torch
import triton
import triton.language as tl
import math
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def mha_bwd_dQ_kernel(
    Q_desc, K_desc, V_desc, O_desc, dO_desc, L, dQ_desc, S, scale
):
    bh = tl.program_id(1)
    tile = tl.program_id(0)
    
    s_len = 128
    s_start = tile * s_len
    
    r_idx = tl.arange(0, 128)
    c_idx = tl.arange(0, 64)
    
    Q_0 = Q_desc.load([bh * S + s_start, 0]).to(tl.float32)
    Q_1 = Q_desc.load([bh * S + s_start, 64]).to(tl.float32)
    O_0 = O_desc.load([bh * S + s_start, 0]).to(tl.float32)
    O_1 = O_desc.load([bh * S + s_start, 64]).to(tl.float32)
    dO_0 = dO_desc.load([bh * S + s_start, 0]).to(tl.float32)
    dO_1 = dO_desc.load([bh * S + s_start, 64]).to(tl.float32)
    
    dQ_0 = tl.zeros((128, 64), dtype=tl.float32)
    dQ_1 = tl.zeros((128, 64), dtype=tl.float32)
    
    s_end_mask = s_start + 128
    if s_end_mask > S:
        s_end_mask = S
        
    for k_tile in range(tile + 1):
        k_start = k_tile * 128
        if k_start > s_end_mask:
            break
            
        K_0 = K_desc.load([bh * S + k_start, 0]).to(tl.float32)
        K_1 = K_desc.load([bh * S + k_start, 64]).to(tl.float32)
        V_0 = V_desc.load([bh * S + k_start, 0]).to(tl.float32)
        V_1 = V_desc.load([bh * S + k_start, 64]).to(tl.float32)
        
        S_h = (tl.dot(Q_0, K_0.T) + tl.dot(Q_1, K_1.T)) * scale
        
        L_h = tl.load(L + bh * S + s_start + r_idx, mask=(s_start + r_idx < S), other=0.0)
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
    Q_desc, K_desc, V_desc, O_desc, dO_desc, L, dK_desc, S, scale
):
    bh = tl.program_id(1)
    tile = tl.program_id(0)
    
    s_len = 128
    k_start = tile * s_len
    
    num_tiles = tl.cdiv(S, s_len)
    
    r_idx = tl.arange(0, 128)
    c_idx = tl.arange(0, 64)
    
    dK_0 = tl.zeros((128, 64), dtype=tl.float32)
    dK_1 = tl.zeros((128, 64), dtype=tl.float32)
    
    for q_tile in range(tile, num_tiles):
        q_start = q_tile * 128
        
        Q_0 = Q_desc.load([bh * S + q_start, 0]).to(tl.float32)
        Q_1 = Q_desc.load([bh * S + q_start, 64]).to(tl.float32)
        K_0 = K_desc.load([bh * S + k_start, 0]).to(tl.float32)
        K_1 = K_desc.load([bh * S + k_start, 64]).to(tl.float32)
        V_0 = V_desc.load([bh * S + k_start, 0]).to(tl.float32)
        V_1 = V_desc.load([bh * S + k_start, 64]).to(tl.float32)
        O_0 = O_desc.load([bh * S + q_start, 0]).to(tl.float32)
        O_1 = O_desc.load([bh * S + q_start, 64]).to(tl.float32)
        dO_0 = dO_desc.load([bh * S + q_start, 0]).to(tl.float32)
        dO_1 = dO_desc.load([bh * S + q_start, 64]).to(tl.float32)
        
        S_h = (tl.dot(Q_0, K_0.T) + tl.dot(Q_1, K_1.T)) * scale
        
        L_h = tl.load(L + bh * S + q_start + r_idx, mask=(q_start + r_idx < S), other=0.0)
        P_h = tl.math.exp(S_h - L_h[:, None])
        
        global_r = q_start + r_idx[:, None]
        global_c = k_start + c_idx[None, :]
        mask_causal = global_r >= global_c
        mask_h = mask_causal & (global_r < S) & (global_c < S)
        P_h = P_h * mask_h
        
        dP_h = tl.dot(dO_0, V_0.T) + tl.dot(dO_1, V_1.T)
        
        O_sum_exp_h = (O_0 * dO_0 + O_1 * dO_1).sum(axis=1)[:, None]
        D_S_h = P_h * (dP_h - O_sum_exp_h)
        
        dK_0 += tl.dot(D_S_h.T, Q_0) * scale
        dK_1 += tl.dot(D_S_h.T, Q_1) * scale
        
    dK_desc.store([bh * S + k_start, 0], dK_0.to(tl.bfloat16))
    dK_desc.store([bh * S + k_start, 64], dK_1.to(tl.bfloat16))


@triton.jit
def mha_bwd_dV_kernel(
    Q_desc, K_desc, V_desc, O_desc, dO_desc, L, dV_desc, S, scale
):
    bh = tl.program_id(1)
    tile = tl.program_id(0)
    
    s_len = 128
    k_start = tile * s_len
    
    num_tiles = tl.cdiv(S, s_len)
    
    r_idx = tl.arange(0, 128)
    c_idx = tl.arange(0, 64)
    
    dV_0 = tl.zeros((128, 64), dtype=tl.float32)
    dV_1 = tl.zeros((128, 64), dtype=tl.float32)
    
    for q_tile in range(tile, num_tiles):
        q_start = q_tile * 128
        
        Q_0 = Q_desc.load([bh * S + q_start, 0]).to(tl.float32)
        Q_1 = Q_desc.load([bh * S + q_start, 64]).to(tl.float32)
        K_0 = K_desc.load([bh * S + k_start, 0]).to(tl.float32)
        K_1 = K_desc.load([bh * S + k_start, 64]).to(tl.float32)
        V_0 = V_desc.load([bh * S + k_start, 0]).to(tl.float32)
        V_1 = V_desc.load([bh * S + k_start, 64]).to(tl.float32)
        O_0 = O_desc.load([bh * S + q_start, 0]).to(tl.float32)
        O_1 = O_desc.load([bh * S + q_start, 64]).to(tl.float32)
        dO_0 = dO_desc.load([bh * S + q_start, 0]).to(tl.float32)
        dO_1 = dO_desc.load([bh * S + q_start, 64]).to(tl.float32)
        
        S_h = (tl.dot(Q_0, K_0.T) + tl.dot(Q_1, K_1.T)) * scale
        
        L_h = tl.load(L + bh * S + q_start + r_idx, mask=(q_start + r_idx < S), other=0.0)
        P_h = tl.math.exp(S_h - L_h[:, None])
        
        global_r = q_start + r_idx[:, None]
        global_c = k_start + c_idx[None, :]
        mask_causal = global_r >= global_c
        mask_h = mask_causal & (global_r < S) & (global_c < S)
        P_h = P_h * mask_h
        
        dP_h = tl.dot(dO_0, V_0.T) + tl.dot(dO_1, V_1.T)
        
        O_sum_exp_h = (O_0 * dO_0 + O_1 * dO_1).sum(axis=1)[:, None]
        D_S_h = P_h * (dP_h - O_sum_exp_h)
        
        dV_0 += tl.dot(P_h.T, dO_0)
        dV_1 += tl.dot(P_h.T, dO_1)
        
    dV_desc.store([bh * S + k_start, 0], dV_0.to(tl.bfloat16))
    dV_desc.store([bh * S + k_start, 64], dV_1.to(tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    b, h, s, d = Q.shape
    S = s
    
    BLOCK_M = 128
    BLOCK_N = 64
    
    Q_desc = TensorDescriptor.from_tensor(Q.reshape(b * h * S, d), [BLOCK_M, BLOCK_N])
    K_desc = TensorDescriptor.from_tensor(K.reshape(b * h * S, d), [BLOCK_M, BLOCK_N])
    V_desc = TensorDescriptor.from_tensor(V.reshape(b * h * S, d), [BLOCK_M, BLOCK_N])
    O_desc = TensorDescriptor.from_tensor(O.reshape(b * h * S, d), [BLOCK_M, BLOCK_N])
    dO_desc = TensorDescriptor.from_tensor(dO.reshape(b * h * S, d), [BLOCK_M, BLOCK_N])
    
    dQ_desc = TensorDescriptor.from_tensor(dQ.reshape(b * h * S, d), [BLOCK_M, BLOCK_N])
    dK_desc = TensorDescriptor.from_tensor(dK.reshape(b * h * S, d), [BLOCK_M, BLOCK_N])
    dV_desc = TensorDescriptor.from_tensor(dV.reshape(b * h * S, d), [BLOCK_M, BLOCK_N])
    
    num_tiles = triton.cdiv(S, 128)
    grid = (num_tiles, b * h)
    
    scale_val = 1.0 / math.sqrt(128)
    
    mha_bwd_dQ_kernel[grid](Q_desc, K_desc, V_desc, O_desc, dO_desc, L, dQ_desc, S, scale_val, num_warps=4)
    mha_bwd_dK_kernel[grid](Q_desc, K_desc, V_desc, O_desc, dO_desc, L, dK_desc, S, scale_val, num_warps=4)
    mha_bwd_dV_kernel[grid](Q_desc, K_desc, V_desc, O_desc, dO_desc, L, dV_desc, S, scale_val, num_warps=4)