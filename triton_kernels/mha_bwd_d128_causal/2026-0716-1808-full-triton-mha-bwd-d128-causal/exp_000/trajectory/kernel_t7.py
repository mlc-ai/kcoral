import torch
import triton
import triton.language as tl
import math
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def mha_bwd_dQ_kernel(
    desc_Q, desc_K, desc_V, desc_O, desc_dO, L, desc_dQ, S, scale
):
    bh = tl.program_id(1)
    tile = tl.program_id(0)
    
    s_len = 128
    s_start = tile * s_len
    
    r_idx = tl.arange(0, 128)
    c_idx = tl.arange(0, 128)
    
    Q_0 = desc_Q.load([bh, s_start, 0]).to(tl.float32)
    O_0 = desc_O.load([bh, s_start, 0]).to(tl.float32)
    dO_0 = desc_dO.load([bh, s_start, 0]).to(tl.float32)
    
    dQ_0 = tl.zeros((1, 128, 128), dtype=tl.float32)
    
    for k_tile in range(tile + 1):
        k_start = k_tile * s_len
        
        K_0 = desc_K.load([bh, k_start, 0]).to(tl.float32)
        V_0 = desc_V.load([bh, k_start, 0]).to(tl.float32)
        
        S_h = tl.dot(Q_0, K_0.T) * scale
        
        L_h = tl.load(L + bh * S + s_start + r_idx, mask=(s_start + r_idx < S), other=0.0)
        P_h = tl.math.exp(S_h - L_h[:, None])
        
        global_r = s_start + r_idx[:, None]
        global_c = k_start + c_idx[None, :]
        mask_causal = global_r >= global_c
        mask_h = mask_causal & (global_r < S) & (global_c < S)
        P_h = P_h * mask_h
        
        dP_h = tl.dot(dO_0, V_0.T)
        
        O_sum_exp_h = (O_0 * dO_0).sum(axis=1)[:, None]
        D_S_h = P_h * (dP_h - O_sum_exp_h)
        
        dQ_0 += tl.dot(D_S_h, K_0) * scale
        
    desc_dQ.store([bh, s_start, 0], dQ_0.to(tl.bfloat16))


@triton.jit
def mha_bwd_dK_kernel(
    desc_Q, desc_K, desc_V, desc_O, desc_dO, L, desc_dK, S, scale
):
    bh = tl.program_id(1)
    tile = tl.program_id(0)
    
    s_len = 128
    k_start = tile * s_len
    
    num_tiles = tl.cdiv(S, s_len)
    
    r_idx = tl.arange(0, 128)
    c_idx = tl.arange(0, 128)
    
    dK_0 = tl.zeros((1, 128, 128), dtype=tl.float32)
    
    for q_tile in range(tile, num_tiles):
        q_start = q_tile * s_len
        
        Q_0 = desc_Q.load([bh, q_start, 0]).to(tl.float32)
        K_0 = desc_K.load([bh, k_start, 0]).to(tl.float32)
        V_0 = desc_V.load([bh, k_start, 0]).to(tl.float32)
        O_0 = desc_O.load([bh, q_start, 0]).to(tl.float32)
        dO_0 = desc_dO.load([bh, q_start, 0]).to(tl.float32)
        
        S_h = tl.dot(Q_0, K_0.T) * scale
        
        L_h = tl.load(L + bh * S + q_start + r_idx, mask=(q_start + r_idx < S), other=0.0)
        P_h = tl.math.exp(S_h - L_h[:, None])
        
        global_r = q_start + r_idx[:, None]
        global_c = k_start + c_idx[None, :]
        mask_causal = global_r >= global_c
        mask_h = mask_causal & (global_r < S) & (global_c < S)
        P_h = P_h * mask_h
        
        dP_h = tl.dot(dO_0, V_0.T)
        
        O_sum_exp_h = (O_0 * dO_0).sum(axis=1)[:, None]
        D_S_h = P_h * (dP_h - O_sum_exp_h)
        
        dK_0 += tl.dot(D_S_h.T, Q_0) * scale
        
    desc_dK.store([bh, k_start, 0], dK_0.to(tl.bfloat16))


@triton.jit
def mha_bwd_dV_kernel(
    desc_Q, desc_K, desc_V, desc_O, desc_dO, L, desc_dV, S, scale
):
    bh = tl.program_id(1)
    tile = tl.program_id(0)
    
    s_len = 128
    k_start = tile * s_len
    
    num_tiles = tl.cdiv(S, s_len)
    
    r_idx = tl.arange(0, 128)
    c_idx = tl.arange(0, 128)
    
    dV_0 = tl.zeros((1, 128, 128), dtype=tl.float32)
    
    for q_tile in range(tile, num_tiles):
        q_start = q_tile * s_len
        
        Q_0 = desc_Q.load([bh, q_start, 0]).to(tl.float32)
        K_0 = desc_K.load([bh, k_start, 0]).to(tl.float32)
        V_0 = desc_V.load([bh, k_start, 0]).to(tl.float32)
        O_0 = desc_O.load([bh, q_start, 0]).to(tl.float32)
        dO_0 = desc_dO.load([bh, q_start, 0]).to(tl.float32)
        
        S_h = tl.dot(Q_0, K_0.T) * scale
        
        L_h = tl.load(L + bh * S + q_start + r_idx, mask=(q_start + r_idx < S), other=0.0)
        P_h = tl.math.exp(S_h - L_h[:, None])
        
        global_r = q_start + r_idx[:, None]
        global_c = k_start + c_idx[None, :]
        mask_causal = global_r >= global_c
        mask_h = mask_causal & (global_r < S) & (global_c < S)
        P_h = P_h * mask_h
        
        dP_h = tl.dot(dO_0, V_0.T)
        
        O_sum_exp_h = (O_0 * dO_0).sum(axis=1)[:, None]
        D_S_h = P_h * (dP_h - O_sum_exp_h)
        
        dV_0 += tl.dot(P_h.T, dO_0)
        
    desc_dV.store([bh, k_start, 0], dV_0.to(tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    b, h, s, d = Q.shape
    S = s
    
    BLOCK_M = 128
    BLOCK_N = 128
    
    Q_desc = TensorDescriptor.from_tensor(Q.reshape(b * h, S, d), [1, BLOCK_M, BLOCK_N])
    K_desc = TensorDescriptor.from_tensor(K.reshape(b * h, S, d), [1, BLOCK_M, BLOCK_N])
    V_desc = TensorDescriptor.from_tensor(V.reshape(b * h, S, d), [1, BLOCK_M, BLOCK_N])
    O_desc = TensorDescriptor.from_tensor(O.reshape(b * h, S, d), [1, BLOCK_M, BLOCK_N])
    dO_desc = TensorDescriptor.from_tensor(dO.reshape(b * h, S, d), [1, BLOCK_M, BLOCK_N])
    
    dQ_desc = TensorDescriptor.from_tensor(dQ.reshape(b * h, S, d), [1, BLOCK_M, BLOCK_N])
    dK_desc = TensorDescriptor.from_tensor(dK.reshape(b * h, S, d), [1, BLOCK_M, BLOCK_N])
    dV_desc = TensorDescriptor.from_tensor(dV.reshape(b * h, S, d), [1, BLOCK_M, BLOCK_N])
    
    num_tiles = triton.cdiv(S, 128)
    grid = (num_tiles, b * h)
    
    scale_val = 1.0 / math.sqrt(128)
    
    mha_bwd_dQ_kernel[grid](Q_desc, K_desc, V_desc, O_desc, dO_desc, L, dQ_desc, S, scale_val, num_warps=4)
    mha_bwd_dK_kernel[grid](Q_desc, K_desc, V_desc, O_desc, dO_desc, L, dK_desc, S, scale_val, num_warps=4)
    mha_bwd_dV_kernel[grid](Q_desc, K_desc, V_desc, O_desc, dO_desc, L, dV_desc, S, scale_val, num_warps=4)