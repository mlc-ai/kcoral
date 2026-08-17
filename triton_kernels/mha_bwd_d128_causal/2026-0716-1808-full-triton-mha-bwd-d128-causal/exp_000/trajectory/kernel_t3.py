import torch
import triton
import triton.language as tl
import math
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def mha_bwd_kernel(
    Q_desc, K_desc, V_desc, O_desc, dO_desc, L,
    dQ_desc, dK_desc, dV_desc,
    S, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr
):
    bh = tl.program_id(0)
    tile = tl.program_id(1)
    
    s_len = 128
    s_start = tile * s_len
    s_end = s_start + s_len
    
    num_tiles = tl.cdiv(S, s_len)
    
    batch_off = bh * S
    L_bh = L + bh * S
    
    scale = 1.0 / math.sqrt(128)
    
    dQ_0 = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    dQ_1 = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    
    r_idx = tl.arange(0, BLOCK_M)
    c_idx = tl.arange(0, BLOCK_M)
    
    Q_0 = Q_desc.load([batch_off + s_start, 0])
    Q_1 = Q_desc.load([batch_off + s_start, 64])
    O_0 = O_desc.load([batch_off + s_start, 0])
    O_1 = O_desc.load([batch_off + s_start, 64])
    dO_0 = dO_desc.load([batch_off + s_start, 0])
    dO_1 = dO_desc.load([batch_off + s_start, 64])
    
    for k_tile in range(0, tile + 1):
        k_start = k_tile * s_len
        
        K_0 = K_desc.load([batch_off + k_start, 0])
        K_1 = K_desc.load([batch_off + k_start, 64])
        V_0 = V_desc.load([batch_off + k_start, 0])
        V_1 = V_desc.load([batch_off + k_start, 64])
        
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
        
    dQ_desc.store([batch_off + s_start, 0], dQ_0)
    dQ_desc.store([batch_off + s_start, 64], dQ_1)
    
    dK_0 = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    dK_1 = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    dV_0 = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    dV_1 = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    
    K_0 = K_desc.load([batch_off + s_start, 0])
    K_1 = K_desc.load([batch_off + s_start, 64])
    V_0 = V_desc.load([batch_off + s_start, 0])
    V_1 = V_desc.load([batch_off + s_start, 64])
    
    for q_tile in range(tile, num_tiles):
        q_start = q_tile * s_len
        
        Q_0 = Q_desc.load([batch_off + q_start, 0])
        Q_1 = Q_desc.load([batch_off + q_start, 64])
        O_0 = O_desc.load([batch_off + q_start, 0])
        O_1 = O_desc.load([batch_off + q_start, 64])
        dO_0 = dO_desc.load([batch_off + q_start, 0])
        dO_1 = dO_desc.load([batch_off + q_start, 64])
        
        S_h = (tl.dot(Q_0, K_0.T) + tl.dot(Q_1, K_1.T)) * scale
        
        L_h = tl.load(L_bh + q_start + r_idx, mask=(q_start + r_idx < S), other=0.0)
        P_h = tl.math.exp(S_h - L_h[:, None])
        
        global_r = q_start + r_idx[:, None]
        global_c = s_start + c_idx[None, :]
        mask_causal = global_r >= global_c
        mask_h = mask_causal & (global_r < S) & (global_c < S)
        P_h = P_h * mask_h
        
        dP_h = tl.dot(dO_0, V_0.T) + tl.dot(dO_1, V_1.T)
        
        O_sum_exp_h = (O_0 * dO_0 + O_1 * dO_1).sum(axis=1)[:, None]
        D_S_h = P_h * (dP_h - O_sum_exp_h)
        
        dK_0 += tl.dot(D_S_h.T, Q_0) * scale
        dK_1 += tl.dot(D_S_h.T, Q_1) * scale
        
        dV_0 += tl.dot(P_h.T, dO_0)
        dV_1 += tl.dot(P_h.T, dO_1)
        
    dK_desc.store([batch_off + s_start, 0], dK_0)
    dK_desc.store([batch_off + s_start, 64], dK_1)
    
    dV_desc.store([batch_off + s_start, 0], dV_0)
    dV_desc.store([batch_off + s_start, 64], dV_1)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    b, h, s, d = Q.shape
    S = s
    
    BLOCK_M = 128
    BLOCK_K = 64
    
    Q_desc = TensorDescriptor.from_tensor(Q.reshape(-1, d), [BLOCK_M, BLOCK_K])
    K_desc = TensorDescriptor.from_tensor(K.reshape(-1, d), [BLOCK_M, BLOCK_K])
    V_desc = TensorDescriptor.from_tensor(V.reshape(-1, d), [BLOCK_M, BLOCK_K])
    O_desc = TensorDescriptor.from_tensor(O.reshape(-1, d), [BLOCK_M, BLOCK_K])
    dO_desc = TensorDescriptor.from_tensor(dO.reshape(-1, d), [BLOCK_M, BLOCK_K])
    
    dQ_desc = TensorDescriptor.from_tensor(dQ.reshape(-1, d), [BLOCK_M, BLOCK_K])
    dK_desc = TensorDescriptor.from_tensor(dK.reshape(-1, d), [BLOCK_M, BLOCK_K])
    dV_desc = TensorDescriptor.from_tensor(dV.reshape(-1, d), [BLOCK_M, BLOCK_K])
    
    grid = (b * h, triton.cdiv(S, 128))
    
    mha_bwd_kernel[grid](
        Q_desc, K_desc, V_desc, O_desc, dO_desc, L,
        dQ_desc, dK_desc, dV_desc,
        S, BLOCK_M, BLOCK_K, num_warps=8
    )