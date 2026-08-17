import torch
import triton
import triton.language as tl
import math


def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


triton.set_allocator(alloc_fn)


@triton.jit
def mha_bwd_dQ_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L, dQ_ptr, S,
    b_h: tl.constexpr, scale: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    bh = tl.program_id(1)
    tile = tl.program_id(0)
    
    s_len = 128
    s_start = tile * s_len
    
    num_tiles = tl.cdiv(S, s_len)
    
    desc_Q = tl.make_tensor_descriptor(
        Q_ptr, shape=[b_h, S, 128], strides=[S * 128, 128, 1],
        block_shape=[1, BLOCK_M, BLOCK_N], padding_option="zero")
    desc_K = tl.make_tensor_descriptor(
        K_ptr, shape=[b_h, S, 128], strides=[S * 128, 128, 1],
        block_shape=[1, BLOCK_M, BLOCK_N], padding_option="zero")
    desc_V = tl.make_tensor_descriptor(
        V_ptr, shape=[b_h, S, 128], strides=[S * 128, 128, 1],
        block_shape=[1, BLOCK_M, BLOCK_N], padding_option="zero")
    desc_O = tl.make_tensor_descriptor(
        O_ptr, shape=[b_h, S, 128], strides=[S * 128, 128, 1],
        block_shape=[1, BLOCK_M, BLOCK_N], padding_option="zero")
    desc_dO = tl.make_tensor_descriptor(
        dO_ptr, shape=[b_h, S, 128], strides=[S * 128, 128, 1],
        block_shape=[1, BLOCK_M, BLOCK_N], padding_option="zero")
    
    desc_dQ = tl.make_tensor_descriptor(
        dQ_ptr, shape=[b_h, S, 128], strides=[S * 128, 128, 1],
        block_shape=[1, BLOCK_M, BLOCK_N], padding_option="zero")
    
    L_bh = L + bh * S
    
    r_idx = tl.arange(0, BLOCK_M)
    c_idx = tl.arange(0, BLOCK_N)
    
    Q_0 = desc_Q.load([bh, s_start, 0]).to(tl.float32)
    Q_1 = desc_Q.load([bh, s_start, 64]).to(tl.float32)
    O_0 = desc_O.load([bh, s_start, 0]).to(tl.float32)
    O_1 = desc_O.load([bh, s_start, 64]).to(tl.float32)
    dO_0 = desc_dO.load([bh, s_start, 0]).to(tl.float32)
    dO_1 = desc_dO.load([bh, s_start, 64]).to(tl.float32)
    
    dQ_0 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    dQ_1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    s_end = s_start + 128
    
    for k_tile in range(num_tiles):
        k_start = k_tile * 128
        
        if min(s_end, S) - k_start <= 0:
            break
        
        K_0 = desc_K.load([bh, k_start, 0]).to(tl.float32)
        K_1 = desc_K.load([bh, k_start, 64]).to(tl.float32)
        V_0 = desc_V.load([bh, k_start, 0]).to(tl.float32)
        V_1 = desc_V.load([bh, k_start, 64]).to(tl.float32)
        
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
        
    desc_dQ.store([bh, s_start, 0], dQ_0.to(tl.bfloat16))
    desc_dQ.store([bh, s_start, 64], dQ_1.to(tl.bfloat16))


@triton.jit
def mha_bwd_dK_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L, dK_ptr, dV_ptr, S,
    b_h: tl.constexpr, scale: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    bh = tl.program_id(1)
    tile = tl.program_id(0)
    
    s_len = 128
    k_start = tile * s_len
    
    num_tiles = tl.cdiv(S, s_len)
    
    desc_Q = tl.make_tensor_descriptor(
        Q_ptr, shape=[b_h, S, 128], strides=[S * 128, 128, 1],
        block_shape=[1, BLOCK_M, BLOCK_N], padding_option="zero")
    desc_K = tl.make_tensor_descriptor(
        K_ptr, shape=[b_h, S, 128], strides=[S * 128, 128, 1],
        block_shape=[1, BLOCK_M, BLOCK_N], padding_option="zero")
    desc_V = tl.make_tensor_descriptor(
        V_ptr, shape=[b_h, S, 128], strides=[S * 128, 128, 1],
        block_shape=[1, BLOCK_M, BLOCK_N], padding_option="zero")
    desc_O = tl.make_tensor_descriptor(
        O_ptr, shape=[b_h, S, 128], strides=[S * 128, 128, 1],
        block_shape=[1, BLOCK_M, BLOCK_N], padding_option="zero")
    desc_dO = tl.make_tensor_descriptor(
        dO_ptr, shape=[b_h, S, 128], strides=[S * 128, 128, 1],
        block_shape=[1, BLOCK_M, BLOCK_N], padding_option="zero")
    
    desc_dK = tl.make_tensor_descriptor(
        dK_ptr, shape=[b_h, S, 128], strides=[S * 128, 128, 1],
        block_shape=[1, BLOCK_M, BLOCK_N], padding_option="zero")
    desc_dV = tl.make_tensor_descriptor(
        dV_ptr, shape=[b_h, S, 128], strides=[S * 128, 128, 1],
        block_shape=[1, BLOCK_M, BLOCK_N], padding_option="zero")
    
    L_bh = L + bh * S
    
    r_idx = tl.arange(0, BLOCK_M)
    c_idx = tl.arange(0, BLOCK_N)
    
    dK_0 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    dK_1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    dV_0 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    dV_1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    K_0 = desc_K.load([bh, k_start, 0]).to(tl.float32)
    K_1 = desc_K.load([bh, k_start, 64]).to(tl.float32)
    V_0 = desc_V.load([bh, k_start, 0]).to(tl.float32)
    V_1 = desc_V.load([bh, k_start, 64]).to(tl.float32)
    
    for q_tile in range(tile, num_tiles):
        q_start = q_tile * s_len
        
        Q_0 = desc_Q.load([bh, q_start, 0]).to(tl.float32)
        Q_1 = desc_Q.load([bh, q_start, 64]).to(tl.float32)
        O_0 = desc_O.load([bh, q_start, 0]).to(tl.float32)
        O_1 = desc_O.load([bh, q_start, 64]).to(tl.float32)
        dO_0 = desc_dO.load([bh, q_start, 0]).to(tl.float32)
        dO_1 = desc_dO.load([bh, q_start, 64]).to(tl.float32)
        
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
        
        P_h_T = P_h.T
        dV_0 += tl.dot(P_h_T, dO_0)
        dV_1 += tl.dot(P_h_T, dO_1)
        
    desc_dK.store([bh, k_start, 0], dK_0.to(tl.bfloat16))
    desc_dK.store([bh, k_start, 64], dK_1.to(tl.bfloat16))
    
    desc_dV.store([bh, k_start, 0], dV_0.to(tl.bfloat16))
    desc_dV.store([bh, k_start, 64], dV_1.to(tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    b, h, s, d = Q.shape
    S = s
    
    BLOCK_M = 128
    BLOCK_N = 128
    
    b_h = b * h
    num_tiles = triton.cdiv(S, 128)
    grid = (num_tiles, b_h)
    
    scale_val = 1.0 / math.sqrt(128)
    
    mha_bwd_dQ_kernel[grid](
        Q, K, V, O, dO, L, dQ, S,
        b_h, scale_val, BLOCK_M, BLOCK_N, num_warps=4
    )
    mha_bwd_dK_kernel[grid](
        Q, K, V, O, dO, L, dK, dV, S,
        b_h, scale_val, BLOCK_M, BLOCK_N, num_warps=4
    )