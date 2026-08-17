import torch
import triton
import triton.language as tl
import math


@triton.jit
def load_tile(base_ptr, b_h, row_offset, col_offset, S_len):
    row_idx = b_h * S_len + row_offset
    ptr = base_ptr + row_idx * 128 + col_offset
    val = tl.load(ptr + tl.arange(0, 128)[:, None] * 128 + tl.arange(0, 64)[None, :],
                  mask=(row_offset + tl.arange(0, 128))[:, None] < S_len, other=0.0)
    return val


@triton.jit
def store_tile(base_ptr, b_h, row_offset, col_offset, val, S_len):
    row_idx = b_h * S_len + row_offset
    ptr = base_ptr + row_idx * 128 + col_offset
    tl.store(ptr + tl.arange(0, 128)[:, None] * 128 + tl.arange(0, 64)[None, :],
             val.to(tl.bfloat16),
             mask=(row_offset + tl.arange(0, 128))[:, None] < S_len)


@triton.jit
def load_l(L_ptr, b_h, offset, length, S_len):
    base_offset = b_h * S_len + offset
    l = tl.load(L_ptr + base_offset + tl.arange(0, length), mask=(offset + tl.arange(0, length)) < S_len, other=0.0)
    return l


@triton.jit
def bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, dQ_ptr,
    S_len, sqrt_d,
):
    q_tile = tl.program_id(0)
    b_h = tl.program_id(1)
    
    q_offset = q_tile * 128
    if q_offset >= S_len:
        return
    
    q0 = load_tile(Q_ptr, b_h, q_offset, 0, S_len)
    q1 = load_tile(Q_ptr, b_h, q_offset, 64, S_len)
    do0 = load_tile(dO_ptr, b_h, q_offset, 0, S_len)
    do1 = load_tile(dO_ptr, b_h, q_offset, 64, S_len)
    
    l_q = load_l(L_ptr, b_h, q_offset, 128, S_len)
    
    acc_dQ0 = tl.zeros((128, 64), tl.float32)
    acc_dQ1 = tl.zeros((128, 64), tl.float32)
    
    num_k_tiles = tl.cdiv(S_len, 128)
    
    for k_tile in range(num_k_tiles):
        k_offset = k_tile * 128
        if k_offset >= S_len:
            break
            
        k0 = load_tile(K_ptr, b_h, k_offset, 0, S_len)
        k1 = load_tile(K_ptr, b_h, k_offset, 64, S_len)
        v0 = load_tile(V_ptr, b_h, k_offset, 0, S_len)
        v1 = load_tile(V_ptr, b_h, k_offset, 64, S_len)
        
        s = tl.dot(q0, k0) + tl.dot(q1, k1)
        dp = tl.dot(do0, v0) + tl.dot(do1, v1)
        
        p = tl.exp(s * sqrt_d - l_q[:, None])
        
        valid_q = (q_offset + tl.arange(0, 128))[:, None] < S_len
        valid_k = (k_offset + tl.arange(0, 128))[None, :] < S_len
        p = tl.where(valid_q & valid_k, p, 0.0)
        
        ds = dp * p
        
        acc_dQ0 += tl.dot(ds, k0.T)
        acc_dQ1 += tl.dot(ds, k1.T)
        
    store_tile(dQ_ptr, b_h, q_offset, 0, acc_dQ0 * sqrt_d, S_len)
    store_tile(dQ_ptr, b_h, q_offset, 64, acc_dQ1 * sqrt_d, S_len)


@triton.jit
def bwd_dk_dv_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, dK_ptr, dV_ptr,
    S_len, sqrt_d,
):
    k_tile = tl.program_id(0)
    b_h = tl.program_id(1)
    
    k_offset = k_tile * 128
    if k_offset >= S_len:
        return
        
    k0 = load_tile(K_ptr, b_h, k_offset, 0, S_len)
    k1 = load_tile(K_ptr, b_h, k_offset, 64, S_len)
    v0 = load_tile(V_ptr, b_h, k_offset, 0, S_len)
    v1 = load_tile(V_ptr, b_h, k_offset, 64, S_len)
    
    acc_dK0 = tl.zeros((128, 64), tl.float32)
    acc_dK1 = tl.zeros((128, 64), tl.float32)
    acc_dV0 = tl.zeros((128, 64), tl.float32)
    acc_dV1 = tl.zeros((128, 64), tl.float32)
    
    num_q_tiles = tl.cdiv(S_len, 128)
    
    for q_tile in range(num_q_tiles):
        q_offset = q_tile * 128
        if q_offset >= S_len:
            break
            
        q0 = load_tile(Q_ptr, b_h, q_offset, 0, S_len)
        q1 = load_tile(Q_ptr, b_h, q_offset, 64, S_len)
        do0 = load_tile(dO_ptr, b_h, q_offset, 0, S_len)
        do1 = load_tile(dO_ptr, b_h, q_offset, 64, S_len)
        
        l_q = load_l(L_ptr, b_h, q_offset, 128, S_len)
        
        s = tl.dot(q0, k0) + tl.dot(q1, k1)
        dp = tl.dot(do0, v0) + tl.dot(do1, v1)
        
        p = tl.exp(s * sqrt_d - l_q[:, None])
        
        valid_q = (q_offset + tl.arange(0, 128))[:, None] < S_len
        valid_k = (k_offset + tl.arange(0, 128))[None, :] < S_len
        p = tl.where(valid_q & valid_k, p, 0.0)
        
        ds = dp * p
        
        acc_dK0 += tl.dot(ds.T, q0.T)
        acc_dK1 += tl.dot(ds.T, q1.T)
        
        acc_dV0 += tl.dot(p.T, do0.T)
        acc_dV1 += tl.dot(p.T, do1.T)
            
    store_tile(dK_ptr, b_h, k_offset, 0, acc_dK0 * sqrt_d, S_len)
    store_tile(dK_ptr, b_h, k_offset, 64, acc_dK1 * sqrt_d, S_len)
    store_tile(dV_ptr, b_h, k_offset, 0, acc_dV0, S_len)
    store_tile(dV_ptr, b_h, k_offset, 64, acc_dV1, S_len)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S_len, d = Q.shape
    sqrt_d = 1.0 / math.sqrt(d)
    
    Q_ptr = Q.flatten(0, 2)
    K_ptr = K.flatten(0, 2)
    V_ptr = V.flatten(0, 2)
    dO_ptr = dO.flatten(0, 2)
    L_ptr = L.flatten(0, 1)
    dQ_ptr = dQ.flatten(0, 2)
    dK_ptr = dK.flatten(0, 2)
    dV_ptr = dV.flatten(0, 2)
    
    grid_dq = (triton.cdiv(S_len, 128), B * H)
    grid_dk_dv = (triton.cdiv(S_len, 128), B * H)

    bwd_dq_kernel[grid_dq](
        Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, dQ_ptr,
        S_len, sqrt_d,
        num_warps=8, num_stages=3)
    
    bwd_dk_dv_kernel[grid_dk_dv](
        Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, dK_ptr, dV_ptr,
        S_len, sqrt_d,
        num_warps=8, num_stages=3)