import math
import torch
import triton
import triton.language as tl


@triton.jit
def bwd_dq_kernel(
    Q, K, V, O, dO, L, dQ,
    S_len, D, H,
    BLOCK: tl.constexpr,
    SCALE: tl.constexpr,
):
    i_start = tl.program_id(0) * BLOCK
    pid = tl.program_id(1)
    batch_idx = pid // H
    head_idx = pid % H
    
    row_offsets = tl.arange(0, BLOCK)
    col_offsets = tl.arange(0, BLOCK)
    
    Q_ptr = Q + batch_idx * Q.stride(0) + head_idx * Q.stride(1)
    K_ptr = K + batch_idx * K.stride(0) + head_idx * K.stride(1)
    V_ptr = V + batch_idx * V.stride(0) + head_idx * V.stride(1)
    O_ptr = O + batch_idx * O.stride(0) + head_idx * O.stride(1)
    dO_ptr = dO + batch_idx * dO.stride(0) + head_idx * dO.stride(1)
    dQ_ptr = dQ + batch_idx * dQ.stride(0) + head_idx * dQ.stride(1)
    L_ptr = L + batch_idx * L.stride(0) + head_idx * L.stride(1)
    
    base_addr = Q_ptr
    stride_m = Q.stride(2)
    stride_d = Q.stride(3)
    
    q0 = tl.load(base_addr + (i_start + row_offsets[:, None]) * stride_m + (0 + col_offsets[None, :]) * stride_d,
                 cache_modifier=".ca")
    q1 = tl.load(base_addr + (i_start + row_offsets[:, None]) * stride_m + (64 + col_offsets[None, :]) * stride_d,
                 cache_modifier=".ca")
    
    base_addr = O_ptr
    o0 = tl.load(base_addr + (i_start + row_offsets[:, None]) * stride_m + (0 + col_offsets[None, :]) * stride_d,
                 cache_modifier=".ca")
    o1 = tl.load(base_addr + (i_start + row_offsets[:, None]) * stride_m + (64 + col_offsets[None, :]) * stride_d,
                 cache_modifier=".ca")
                 
    base_addr = dO_ptr
    do0 = tl.load(base_addr + (i_start + row_offsets[:, None]) * stride_m + (0 + col_offsets[None, :]) * stride_d,
                  cache_modifier=".ca")
    do1 = tl.load(base_addr + (i_start + row_offsets[:, None]) * stride_m + (64 + col_offsets[None, :]) * stride_d,
                  cache_modifier=".ca")
    
    D_i = tl.sum(o0 * do0, axis=1) + tl.sum(o1 * do1, axis=1)
    D_i = tl.reshape(D_i, (BLOCK, 1))
    
    l_val = tl.load(L_ptr + i_start + row_offsets)
    
    dQ0_acc = tl.zeros((BLOCK, BLOCK), tl.float32)
    dQ1_acc = tl.zeros((BLOCK, BLOCK), tl.float32)
    
    for j_start in range(0, S_len, BLOCK):
        
        base_addr = K_ptr
        k0 = tl.load(base_addr + (j_start + row_offsets[:, None]) * stride_m + (0 + col_offsets[None, :]) * stride_d,
                     cache_modifier=".ca")
        k1 = tl.load(base_addr + (j_start + row_offsets[:, None]) * stride_m + (64 + col_offsets[None, :]) * stride_d,
                     cache_modifier=".ca")
                     
        base_addr = V_ptr
        v0 = tl.load(base_addr + (j_start + row_offsets[:, None]) * stride_m + (0 + col_offsets[None, :]) * stride_d,
                     cache_modifier=".ca")
        v1 = tl.load(base_addr + (j_start + row_offsets[:, None]) * stride_m + (64 + col_offsets[None, :]) * stride_d,
                     cache_modifier=".ca")
        
        S = tl.dot(q0, k0.T) + tl.dot(q1, k1.T)
        P = math.exp(S * SCALE - l_val[:, None])
        
        dP = tl.dot(do0, v0.T) + tl.dot(do1, v1.T)
        dS = P * (dP - D_i) * SCALE
        
        dQ0_acc = tl.dot(dS, k0, dQ0_acc)
        dQ1_acc = tl.dot(dS, k1, dQ1_acc)
        
    base_addr = dQ_ptr
    tl.store(base_addr + (i_start + row_offsets[:, None]) * stride_m + (0 + col_offsets[None, :]) * stride_d, dQ0_acc.to(torch.bfloat16))
    tl.store(base_addr + (i_start + row_offsets[:, None]) * stride_m + (64 + col_offsets[None, :]) * stride_d, dQ1_acc.to(torch.bfloat16))


@triton.jit
def bwd_dkv_kernel(
    Q, K, V, O, dO, L, dK, dV,
    S_len, D, H,
    BLOCK: tl.constexpr,
    SCALE: tl.constexpr,
):
    j_start = tl.program_id(0) * BLOCK
    pid = tl.program_id(1)
    batch_idx = pid // H
    head_idx = pid % H
    
    row_offsets = tl.arange(0, BLOCK)
    col_offsets = tl.arange(0, BLOCK)
    
    Q_ptr = Q + batch_idx * Q.stride(0) + head_idx * Q.stride(1)
    K_ptr = K + batch_idx * K.stride(0) + head_idx * K.stride(1)
    V_ptr = V + batch_idx * V.stride(0) + head_idx * V.stride(1)
    O_ptr = O + batch_idx * O.stride(0) + head_idx * O.stride(1)
    dO_ptr = dO + batch_idx * dO.stride(0) + head_idx * dO.stride(1)
    dK_ptr = dK + batch_idx * dK.stride(0) + head_idx * dK.stride(1)
    dV_ptr = dV + batch_idx * dV.stride(0) + head_idx * dV.stride(1)
    L_ptr = L + batch_idx * L.stride(0) + head_idx * L.stride(1)
    
    base_addr = K_ptr
    stride_m = K.stride(2)
    stride_d = K.stride(3)
    
    k0 = tl.load(base_addr + (j_start + row_offsets[:, None]) * stride_m + (0 + col_offsets[None, :]) * stride_d,
                 cache_modifier=".ca")
    k1 = tl.load(base_addr + (j_start + row_offsets[:, None]) * stride_m + (64 + col_offsets[None, :]) * stride_d,
                 cache_modifier=".ca")
                 
    base_addr = V_ptr
    v0 = tl.load(base_addr + (j_start + row_offsets[:, None]) * stride_m + (0 + col_offsets[None, :]) * stride_d,
                 cache_modifier=".ca")
    v1 = tl.load(base_addr + (j_start + row_offsets[:, None]) * stride_m + (64 + col_offsets[None, :]) * stride_d,
                 cache_modifier=".ca")
    
    dK0_acc = tl.zeros((BLOCK, BLOCK), tl.float32)
    dK1_acc = tl.zeros((BLOCK, BLOCK), tl.float32)
    dV0_acc = tl.zeros((BLOCK, BLOCK), tl.float32)
    dV1_acc = tl.zeros((BLOCK, BLOCK), tl.float32)
    
    for i_start in range(0, S_len, BLOCK):
        
        base_addr = Q_ptr
        q0 = tl.load(base_addr + (i_start + row_offsets[:, None]) * stride_m + (0 + col_offsets[None, :]) * stride_d,
                     cache_modifier=".ca")
        q1 = tl.load(base_addr + (i_start + row_offsets[:, None]) * stride_m + (64 + col_offsets[None, :]) * stride_d,
                     cache_modifier=".ca")
        
        base_addr = O_ptr
        o0 = tl.load(base_addr + (i_start + row_offsets[:, None]) * stride_m + (0 + col_offsets[None, :]) * stride_d,
                     cache_modifier=".ca")
        o1 = tl.load(base_addr + (i_start + row_offsets[:, None]) * stride_m + (64 + col_offsets[None, :]) * stride_d,
                     cache_modifier=".ca")
        
        base_addr = dO_ptr
        do0 = tl.load(base_addr + (i_start + row_offsets[:, None]) * stride_m + (0 + col_offsets[None, :]) * stride_d,
                      cache_modifier=".ca")
        do1 = tl.load(base_addr + (i_start + row_offsets[:, None]) * stride_m + (64 + col_offsets[None, :]) * stride_d,
                      cache_modifier=".ca")
        
        D_i = tl.sum(o0 * do0, axis=1) + tl.sum(o1 * do1, axis=1)
        D_i = tl.reshape(D_i, (BLOCK, 1))
        
        l_val = tl.load(L_ptr + i_start + row_offsets)
        
        S = tl.dot(q0, k0.T) + tl.dot(q1, k1.T)
        P = math.exp(S * SCALE - l_val[:, None])
        
        dP = tl.dot(do0, v0.T) + tl.dot(do1, v1.T)
        dS = P * (dP - D_i) * SCALE
        
        dK0_acc = tl.dot(dS.T, q0, dK0_acc)
        dK1_acc = tl.dot(dS.T, q1, dK1_acc)
        
        dV0_acc = tl.dot(P.T, do0, dV0_acc)
        dV1_acc = tl.dot(P.T, do1, dV1_acc)
        
    base_addr = dK_ptr
    tl.store(base_addr + (j_start + row_offsets[:, None]) * stride_m + (0 + col_offsets[None, :]) * stride_d, dK0_acc.to(torch.bfloat16))
    tl.store(base_addr + (j_start + row_offsets[:, None]) * stride_m + (64 + col_offsets[None, :]) * stride_d, dK1_acc.to(torch.bfloat16))
    
    base_addr = dV_ptr
    tl.store(base_addr + (j_start + row_offsets[:, None]) * stride_m + (0 + col_offsets[None, :]) * stride_d, dV0_acc.to(torch.bfloat16))
    tl.store(base_addr + (j_start + row_offsets[:, None]) * stride_m + (64 + col_offsets[None, :]) * stride_d, dV1_acc.to(torch.bfloat16))


BLOCK = 64
NUM_WARPS = 4
NUM_STAGES = 2

def run(Q, K, V, O, dO, L, dQ, dK, dV):
    b, h, s, d = Q.shape
    grid = (s // BLOCK, b * h)
    
    scale = 1.0 / math.sqrt(d)
    
    bwd_dq_kernel[grid](
        Q, K, V, O, dO, L, dQ,
        s, d, h,
        BLOCK=BLOCK,
        SCALE=scale,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    
    bwd_dkv_kernel[grid](
        Q, K, V, O, dO, L, dK, dV,
        s, d, h,
        BLOCK=BLOCK,
        SCALE=scale,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )