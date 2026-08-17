import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
import math


@triton.jit
def bwd_dq_kernel(
    Q_desc, K_desc, V_desc, dO_desc, L_ptr_flat, dQ_desc,
    S_len, sqrt_d,
):
    q_tile = tl.program_id(0)
    b_h = tl.program_id(1)
    
    q_offset = q_tile * 128
    if q_offset >= S_len:
        return
    
    flat_q_offset = b_h * S_len + q_offset
    
    q0_0 = Q_desc.load([flat_q_offset, 0])
    q0_1 = Q_desc.load([flat_q_offset + 64, 0])
    q1_0 = Q_desc.load([flat_q_offset, 64])
    q1_1 = Q_desc.load([flat_q_offset + 64, 64])
    
    do0_0 = dO_desc.load([flat_q_offset, 0])
    do0_1 = dO_desc.load([flat_q_offset + 64, 0])
    do1_0 = dO_desc.load([flat_q_offset, 64])
    do1_1 = dO_desc.load([flat_q_offset + 64, 64])
    
    L_q_0 = tl.load(L_ptr_flat + b_h * S_len + q_offset + tl.arange(0, 64), mask=(q_offset + tl.arange(0, 64)) < S_len, other=0.0)
    L_q_1 = tl.load(L_ptr_flat + b_h * S_len + q_offset + 64 + tl.arange(0, 64), mask=(q_offset + 64 + tl.arange(0, 64)) < S_len, other=0.0)
    
    acc_dQ0_0 = tl.zeros((64, 64), tl.float32)
    acc_dQ0_1 = tl.zeros((64, 64), tl.float32)
    acc_dQ1_0 = tl.zeros((64, 64), tl.float32)
    acc_dQ1_1 = tl.zeros((64, 64), tl.float32)
    
    num_k_tiles = tl.cdiv(S_len, 128)
    for k_tile in range(num_k_tiles):
        k_offset = k_tile * 128
        if k_offset >= S_len:
            break
            
        flat_k_offset = b_h * S_len + k_offset
        
        k0_0 = K_desc.load([flat_k_offset, 0])
        k0_1 = K_desc.load([flat_k_offset + 64, 0])
        k1_0 = K_desc.load([flat_k_offset, 64])
        k1_1 = K_desc.load([flat_k_offset + 64, 64])
        
        v0_0 = V_desc.load([flat_k_offset, 0])
        v0_1 = V_desc.load([flat_k_offset + 64, 0])
        v1_0 = V_desc.load([flat_k_offset, 64])
        v1_1 = V_desc.load([flat_k_offset + 64, 64])
        
        S_00 = tl.dot(q0_0, k0_0.T) + tl.dot(q1_0, k1_0.T)
        S_01 = tl.dot(q0_0, k0_1.T) + tl.dot(q1_0, k1_1.T)
        S_10 = tl.dot(q0_1, k0_0.T) + tl.dot(q1_1, k1_0.T)
        S_11 = tl.dot(q0_1, k0_1.T) + tl.dot(q1_1, k1_1.T)
        
        dP_00 = tl.dot(do0_0, v0_0.T) + tl.dot(do1_0, v1_0.T)
        dP_01 = tl.dot(do0_0, v0_1.T) + tl.dot(do1_0, v1_1.T)
        dP_10 = tl.dot(do0_1, v0_0.T) + tl.dot(do1_1, v1_0.T)
        dP_11 = tl.dot(do0_1, v0_1.T) + tl.dot(do1_1, v1_1.T)
        
        P_00 = tl.exp(S_00 * sqrt_d - L_q_0[:, None])
        P_01 = tl.exp(S_01 * sqrt_d - L_q_0[:, None])
        P_10 = tl.exp(S_10 * sqrt_d - L_q_1[:, None])
        P_11 = tl.exp(S_11 * sqrt_d - L_q_1[:, None])
        
        valid_k_0 = (k_offset + tl.arange(0, 64)) < S_len
        valid_k_1 = (k_offset + 64 + tl.arange(0, 64)) < S_len
        valid_q_0 = (q_offset + tl.arange(0, 64)) < S_len
        valid_q_1 = (q_offset + 64 + tl.arange(0, 64)) < S_len
        
        P_00 = tl.where(valid_q_0[:, None] & valid_k_0[None, :], P_00, 0.0)
        P_01 = tl.where(valid_q_0[:, None] & valid_k_1[None, :], P_01, 0.0)
        P_10 = tl.where(valid_q_1[:, None] & valid_k_0[None, :], P_10, 0.0)
        P_11 = tl.where(valid_q_1[:, None] & valid_k_1[None, :], P_11, 0.0)
        
        ds_00 = dP_00 * P_00
        ds_01 = dP_01 * P_01
        ds_10 = dP_10 * P_10
        ds_11 = dP_11 * P_11
        
        acc_dQ0_0 += tl.dot(ds_00, k0_0) + tl.dot(ds_01, k0_1)
        acc_dQ0_1 += tl.dot(ds_10, k0_0) + tl.dot(ds_11, k0_1)
        acc_dQ1_0 += tl.dot(ds_00, k1_0) + tl.dot(ds_01, k1_1)
        acc_dQ1_1 += tl.dot(ds_10, k1_0) + tl.dot(ds_11, k1_1)
        
    dQ_desc.store([flat_q_offset, 0], (acc_dQ0_0 * sqrt_d).to(tl.bfloat16))
    dQ_desc.store([flat_q_offset + 64, 0], (acc_dQ0_1 * sqrt_d).to(tl.bfloat16))
    dQ_desc.store([flat_q_offset, 64], (acc_dQ1_0 * sqrt_d).to(tl.bfloat16))
    dQ_desc.store([flat_q_offset + 64, 64], (acc_dQ1_1 * sqrt_d).to(tl.bfloat16))


@triton.jit
def bwd_dk_dv_kernel(
    Q_desc, K_desc, V_desc, dO_desc, L_ptr_flat, dK_desc, dV_desc,
    S_len, sqrt_d,
):
    k_tile = tl.program_id(0)
    b_h = tl.program_id(1)
    
    k_offset = k_tile * 128
    if k_offset >= S_len:
        return
        
    flat_k_offset = b_h * S_len + k_offset

    k0_0 = K_desc.load([flat_k_offset, 0])
    k0_1 = K_desc.load([flat_k_offset + 64, 0])
    k1_0 = K_desc.load([flat_k_offset, 64])
    k1_1 = K_desc.load([flat_k_offset + 64, 64])
    
    v0_0 = V_desc.load([flat_k_offset, 0])
    v0_1 = V_desc.load([flat_k_offset + 64, 0])
    v1_0 = V_desc.load([flat_k_offset, 64])
    v1_1 = V_desc.load([flat_k_offset + 64, 64])
    
    acc_dK0_0 = tl.zeros((64, 64), tl.float32)
    acc_dK0_1 = tl.zeros((64, 64), tl.float32)
    acc_dK1_0 = tl.zeros((64, 64), tl.float32)
    acc_dK1_1 = tl.zeros((64, 64), tl.float32)
    acc_dV0_0 = tl.zeros((64, 64), tl.float32)
    acc_dV0_1 = tl.zeros((64, 64), tl.float32)
    acc_dV1_0 = tl.zeros((64, 64), tl.float32)
    acc_dV1_1 = tl.zeros((64, 64), tl.float32)
    
    num_q_tiles = tl.cdiv(S_len, 128)
    for q_tile in range(num_q_tiles):
        q_offset = q_tile * 128
        if q_offset >= S_len:
            break
            
        flat_q_offset = b_h * S_len + q_offset
        
        q0_0 = Q_desc.load([flat_q_offset, 0])
        q0_1 = Q_desc.load([flat_q_offset + 64, 0])
        q1_0 = Q_desc.load([flat_q_offset, 64])
        q1_1 = Q_desc.load([flat_q_offset + 64, 64])
        
        do0_0 = dO_desc.load([flat_q_offset, 0])
        do0_1 = dO_desc.load([flat_q_offset + 64, 0])
        do1_0 = dO_desc.load([flat_q_offset, 64])
        do1_1 = dO_desc.load([flat_q_offset + 64, 64])
        
        L_q_0 = tl.load(L_ptr_flat + b_h * S_len + q_offset + tl.arange(0, 64), mask=(q_offset + tl.arange(0, 64)) < S_len, other=0.0)
        L_q_1 = tl.load(L_ptr_flat + b_h * S_len + q_offset + 64 + tl.arange(0, 64), mask=(q_offset + 64 + tl.arange(0, 64)) < S_len, other=0.0)
        
        S_00 = tl.dot(q0_0, k0_0.T) + tl.dot(q1_0, k1_0.T)
        S_01 = tl.dot(q0_0, k0_1.T) + tl.dot(q1_0, k1_1.T)
        S_10 = tl.dot(q0_1, k0_0.T) + tl.dot(q1_1, k1_0.T)
        S_11 = tl.dot(q0_1, k0_1.T) + tl.dot(q1_1, k1_1.T)
        
        dP_00 = tl.dot(do0_0, v0_0.T) + tl.dot(do1_0, v1_0.T)
        dP_01 = tl.dot(do0_0, v0_1.T) + tl.dot(do1_0, v1_1.T)
        dP_10 = tl.dot(do0_1, v0_0.T) + tl.dot(do1_1, v1_0.T)
        dP_11 = tl.dot(do0_1, v0_1.T) + tl.dot(do1_1, v1_1.T)
        
        P_00 = tl.exp(S_00 * sqrt_d - L_q_0[:, None])
        P_01 = tl.exp(S_01 * sqrt_d - L_q_0[:, None])
        P_10 = tl.exp(S_10 * sqrt_d - L_q_1[:, None])
        P_11 = tl.exp(S_11 * sqrt_d - L_q_1[:, None])
        
        valid_q_0 = (q_offset + tl.arange(0, 64)) < S_len
        valid_q_1 = (q_offset + 64 + tl.arange(0, 64)) < S_len
        valid_k_0 = (k_offset + tl.arange(0, 64)) < S_len
        valid_k_1 = (k_offset + 64 + tl.arange(0, 64)) < S_len
        
        P_00 = tl.where(valid_q_0[:, None] & valid_k_0[None, :], P_00, 0.0)
        P_01 = tl.where(valid_q_0[:, None] & valid_k_1[None, :], P_01, 0.0)
        P_10 = tl.where(valid_q_1[:, None] & valid_k_0[None, :], P_10, 0.0)
        P_11 = tl.where(valid_q_1[:, None] & valid_k_1[None, :], P_11, 0.0)
        
        ds_00 = dP_00 * P_00
        ds_01 = dP_01 * P_01
        ds_10 = dP_10 * P_10
        ds_11 = dP_11 * P_11
        
        acc_dK0_0 += tl.dot(ds_00.T, q0_0) + tl.dot(ds_10.T, q0_1)
        acc_dK0_1 += tl.dot(ds_01.T, q0_0) + tl.dot(ds_11.T, q0_1)
        acc_dK1_0 += tl.dot(ds_00.T, q1_0) + tl.dot(ds_10.T, q1_1)
        acc_dK1_1 += tl.dot(ds_01.T, q1_0) + tl.dot(ds_11.T, q1_1)
        
        acc_dV0_0 += tl.dot(P_00.T, do0_0) + tl.dot(P_10.T, do0_1)
        acc_dV0_1 += tl.dot(P_01.T, do0_0) + tl.dot(P_11.T, do0_1)
        acc_dV1_0 += tl.dot(P_00.T, do1_0) + tl.dot(P_10.T, do1_1)
        acc_dV1_1 += tl.dot(P_01.T, do1_0) + tl.dot(P_11.T, do1_1)
            
    dK_desc.store([flat_k_offset, 0], (acc_dK0_0 * sqrt_d).to(tl.bfloat16))
    dK_desc.store([flat_k_offset + 64, 0], (acc_dK0_1 * sqrt_d).to(tl.bfloat16))
    dK_desc.store([flat_k_offset, 64], (acc_dK1_0 * sqrt_d).to(tl.bfloat16))
    dK_desc.store([flat_k_offset + 64, 64], (acc_dK1_1 * sqrt_d).to(tl.bfloat16))
    
    dV_desc.store([flat_k_offset, 0], acc_dV0_0.to(tl.bfloat16))
    dV_desc.store([flat_k_offset + 64, 0], acc_dV0_1.to(tl.bfloat16))
    dV_desc.store([flat_k_offset, 64], acc_dV1_0.to(tl.bfloat16))
    dV_desc.store([flat_k_offset + 64, 64], acc_dV1_1.to(tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S_len, d = Q.shape
    sqrt_d = 1.0 / math.sqrt(d)
    
    Q_f = Q.flatten(0, 2)
    K_f = K.flatten(0, 2)
    V_f = V.flatten(0, 2)
    dO_f = dO.flatten(0, 2)
    L_f = L.flatten(0, 1)
    dQ_f = dQ.flatten(0, 2)
    dK_f = dK.flatten(0, 2)
    dV_f = dV.flatten(0, 2)
    
    Q_desc = TensorDescriptor.from_tensor(Q_f, [64, 64])
    K_desc = TensorDescriptor.from_tensor(K_f, [64, 64])
    V_desc = TensorDescriptor.from_tensor(V_f, [64, 64])
    dO_desc = TensorDescriptor.from_tensor(dO_f, [64, 64])
    dQ_desc = TensorDescriptor.from_tensor(dQ_f, [64, 64])
    dK_desc = TensorDescriptor.from_tensor(dK_f, [64, 64])
    dV_desc = TensorDescriptor.from_tensor(dV_f, [64, 64])
    
    grid_dq = (triton.cdiv(S_len, 128), B * H)
    grid_dk_dv = (triton.cdiv(S_len, 128), B * H)

    bwd_dq_kernel[grid_dq](
        Q_desc, K_desc, V_desc, dO_desc, L_f, dQ_desc,
        S_len, sqrt_d,
        num_warps=8, num_stages=3)
    
    bwd_dk_dv_kernel[grid_dk_dv](
        Q_desc, K_desc, V_desc, dO_desc, L_f, dK_desc, dV_desc,
        S_len, sqrt_d,
        num_warps=8, num_stages=3)