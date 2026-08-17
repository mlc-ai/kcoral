import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
import math


@triton.jit
def bwd_dq_kernel(
    Q_desc, K_desc, V_desc, dO_desc, O_desc, L_ptr_flat, dQ_desc,
    S_len, sqrt_d,
):
    q_tile = tl.program_id(0)
    b_h = tl.program_id(1)
    
    q_offset = q_tile * 128
    if q_offset >= S_len:
        return
    
    flat_q_offset = b_h * S_len + q_offset
    q_rows_start = b_h * S_len + q_offset
    
    q0 = Q_desc.load([flat_q_offset, 0])
    q1 = Q_desc.load([flat_q_offset, 64])
    do0 = dO_desc.load([flat_q_offset, 0])
    do1 = dO_desc.load([flat_q_offset, 64])
    o0 = O_desc.load([flat_q_offset, 0])
    o1 = O_desc.load([flat_q_offset, 64])
    
    L_q = tl.load(L_ptr_flat + q_rows_start + tl.arange(0, 128), mask=(q_rows_start + tl.arange(0, 128)) < S_len, other=0.0)
    L_q_expanded = L_q[:, None]
    
    acc_dQ0 = tl.zeros((128, 64), tl.float32)
    acc_dQ1 = tl.zeros((128, 64), tl.float32)
    
    num_k_tiles = tl.cdiv(S_len, 128)
    
    for k_tile in range(num_k_tiles):
        k_offset = k_tile * 128
        if k_offset >= S_len:
            break
            
        flat_k_offset = b_h * S_len + k_offset
        
        k0 = K_desc.load([flat_k_offset, 0])
        k1 = K_desc.load([flat_k_offset, 64])
        v0 = V_desc.load([flat_k_offset, 0])
        v1 = V_desc.load([flat_k_offset, 64])
        
        S_acc = tl.zeros((128, 128), tl.float32)
        d_acc = tl.zeros((128, 128), tl.float32)
        
        S_acc += tl.dot(q0, k0)
        d_acc += tl.dot(do0, v0)
        S_acc += tl.dot(q1, k1)
        d_acc += tl.dot(do1, v1)
        
        S_scaled = S_acc * sqrt_d
        k_rows = k_offset + tl.arange(0, 128)
        valid_mask = k_rows[None, :] < S_len
        S_scaled = tl.where(valid_mask, S_scaled, -1e20)
        
        p = tl.exp(S_scaled - L_q_expanded)
        p = tl.where(valid_mask, p, 0.0)
        
        ds = d_acc * p
        
        p_T = p.T
        
        k0_T = k0.T
        k1_T = k1.T
        
        acc_dQ0 += tl.dot(ds, k0_T)
        acc_dQ1 += tl.dot(ds, k1_T)
        
    dQ_desc.store([flat_q_offset, 0], (acc_dQ0 * sqrt_d).to(tl.bfloat16))
    dQ_desc.store([flat_q_offset, 64], (acc_dQ1 * sqrt_d).to(tl.bfloat16))


@triton.jit
def bwd_dk_dv_kernel(
    Q_desc, K_desc, V_desc, dO_desc, O_desc, L_ptr_flat, dK_desc, dV_desc,
    S_len, sqrt_d,
):
    k_tile = tl.program_id(0)
    b_h = tl.program_id(1)
    
    k_offset = k_tile * 128
    if k_offset >= S_len:
        return
        
    flat_k_offset = b_h * S_len + k_offset
    k_rows = b_h * S_len + k_offset

    k0 = K_desc.load([flat_k_offset, 0])
    k1 = K_desc.load([flat_k_offset, 64])
    v0 = V_desc.load([flat_k_offset, 0])
    v1 = V_desc.load([flat_k_offset, 64])
    
    L_k = tl.load(L_ptr_flat + k_rows + tl.arange(0, 128), mask=(k_rows + tl.arange(0, 128)) < S_len, other=0.0)
    L_k_expanded = L_k[:, None]
    
    acc_dK0 = tl.zeros((128, 64), tl.float32)
    acc_dK1 = tl.zeros((128, 64), tl.float32)
    acc_dV0 = tl.zeros((128, 64), tl.float32)
    acc_dV1 = tl.zeros((128, 64), tl.float32)
    
    num_q_tiles = tl.cdiv(S_len, 128)
    
    for q_tile in range(num_q_tiles):
        q_offset = q_tile * 128
        if q_offset >= S_len:
            break
            
        flat_q_offset = b_h * S_len + q_offset
        
        q0 = Q_desc.load([flat_q_offset, 0])
        q1 = Q_desc.load([flat_q_offset, 64])
        do0 = dO_desc.load([flat_q_offset, 0])
        do1 = dO_desc.load([flat_q_offset, 64])
        o0 = O_desc.load([flat_q_offset, 0])
        o1 = O_desc.load([flat_q_offset, 64])
        
        L_q = tl.load(L_ptr_flat + b_h * S_len + q_offset + tl.arange(0, 128), mask=(b_h * S_len + q_offset + tl.arange(0, 128)) < S_len, other=0.0)
        L_q_expanded = L_q[:, None]
        
        S_acc = tl.zeros((128, 128), tl.float32)
        d_acc = tl.zeros((128, 128), tl.float32)
        
        S_acc += tl.dot(q0, k0)
        d_acc += tl.dot(do0, v0)
        S_acc += tl.dot(q1, k1)
        d_acc += tl.dot(do1, v1)
        
        S_scaled = S_acc * sqrt_d
        
        q_rows = q_offset + tl.arange(0, 128)
        valid_mask_q = q_rows[:, None] < S_len
        S_scaled = tl.where(valid_mask_q, S_scaled, -1e20)
        
        k_rows_c = k_offset + tl.arange(0, 128)
        valid_mask_k = k_rows_c[None, :] < S_len
        S_scaled = tl.where(valid_mask_k, S_scaled, -1e20)
        
        p = tl.exp(S_scaled - L_k_expanded)
        
        p = tl.where(valid_mask_q, p, 0.0)
        p = tl.where(valid_mask_k, p, 0.0)
        
        ds = d_acc * p
        
        p_T = p.T
        ds_T = ds.T
        
        q0_T = q0.T
        q1_T = q1.T
        do0_T = do0.T
        do1_T = do1.T
        
        p_half0 = p[:, :64]
        p_half1 = p[:, 64:]
        
        p_half0_T = p_half0.T
        p_half1_T = p_half1.T
        
        ds_half0 = ds[:, :64]
        ds_half1 = ds[:, 64:]
        
        ds_half0_T = ds_half0.T
        ds_half1_T = ds_half1.T
        
        acc_dK0 += tl.dot(ds_half0_T, q0_T)
        acc_dK1 += tl.dot(ds_half1_T, q1_T)
        acc_dV0 += tl.dot(p_half0_T, do0_T)
        acc_dV1 += tl.dot(p_half1_T, do1_T)
        
    dK_desc.store([flat_k_offset, 0], (acc_dK0 * sqrt_d).to(tl.bfloat16))
    dK_desc.store([flat_k_offset, 64], (acc_dK1 * sqrt_d).to(tl.bfloat16))
    dV_desc.store([flat_k_offset, 0], acc_dV0.to(tl.bfloat16))
    dV_desc.store([flat_k_offset, 64], acc_dV1.to(tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S_len, d = Q.shape
    sqrt_d = 1.0 / math.sqrt(d)
    
    Q_f = Q.flatten(0, 2)
    K_f = K.flatten(0, 2)
    V_f = V.flatten(0, 2)
    O_f = O.flatten(0, 2)
    dO_f = dO.flatten(0, 2)
    L_f = L.flatten(0, 1)
    dQ_f = dQ.flatten(0, 2)
    dK_f = dK.flatten(0, 2)
    dV_f = dV.flatten(0, 2)
    
    Q_desc = TensorDescriptor.from_tensor(Q_f, [128, 64])
    K_desc = TensorDescriptor.from_tensor(K_f, [128, 64])
    V_desc = TensorDescriptor.from_tensor(V_f, [128, 64])
    O_desc = TensorDescriptor.from_tensor(O_f, [128, 64])
    dO_desc = TensorDescriptor.from_tensor(dO_f, [128, 64])
    dQ_desc = TensorDescriptor.from_tensor(dQ_f, [128, 64])
    dK_desc = TensorDescriptor.from_tensor(dK_f, [128, 64])
    dV_desc = TensorDescriptor.from_tensor(dV_f, [128, 64])
    
    grid_dq = (triton.cdiv(S_len, 128), B * H)
    grid_dk_dv = (triton.cdiv(S_len, 128), B * H)

    bwd_dq_kernel[grid_dq](
        Q_desc, K_desc, V_desc, dO_desc, O_desc, L_f, dQ_desc,
        S_len, sqrt_d,
        num_warps=8, num_stages=3)
    
    bwd_dk_dv_kernel[grid_dk_dv](
        Q_desc, K_desc, V_desc, dO_desc, O_desc, L_f, dK_desc, dV_desc,
        S_len, sqrt_d,
        num_warps=8, num_stages=3)