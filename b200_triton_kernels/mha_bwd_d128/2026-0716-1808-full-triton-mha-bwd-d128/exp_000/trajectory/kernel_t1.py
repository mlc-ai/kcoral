import torch
import triton
import triton.language as tl
import math


@triton.jit
def bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, dQ_ptr,
    S_len, sqrt_d,
):
    pid_q = tl.program_id(0)
    b_h = tl.program_id(1)
    
    q_rows = pid_q * 128 + tl.arange(0, 128)
    d_col = tl.arange(0, 128)
    
    q_full = tl.load(Q_ptr + (b_h * S_len + q_rows[:, None]) * 128 + d_col[None, :],
                     mask=q_rows[:, None] < S_len, other=0.0)
    do_full = tl.load(dO_ptr + (b_h * S_len + q_rows[:, None]) * 128 + d_col[None, :],
                      mask=q_rows[:, None] < S_len, other=0.0)
    
    q_h0, q_h1 = tl.split(q_full, 2)
    do_h0, do_h1 = tl.split(do_full, 2)
    
    l_q = tl.load(L_ptr + b_h * S_len + q_rows, mask=q_rows < S_len, other=0.0)
    
    acc_dQ_h0 = tl.zeros((128, 64), tl.float32)
    acc_dQ_h1 = tl.zeros((128, 64), tl.float32)
    
    num_k_tiles = tl.cdiv(S_len, 128)
    for k_tile in range(num_k_tiles):
        k_rows = k_tile * 128 + tl.arange(0, 128)
        
        k_full = tl.load(K_ptr + (b_h * S_len + k_rows[:, None]) * 128 + d_col[None, :],
                         mask=k_rows[:, None] < S_len, other=0.0)
        v_full = tl.load(V_ptr + (b_h * S_len + k_rows[:, None]) * 128 + d_col[None, :],
                         mask=k_rows[:, None] < S_len, other=0.0)
        
        k_h0, k_h1 = tl.split(k_full, 2)
        v_h0, v_h1 = tl.split(v_full, 2)
        
        s = tl.zeros((128, 128), tl.float32)
        d = tl.zeros((128, 128), tl.float32)
        
        s += tl.dot(q_h0, k_h0.T)
        d += tl.dot(do_h0, v_h0.T)
        s += tl.dot(q_h1, k_h1.T)
        d += tl.dot(do_h1, v_h1.T)
        
        s *= sqrt_d
        p = tl.exp(s - l_q[:, None])
        ds = d * p
        
        acc_dQ_h0 += tl.dot(ds, k_h0)
        acc_dQ_h1 += tl.dot(ds, k_h1)
    
    mask_q = q_rows[:, None] < S_len
    out_ptr_h0 = dQ_ptr + (b_h * S_len + q_rows[:, None]) * 128 + tl.arange(0, 64)[None, :]
    out_ptr_h1 = out_ptr_h0 + 64
    
    tl.store(out_ptr_h0, (acc_dQ_h0 * sqrt_d).to(tl.bfloat16), mask=mask_q)
    tl.store(out_ptr_h1, (acc_dQ_h1 * sqrt_d).to(tl.bfloat16), mask=mask_q)


@triton.jit
def bwd_dk_dv_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, dK_ptr, dV_ptr,
    S_len, sqrt_d,
):
    pid_k = tl.program_id(0)
    b_h = tl.program_id(1)
    
    k_rows = pid_k * 128 + tl.arange(0, 128)
    d_col = tl.arange(0, 128)
    
    k_full = tl.load(K_ptr + (b_h * S_len + k_rows[:, None]) * 128 + d_col[None, :],
                     mask=k_rows[:, None] < S_len, other=0.0)
    v_full = tl.load(V_ptr + (b_h * S_len + k_rows[:, None]) * 128 + d_col[None, :],
                     mask=k_rows[:, None] < S_len, other=0.0)
    
    k_h0, k_h1 = tl.split(k_full, 2)
    v_h0, v_h1 = tl.split(v_full, 2)
    
    acc_dK_h0 = tl.zeros((128, 64), tl.float32)
    acc_dK_h1 = tl.zeros((128, 64), tl.float32)
    acc_dV_h0 = tl.zeros((128, 64), tl.float32)
    acc_dV_h1 = tl.zeros((128, 64), tl.float32)
    
    num_q_tiles = tl.cdiv(S_len, 128)
    for q_tile in range(num_q_tiles):
        q_rows = q_tile * 128 + tl.arange(0, 128)
        
        q_full = tl.load(Q_ptr + (b_h * S_len + q_rows[:, None]) * 128 + d_col[None, :],
                         mask=q_rows[:, None] < S_len, other=0.0)
        do_full = tl.load(dO_ptr + (b_h * S_len + q_rows[:, None]) * 128 + d_col[None, :],
                          mask=q_rows[:, None] < S_len, other=0.0)
        
        q_h0, q_h1 = tl.split(q_full, 2)
        do_h0, do_h1 = tl.split(do_full, 2)
        
        l_q = tl.load(L_ptr + b_h * S_len + q_rows, mask=q_rows < S_len, other=0.0)
        
        s = tl.zeros((128, 128), tl.float32)
        d = tl.zeros((128, 128), tl.float32)
        
        s += tl.dot(q_h0, k_h0.T)
        d += tl.dot(do_h0, v_h0.T)
        s += tl.dot(q_h1, k_h1.T)
        d += tl.dot(do_h1, v_h1.T)
        
        s *= sqrt_d
        
        mask_k = k_rows[None, :] < S_len
        p = tl.exp(s - l_q[:, None]) * mask_k
        ds = d * p
        
        acc_dK_h0 += tl.dot(ds.T, q_h0)
        acc_dK_h1 += tl.dot(ds.T, q_h1)
        acc_dV_h0 += tl.dot(p.T, do_h0)
        acc_dV_h1 += tl.dot(p.T, do_h1)
    
    mask_k = k_rows[:, None] < S_len
    out_ptr_k_h0 = dK_ptr + (b_h * S_len + k_rows[:, None]) * 128 + tl.arange(0, 64)[None, :]
    out_ptr_k_h1 = out_ptr_k_h0 + 64
    out_ptr_v_h0 = dV_ptr + (b_h * S_len + k_rows[:, None]) * 128 + tl.arange(0, 64)[None, :]
    out_ptr_v_h1 = out_ptr_v_h0 + 64
    
    tl.store(out_ptr_k_h0, (acc_dK_h0 * sqrt_d).to(tl.bfloat16), mask=mask_k)
    tl.store(out_ptr_k_h1, (acc_dK_h1 * sqrt_d).to(tl.bfloat16), mask=mask_k)
    tl.store(out_ptr_v_h0, acc_dV_h0.to(tl.bfloat16), mask=mask_k)
    tl.store(out_ptr_v_h1, acc_dV_h1.to(tl.bfloat16), mask=mask_k)


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