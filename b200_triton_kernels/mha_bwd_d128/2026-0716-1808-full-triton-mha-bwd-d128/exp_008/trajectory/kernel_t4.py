import math
import torch
import triton
import triton.language as tl


@triton.jit
def dKdV_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
    dK_ptr, dV_ptr,
    S,
    SCALE: tl.constexpr,
):
    pid_i = tl.program_id(0)
    bh = tl.program_id(1)
    
    i_start = pid_i * 64
    
    row = tl.arange(0, 64)
    col = tl.arange(0, 64)
    
    K_0 = tl.load(K_ptr + bh * S * 128 + (i_start + row)[:, None] * 128 + col[None, :], mask=(i_start + row)[:, None] < S, other=0.0)
    K_1 = tl.load(K_ptr + bh * S * 128 + (i_start + row)[:, None] * 128 + (col + 64)[None, :], mask=(i_start + row)[:, None] < S, other=0.0)
    
    V_0 = tl.load(V_ptr + bh * S * 128 + (i_start + row)[:, None] * 128 + col[None, :], mask=(i_start + row)[:, None] < S, other=0.0)
    V_1 = tl.load(V_ptr + bh * S * 128 + (i_start + row)[:, None] * 128 + (col + 64)[None, :], mask=(i_start + row)[:, None] < S, other=0.0)
    
    dK_0 = tl.zeros((64, 64), tl.float32)
    dK_1 = tl.zeros((64, 64), tl.float32)
    dV_0 = tl.zeros((64, 64), tl.float32)
    dV_1 = tl.zeros((64, 64), tl.float32)
    
    for j_start in range(0, S, 64):
        Q_0 = tl.load(Q_ptr + bh * S * 128 + (j_start + row)[:, None] * 128 + col[None, :], mask=(j_start + row)[:, None] < S, other=0.0)
        Q_1 = tl.load(Q_ptr + bh * S * 128 + (j_start + row)[:, None] * 128 + (col + 64)[None, :], mask=(j_start + row)[:, None] < S, other=0.0)
        
        O_0 = tl.load(O_ptr + bh * S * 128 + (j_start + row)[:, None] * 128 + col[None, :], mask=(j_start + row)[:, None] < S, other=0.0)
        O_1 = tl.load(O_ptr + bh * S * 128 + (j_start + row)[:, None] * 128 + (col + 64)[None, :], mask=(j_start + row)[:, None] < S, other=0.0)
        
        dO_0 = tl.load(dO_ptr + bh * S * 128 + (j_start + row)[:, None] * 128 + col[None, :], mask=(j_start + row)[:, None] < S, other=0.0)
        dO_1 = tl.load(dO_ptr + bh * S * 128 + (j_start + row)[:, None] * 128 + (col + 64)[None, :], mask=(j_start + row)[:, None] < S, other=0.0)
        
        D = tl.sum(O_0 * dO_0 + O_1 * dO_1, axis=1)
        D = tl.where((j_start + row) < S, D, 0.0)
        
        S_acc = tl.zeros((64, 64), tl.float32)
        S_acc = tl.dot(Q_0, K_0.T, S_acc)
        S_acc = tl.dot(Q_1, K_1.T, S_acc)
        
        dP_acc = tl.zeros((64, 64), tl.float32)
        dP_acc = tl.dot(dO_0, V_0.T, dP_acc)
        dP_acc = tl.dot(dO_1, V_1.T, dP_acc)
        
        L_j = tl.load(L_ptr + bh * S + j_start + row, mask=(j_start + row) < S, other=float('inf'))
        
        P = tl.exp(S_acc * SCALE - L_j[:, None])
        
        dS = P * (dP_acc - D[:, None]) * SCALE
        
        P_T = P.T
        dS_T = dS.T
        
        dV_0 = tl.dot(P_T, dO_0, dV_0)
        dV_1 = tl.dot(P_T, dO_1, dV_1)
        
        dK_0 = tl.dot(dS_T, Q_0, dK_0)
        dK_1 = tl.dot(dS_T, Q_1, dK_1)
        
    ptr_k0 = dK_ptr + bh * S * 128 + (i_start + row)[:, None] * 128 + col[None, :]
    tl.store(ptr_k0, dK_0.to(tl.bfloat16), mask=(i_start + row)[:, None] < S)
    
    ptr_k1 = dK_ptr + bh * S * 128 + (i_start + row)[:, None] * 128 + (col + 64)[None, :]
    tl.store(ptr_k1, dK_1.to(tl.bfloat16), mask=(i_start + row)[:, None] < S)
    
    ptr_v0 = dV_ptr + bh * S * 128 + (i_start + row)[:, None] * 128 + col[None, :]
    tl.store(ptr_v0, dV_0.to(tl.bfloat16), mask=(i_start + row)[:, None] < S)
    
    ptr_v1 = dV_ptr + bh * S * 128 + (i_start + row)[:, None] * 128 + (col + 64)[None, :]
    tl.store(ptr_v1, dV_1.to(tl.bfloat16), mask=(i_start + row)[:, None] < S)


@triton.jit
def dQ_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
    dQ_ptr,
    S,
    SCALE: tl.constexpr,
):
    pid_i = tl.program_id(0)
    bh = tl.program_id(1)
    
    i_start = pid_i * 64
    
    row = tl.arange(0, 64)
    col = tl.arange(0, 64)
    
    Q_0 = tl.load(Q_ptr + bh * S * 128 + (i_start + row)[:, None] * 128 + col[None, :], mask=(i_start + row)[:, None] < S, other=0.0)
    Q_1 = tl.load(Q_ptr + bh * S * 128 + (i_start + row)[:, None] * 128 + (col + 64)[None, :], mask=(i_start + row)[:, None] < S, other=0.0)
    
    O_0 = tl.load(O_ptr + bh * S * 128 + (i_start + row)[:, None] * 128 + col[None, :], mask=(i_start + row)[:, None] < S, other=0.0)
    O_1 = tl.load(O_ptr + bh * S * 128 + (i_start + row)[:, None] * 128 + (col + 64)[None, :], mask=(i_start + row)[:, None] < S, other=0.0)
    
    dO_0 = tl.load(dO_ptr + bh * S * 128 + (i_start + row)[:, None] * 128 + col[None, :], mask=(i_start + row)[:, None] < S, other=0.0)
    dO_1 = tl.load(dO_ptr + bh * S * 128 + (i_start + row)[:, None] * 128 + (col + 64)[None, :], mask=(i_start + row)[:, None] < S, other=0.0)
    
    D = tl.sum(O_0 * dO_0 + O_1 * dO_1, axis=1)
    D = tl.where((i_start + row) < S, D, 0.0)
    
    L_i = tl.load(L_ptr + bh * S + i_start + row, mask=(i_start + row) < S, other=float('inf'))
    
    dQ_0 = tl.zeros((64, 64), tl.float32)
    dQ_1 = tl.zeros((64, 64), tl.float32)
    
    for j_start in range(0, S, 64):
        K_0 = tl.load(K_ptr + bh * S * 128 + (j_start + row)[:, None] * 128 + col[None, :], mask=(j_start + row)[:, None] < S, other=0.0)
        K_1 = tl.load(K_ptr + bh * S * 128 + (j_start + row)[:, None] * 128 + (col + 64)[None, :], mask=(j_start + row)[:, None] < S, other=0.0)
        
        V_0 = tl.load(V_ptr + bh * S * 128 + (j_start + row)[:, None] * 128 + col[None, :], mask=(j_start + row)[:, None] < S, other=0.0)
        V_1 = tl.load(V_ptr + bh * S * 128 + (j_start + row)[:, None] * 128 + (col + 64)[None, :], mask=(j_start + row)[:, None] < S, other=0.0)
        
        S_acc = tl.zeros((64, 64), tl.float32)
        S_acc = tl.dot(Q_0, K_0.T, S_acc)
        S_acc = tl.dot(Q_1, K_1.T, S_acc)
        
        dP_acc = tl.zeros((64, 64), tl.float32)
        dP_acc = tl.dot(dO_0, V_0.T, dP_acc)
        dP_acc = tl.dot(dO_1, V_1.T, dP_acc)
        
        P = tl.exp(S_acc * SCALE - L_i[:, None])
        
        dS = P * (dP_acc - D[:, None]) * SCALE
        
        dQ_0 = tl.dot(dS, K_0, dQ_0)
        dQ_1 = tl.dot(dS, K_1, dQ_1)
        
    ptr_q0 = dQ_ptr + bh * S * 128 + (i_start + row)[:, None] * 128 + col[None, :]
    tl.store(ptr_q0, dQ_0.to(tl.bfloat16), mask=(i_start + row)[:, None] < S)
    
    ptr_q1 = dQ_ptr + bh * S * 128 + (i_start + row)[:, None] * 128 + (col + 64)[None, :]
    tl.store(ptr_q1, dQ_1.to(tl.bfloat16), mask=(i_start + row)[:, None] < S)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute backward pass for Multi-Head Attention."""
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    
    scale = 1.0 / math.sqrt(d)
    
    Q_flat = Q.view(-1, S, d)
    K_flat = K.view(-1, S, d)
    V_flat = V.view(-1, S, d)
    O_flat = O.view(-1, S, d)
    dO_flat = dO.view(-1, S, d)
    dQ_flat = dQ.view(-1, S, d)
    dK_flat = dK.view(-1, S, d)
    dV_flat = dV.view(-1, S, d)
    
    num_blocks = triton.cdiv(S, 64)
    grid = (num_blocks, B * H)
    
    dKdV_kernel[grid](
        Q_flat.data_ptr(), K_flat.data_ptr(), V_flat.data_ptr(), O_flat.data_ptr(), 
        dO_flat.data_ptr(), L.data_ptr(),
        dK_flat.data_ptr(), dV_flat.data_ptr(),
        S,
        SCALE=scale,
        num_warps=4, num_stages=2
    )
    
    dQ_kernel[grid](
        Q_flat.data_ptr(), K_flat.data_ptr(), V_flat.data_ptr(), O_flat.data_ptr(), 
        dO_flat.data_ptr(), L.data_ptr(),
        dQ_flat.data_ptr(),
        S,
        SCALE=scale,
        num_warps=4, num_stages=2
    )