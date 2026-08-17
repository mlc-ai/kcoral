import math
import torch
import triton
import triton.language as tl


@triton.jit
def dKdV_kernel(
    Q, K, V, O, dO, L,
    dK, dV,
    S,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_i = tl.program_id(0)
    bh = tl.program_id(1)
    
    i_start = pid_i * 64
    
    stride_row = Q.stride(2)
    stride_col = Q.stride(3)
    
    row = tl.arange(0, 64)
    col = tl.arange(0, 64)
    
    K = tl.load(K + (i_start + row)[:, None] * stride_row + col[None, :] * stride_col, mask=(i_start + row)[:, None] < S, other=0.0)
    K_0, K_1 = split(K, 2, dim=1)
    
    V = tl.load(V + (i_start + row)[:, None] * stride_row + col[None, :] * stride_col, mask=(i_start + row)[:, None] < S, other=0.0)
    V_0, V_1 = split(V, 2, dim=1)
    
    dK_0 = tl.zeros((64, 64), tl.float32)
    dK_1 = tl.zeros((64, 64), tl.float32)
    dV_0 = tl.zeros((64, 64), tl.float32)
    dV_1 = tl.zeros((64, 64), tl.float32)
    
    for j_start in range(0, S, 64):
        Q = tl.load(Q + (j_start + row)[:, None] * stride_row + col[None, :] * stride_col, mask=(j_start + row)[:, None] < S, other=0.0)
        Q_0, Q_1 = split(Q, 2, dim=1)
        
        O = tl.load(O + (j_start + row)[:, None] * stride_row + col[None, :] * stride_col, mask=(j_start + row)[:, None] < S, other=0.0)
        O_0, O_1 = split(O, 2, dim=1)
        
        dO = tl.load(dO + (j_start + row)[:, None] * stride_row + col[None, :] * stride_col, mask=(j_start + row)[:, None] < S, other=0.0)
        dO_0, dO_1 = split(dO, 2, dim=1)
        
        D = tl.sum(O_0 * dO_0 + O_1 * dO_1, axis=1)
        D = tl.where((j_start + row)[:, None] < S, D[:, None], 0.0)
        
        S_acc = tl.zeros((64, 64), tl.float32)
        S_acc = dot(Q_0, K_0.T, S_acc)
        S_acc = dot(Q_1, K_1.T, S_acc)
        
        dP_acc = tl.zeros((64, 64), tl.float32)
        dP_acc = dot(dO_0, V_0.T, dP_acc)
        dP_acc = dot(dO_1, V_1.T, dP_acc)
        
        L_j = tl.load(L + (j_start + row)[..., None], mask=(j_start + row) < S, other=float('inf'), boundary_check=((j_start + row)[..., None]))
        
        P = exp(S_acc * SCALE - L_j[..., None])
        
        dS = P * (dP_acc - D) * SCALE
        
        P_T = P.T
        dS_T = dS.T
        
        dV_0 = dot(P_T, dO_0, dV_0)
        dV_1 = dot(P_T, dO_1, dV_1)
        
        dK_0 = dot(dS_T, Q_0, dK_0)
        dK_1 = dot(dS_T, Q_1, dK_1)
        
    dK_0 = dK_0.to(Q.dtype)
    ptr_k0 = dK + (i_start + row)[:, None] * stride_row + col[None, :] * stride_col
    tl.store(ptr_k0, dK_0, mask=(i_start + row)[:, None] < S)
    
    dK_1 = dK_1.to(Q.dtype)
    ptr_k1 = dK + (i_start + row)[:, None] * stride_row + (col + 64)[None, :] * stride_col
    tl.store(ptr_k1, dK_1, mask=(i_start + row)[:, None] < S)
    
    dV_0 = dV_0.to(Q.dtype)
    ptr_v0 = dV + (i_start + row)[:, None] * stride_row + col[None, :] * stride_col
    tl.store(ptr_v0, dV_0, mask=(i_start + row)[:, None] < S)
    
    dV_1 = dV_1.to(Q.dtype)
    ptr_v1 = dV + (i_start + row)[:, None] * stride_row + (col + 64)[None, :] * stride_col
    tl.store(ptr_v1, dV_1, mask=(i_start + row)[:, None] < S)


@triton.jit
def dQ_kernel(
    Q, K, V, O, dO, L,
    dQ,
    S,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_i = tl.program_id(0)
    bh = tl.program_id(1)
    
    i_start = pid_i * 64
    
    stride_row = Q.stride(2)
    stride_col = Q.stride(3)
    
    row = tl.arange(0, 64)
    col = tl.arange(0, 64)
    
    Q = tl.load(Q + (i_start + row)[:, None] * stride_row + col[None, :] * stride_col, mask=(i_start + row)[:, None] < S, other=0.0)
    Q_0, Q_1 = split(Q, 2, dim=1)
    
    O = tl.load(O + (i_start + row)[:, None] * stride_row + col[None, :] * stride_col, mask=(i_start + row)[:, None] < S, other=0.0)
    O_0, O_1 = split(O, 2, dim=1)
    
    dO = tl.load(dO + (i_start + row)[:, None] * stride_row + col[None, :] * stride_col, mask=(i_start + row)[:, None] < S, other=0.0)
    dO_0, dO_1 = split(dO, 2, dim=1)
    
    D = tl.sum(O_0 * dO_0 + O_1 * dO_1, axis=1)
    D = tl.where((i_start + row)[:, None] < S, D[:, None], 0.0)
    
    L_i = tl.load(L + (i_start + row)[..., None], mask=(i_start + row) < S, other=float('inf'), boundary_check=((i_start + row)[..., None]))
    
    dQ_0 = tl.zeros((64, 64), tl.float32)
    dQ_1 = tl.zeros((64, 64), tl.float32)
    
    for j_start in range(0, S, 64):
        K = tl.load(K + (j_start + row)[:, None] * stride_row + col[None, :] * stride_col, mask=(j_start + row)[:, None] < S, other=0.0)
        K_0, K_1 = split(K, 2, dim=1)
        
        V = tl.load(V + (j_start + row)[:, None] * stride_row + col[None, :] * stride_col, mask=(j_start + row)[:, None] < S, other=0.0)
        V_0, V_1 = split(V, 2, dim=1)
        
        S_acc = tl.zeros((64, 64), tl.float32)
        S_acc = dot(Q_0, K_0.T, S_acc)
        S_acc = dot(Q_1, K_1.T, S_acc)
        
        dP_acc = tl.zeros((64, 64), tl.float32)
        dP_acc = dot(dO_0, V_0.T, dP_acc)
        dP_acc = dot(dO_1, V_1.T, dP_acc)
        
        P = exp(S_acc * SCALE - L_i[..., None])
        
        dS = P * (dP_acc - D) * SCALE
        
        dQ_0 = dot(dS, K_0, dQ_0)
        dQ_1 = dot(dS, K_1, dQ_1)
        
    dQ_0 = dQ_0.to(Q.dtype)
    ptr_q0 = dQ + (i_start + row)[:, None] * stride_row + col[None, :] * stride_col
    tl.store(ptr_q0, dQ_0, mask=(i_start + row)[:, None] < S)
    
    dQ_1 = dQ_1.to(Q.dtype)
    ptr_q1 = dQ + (i_start + row)[:, None] * stride_row + (col + 64)[None, :] * stride_col
    tl.store(ptr_q1, dQ_1, mask=(i_start + row)[:, None] < S)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute backward pass for Multi-Head Attention."""
    B, H, S, d = Q.shape
    
    scale = 1.0 / math.sqrt(d)
    
    num_blocks = triton.cdiv(S, 64)
    grid = (num_blocks, B * H)
    
    dKdV_kernel[grid](
        Q, K, V, O, dO, L,
        dK, dV,
        S,
        SCALE=scale,
        BLOCK=128,
        num_warps=4, num_stages=2
    )
    
    dQ_kernel[grid](
        Q, K, V, O, dO, L,
        dQ,
        S,
        SCALE=scale,
        BLOCK=128,
        num_warps=4, num_stages=2
    )