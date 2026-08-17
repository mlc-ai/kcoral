import math
import torch
import triton
import triton.language as tl


@triton.jit
def dKdV_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, do_ptr, l_ptr,
    d_k_ptr, d_v_ptr,
    S, stride_s, stride_b_h,
    SCALE: tl.constexpr,
):
    pid_i = tl.program_id(0)
    bh = tl.program_id(1)
    
    i_start = pid_i * 64
    
    K_0 = tl.load(k_ptr + bh * stride_b_h + (i_start + tl.arange(0, 64))[:, None] * stride_s + tl.arange(0, 64)[None, :], mask=(i_start + tl.arange(0, 64))[:, None] < S, other=0.0)
    K_1 = tl.load(k_ptr + bh * stride_b_h + (i_start + tl.arange(0, 64))[:, None] * stride_s + (64 + tl.arange(0, 64))[None, :], mask=(i_start + tl.arange(0, 64))[:, None] < S, other=0.0)
    V_0 = tl.load(v_ptr + bh * stride_b_h + (i_start + tl.arange(0, 64))[:, None] * stride_s + tl.arange(0, 64)[None, :], mask=(i_start + tl.arange(0, 64))[:, None] < S, other=0.0)
    V_1 = tl.load(v_ptr + bh * stride_b_h + (i_start + tl.arange(0, 64))[:, None] * stride_s + (64 + tl.arange(0, 64))[None, :], mask=(i_start + tl.arange(0, 64))[:, None] < S, other=0.0)
    
    dK_0 = tl.zeros((64, 64), tl.float32)
    dK_1 = tl.zeros((64, 64), tl.float32)
    dV_0 = tl.zeros((64, 64), tl.float32)
    dV_1 = tl.zeros((64, 64), tl.float32)
    
    for j_start in range(0, S, 64):
        Q_0 = tl.load(q_ptr + bh * stride_b_h + (j_start + tl.arange(0, 64))[:, None] * stride_s + tl.arange(0, 64)[None, :], mask=(j_start + tl.arange(0, 64))[:, None] < S, other=0.0)
        Q_1 = tl.load(q_ptr + bh * stride_b_h + (j_start + tl.arange(0, 64))[:, None] * stride_s + (64 + tl.arange(0, 64))[None, :], mask=(j_start + tl.arange(0, 64))[:, None] < S, other=0.0)
        
        O_0 = tl.load(o_ptr + bh * stride_b_h + (j_start + tl.arange(0, 64))[:, None] * stride_s + tl.arange(0, 64)[None, :], mask=(j_start + tl.arange(0, 64))[:, None] < S, other=0.0)
        O_1 = tl.load(o_ptr + bh * stride_b_h + (j_start + tl.arange(0, 64))[:, None] * stride_s + (64 + tl.arange(0, 64))[None, :], mask=(j_start + tl.arange(0, 64))[:, None] < S, other=0.0)
        
        dO_0 = tl.load(do_ptr + bh * stride_b_h + (j_start + tl.arange(0, 64))[:, None] * stride_s + tl.arange(0, 64)[None, :], mask=(j_start + tl.arange(0, 64))[:, None] < S, other=0.0)
        dO_1 = tl.load(do_ptr + bh * stride_b_h + (j_start + tl.arange(0, 64))[:, None] * stride_s + (64 + tl.arange(0, 64))[None, :], mask=(j_start + tl.arange(0, 64))[:, None] < S, other=0.0)
        
        D = tl.sum(O_0 * dO_0 + O_1 * dO_1, axis=1)
        row_idx = j_start + tl.arange(0, 64)
        D = tl.where(row_idx[:, None] < S, D[:, None], 0.0)
        
        S_acc = tl.zeros((64, 64), tl.float32)
        S_acc = tl.dot(Q_0, K_0.T, S_acc)
        S_acc = tl.dot(Q_1, K_1.T, S_acc)
        
        dP_acc = tl.zeros((64, 64), tl.float32)
        dP_acc = tl.dot(dO_0, V_0.T, dP_acc)
        dP_acc = tl.dot(dO_1, V_1.T, dP_acc)
        
        l_ptr_j = l_ptr + bh * S + j_start + tl.arange(0, 64)
        L_j = tl.load(l_ptr_j)
        
        P = tl.exp(S_acc * SCALE - L_j[:, None])
        
        dS = P * (dP_acc - D) * SCALE
        
        P_T = P.T
        dS_T = dS.T
        
        dV_0 = tl.dot(P_T, dO_0, dV_0)
        dV_1 = tl.dot(P_T, dO_1, dV_1)
        
        dK_0 = tl.dot(dS_T, Q_0, dK_0)
        dK_1 = tl.dot(dS_T, Q_1, dK_1)
        
    row_idx = i_start + tl.arange(0, 64)
    col_idx_0 = tl.arange(0, 64)
    ptr_k0 = d_k_ptr + bh * stride_b_h + row_idx[:, None] * stride_s + col_idx_0[None, :]
    tl.store(ptr_k0, dK_0.to(tl.bfloat16), mask=(row_idx[:, None] < S) & (col_idx_0[None, :] < 128))
    
    col_idx_1 = 64 + tl.arange(0, 64)
    ptr_k1 = d_k_ptr + bh * stride_b_h + row_idx[:, None] * stride_s + col_idx_1[None, :]
    tl.store(ptr_k1, dK_1.to(tl.bfloat16), mask=(row_idx[:, None] < S) & (col_idx_1[None, :] < 128))
    
    ptr_v0 = d_v_ptr + bh * stride_b_h + row_idx[:, None] * stride_s + col_idx_0[None, :]
    tl.store(ptr_v0, dV_0.to(tl.bfloat16), mask=(row_idx[:, None] < S) & (col_idx_0[None, :] < 128))
    
    ptr_v1 = d_v_ptr + bh * stride_b_h + row_idx[:, None] * stride_s + col_idx_1[None, :]
    tl.store(ptr_v1, dV_1.to(tl.bfloat16), mask=(row_idx[:, None] < S) & (col_idx_1[None, :] < 128))


@triton.jit
def dQ_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, do_ptr, l_ptr,
    d_q_ptr,
    S, stride_s, stride_b_h,
    SCALE: tl.constexpr,
):
    pid_i = tl.program_id(0)
    bh = tl.program_id(1)
    
    i_start = pid_i * 64
    
    Q_0 = tl.load(q_ptr + bh * stride_b_h + (i_start + tl.arange(0, 64))[:, None] * stride_s + tl.arange(0, 64)[None, :], mask=(i_start + tl.arange(0, 64))[:, None] < S, other=0.0)
    Q_1 = tl.load(q_ptr + bh * stride_b_h + (i_start + tl.arange(0, 64))[:, None] * stride_s + (64 + tl.arange(0, 64))[None, :], mask=(i_start + tl.arange(0, 64))[:, None] < S, other=0.0)
    
    O_0 = tl.load(o_ptr + bh * stride_b_h + (i_start + tl.arange(0, 64))[:, None] * stride_s + tl.arange(0, 64)[None, :], mask=(i_start + tl.arange(0, 64))[:, None] < S, other=0.0)
    O_1 = tl.load(o_ptr + bh * stride_b_h + (i_start + tl.arange(0, 64))[:, None] * stride_s + (64 + tl.arange(0, 64))[None, :], mask=(i_start + tl.arange(0, 64))[:, None] < S, other=0.0)
    
    dO_0 = tl.load(do_ptr + bh * stride_b_h + (i_start + tl.arange(0, 64))[:, None] * stride_s + tl.arange(0, 64)[None, :], mask=(i_start + tl.arange(0, 64))[:, None] < S, other=0.0)
    dO_1 = tl.load(do_ptr + bh * stride_b_h + (i_start + tl.arange(0, 64))[:, None] * stride_s + (64 + tl.arange(0, 64))[None, :], mask=(i_start + tl.arange(0, 64))[:, None] < S, other=0.0)
    
    D = tl.sum(O_0 * dO_0 + O_1 * dO_1, axis=1)
    row_idx = i_start + tl.arange(0, 64)
    D = tl.where(row_idx < S, D, 0.0)
    
    l_ptr_i = l_ptr + bh * S + i_start + tl.arange(0, 64)
    L_i = tl.load(l_ptr_i)
    
    dQ_0 = tl.zeros((64, 64), tl.float32)
    dQ_1 = tl.zeros((64, 64), tl.float32)
    
    for j_start in range(0, S, 64):
        K_0 = tl.load(k_ptr + bh * stride_b_h + (j_start + tl.arange(0, 64))[:, None] * stride_s + tl.arange(0, 64)[None, :], mask=(j_start + tl.arange(0, 64))[:, None] < S, other=0.0)
        K_1 = tl.load(k_ptr + bh * stride_b_h + (j_start + tl.arange(0, 64))[:, None] * stride_s + (64 + tl.arange(0, 64))[None, :], mask=(j_start + tl.arange(0, 64))[:, None] < S, other=0.0)
        
        V_0 = tl.load(v_ptr + bh * stride_b_h + (j_start + tl.arange(0, 64))[:, None] * stride_s + tl.arange(0, 64)[None, :], mask=(j_start + tl.arange(0, 64))[:, None] < S, other=0.0)
        V_1 = tl.load(v_ptr + bh * stride_b_h + (j_start + tl.arange(0, 64))[:, None] * stride_s + (64 + tl.arange(0, 64))[None, :], mask=(j_start + tl.arange(0, 64))[:, None] < S, other=0.0)
        
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
        
    row_idx = i_start + tl.arange(0, 64)
    col_idx_0 = tl.arange(0, 64)
    ptr_q0 = d_q_ptr + bh * stride_b_h + row_idx[:, None] * stride_s + col_idx_0[None, :]
    tl.store(ptr_q0, dQ_0.to(tl.bfloat16), mask=(row_idx[:, None] < S) & (col_idx_0[None, :] < 128))
    
    col_idx_1 = 64 + tl.arange(0, 64)
    ptr_q1 = d_q_ptr + bh * stride_b_h + row_idx[:, None] * stride_s + col_idx_1[None, :]
    tl.store(ptr_q1, dQ_1.to(tl.bfloat16), mask=(row_idx[:, None] < S) & (col_idx_1[None, :] < 128))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute backward pass for Multi-Head Attention."""
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    assert B == 4 and H == 48 and d == 128
    
    scale = 1.0 / math.sqrt(d)
    
    stride_s = 128
    stride_b_h = S * 128
    
    num_blocks = triton.cdiv(S, 64)
    grid = (num_blocks, B * H)
    
    dKdV_kernel[grid](
        Q, K, V, O, dO, L, dK, dV,
        S, stride_s, stride_b_h, SCALE=scale,
        num_warps=4, num_stages=3
    )
    
    dQ_kernel[grid](
        Q, K, V, O, dO, L, dQ,
        S, stride_s, stride_b_h, SCALE=scale,
        num_warps=4, num_stages=3
    )