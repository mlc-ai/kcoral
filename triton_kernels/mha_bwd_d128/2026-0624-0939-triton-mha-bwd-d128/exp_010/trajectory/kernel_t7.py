import torch
import triton
import triton.language as tl
import math


@triton.jit
def _dk_dv_kernel(
    K_ptr, V_ptr, Q_ptr, dO_ptr, O_ptr, L_ptr, dK_ptr, dV_ptr,
    B, H, S, T_r, K,
    stride_h, stride_s, stride_d, stride_l_h,
    scale: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    j = tl.program_id(0)
    bid = tl.program_id(1)
    
    row_offset = tl.arange(0, BLOCK_N)
    col_offset = tl.arange(0, 32)
    
    base_k = bid * stride_h + j * BLOCK_N * stride_s
    mask_k = (j * BLOCK_N + row_offset) < S
    
    K_j_c0 = tl.load(K_ptr + base_k + row_offset[:, None] * stride_s + col_offset[None, :] * stride_d, mask=mask_k[:, None], other=0.0).to(tl.float32)
    K_j_c1 = tl.load(K_ptr + base_k + row_offset[:, None] * stride_s + (col_offset + 32)[None, :] * stride_d, mask=mask_k[:, None], other=0.0).to(tl.float32)
    K_j_c2 = tl.load(K_ptr + base_k + row_offset[:, None] * stride_s + (col_offset + 64)[None, :] * stride_d, mask=mask_k[:, None], other=0.0).to(tl.float32)
    K_j_c3 = tl.load(K_ptr + base_k + row_offset[:, None] * stride_s + (col_offset + 96)[None, :] * stride_d, mask=mask_k[:, None], other=0.0).to(tl.float32)
    
    base_v = bid * stride_h + j * BLOCK_N * stride_s
    V_j_c0 = tl.load(V_ptr + base_v + row_offset[:, None] * stride_s + col_offset[None, :] * stride_d, mask=mask_k[:, None], other=0.0).to(tl.float32)
    V_j_c1 = tl.load(V_ptr + base_v + row_offset[:, None] * stride_s + (col_offset + 32)[None, :] * stride_d, mask=mask_k[:, None], other=0.0).to(tl.float32)
    V_j_c2 = tl.load(V_ptr + base_v + row_offset[:, None] * stride_s + (col_offset + 64)[None, :] * stride_d, mask=mask_k[:, None], other=0.0).to(tl.float32)
    V_j_c3 = tl.load(V_ptr + base_v + row_offset[:, None] * stride_s + (col_offset + 96)[None, :] * stride_d, mask=mask_k[:, None], other=0.0).to(tl.float32)
    
    acc_dK_c0 = tl.zeros((BLOCK_N, 32), tl.float32)
    acc_dK_c1 = tl.zeros((BLOCK_N, 32), tl.float32)
    acc_dK_c2 = tl.zeros((BLOCK_N, 32), tl.float32)
    acc_dK_c3 = tl.zeros((BLOCK_N, 32), tl.float32)
    acc_dV_c0 = tl.zeros((BLOCK_N, 32), tl.float32)
    acc_dV_c1 = tl.zeros((BLOCK_N, 32), tl.float32)
    acc_dV_c2 = tl.zeros((BLOCK_N, 32), tl.float32)
    acc_dV_c3 = tl.zeros((BLOCK_N, 32), tl.float32)
    
    k_row_offset = tl.arange(0, BLOCK_N)
    
    for i in range(T_r):
        base_q = bid * stride_h + i * BLOCK_N * stride_s
        mask_q = (i * BLOCK_N + row_offset) < S
        
        Q_i_c0 = tl.load(Q_ptr + base_q + row_offset[:, None] * stride_s + col_offset[None, :] * stride_d, mask=mask_q[:, None], other=0.0).to(tl.float32)
        Q_i_c1 = tl.load(Q_ptr + base_q + row_offset[:, None] * stride_s + (col_offset + 32)[None, :] * stride_d, mask=mask_q[:, None], other=0.0).to(tl.float32)
        Q_i_c2 = tl.load(Q_ptr + base_q + row_offset[:, None] * stride_s + (col_offset + 64)[None, :] * stride_d, mask=mask_q[:, None], other=0.0).to(tl.float32)
        Q_i_c3 = tl.load(Q_ptr + base_q + row_offset[:, None] * stride_s + (col_offset + 96)[None, :] * stride_d, mask=mask_q[:, None], other=0.0).to(tl.float32)
        
        base_do = bid * stride_h + i * BLOCK_N * stride_s
        dO_i_c0 = tl.load(dO_ptr + base_do + row_offset[:, None] * stride_s + col_offset[None, :] * stride_d, mask=mask_q[:, None], other=0.0).to(tl.float32)
        dO_i_c1 = tl.load(dO_ptr + base_do + row_offset[:, None] * stride_s + (col_offset + 32)[None, :] * stride_d, mask=mask_q[:, None], other=0.0).to(tl.float32)
        dO_i_c2 = tl.load(dO_ptr + base_do + row_offset[:, None] * stride_s + (col_offset + 64)[None, :] * stride_d, mask=mask_q[:, None], other=0.0).to(tl.float32)
        dO_i_c3 = tl.load(dO_ptr + base_do + row_offset[:, None] * stride_s + (col_offset + 96)[None, :] * stride_d, mask=mask_q[:, None], other=0.0).to(tl.float32)
        
        base_o = bid * stride_h + i * BLOCK_N * stride_s
        O_i_c0 = tl.load(O_ptr + base_o + row_offset[:, None] * stride_s + col_offset[None, :] * stride_d, mask=mask_q[:, None], other=0.0).to(tl.float32)
        O_i_c1 = tl.load(O_ptr + base_o + row_offset[:, None] * stride_s + (col_offset + 32)[None, :] * stride_d, mask=mask_q[:, None], other=0.0).to(tl.float32)
        O_i_c2 = tl.load(O_ptr + base_o + row_offset[:, None] * stride_s + (col_offset + 64)[None, :] * stride_d, mask=mask_q[:, None], other=0.0).to(tl.float32)
        O_i_c3 = tl.load(O_ptr + base_o + row_offset[:, None] * stride_s + (col_offset + 96)[None, :] * stride_d, mask=mask_q[:, None], other=0.0).to(tl.float32)
        
        D_i = tl.sum(dO_i_c0 * O_i_c0 + dO_i_c1 * O_i_c1 + dO_i_c2 * O_i_c2 + dO_i_c3 * O_i_c3, axis=1)
        
        base_l = bid * stride_l_h + i * BLOCK_N
        L_i = tl.load(L_ptr + base_l + row_offset, mask=mask_q, other=1e20)
        
        L_i = L_i[:, None]
        D_i = D_i[:, None]
        
        S_val = (tl.dot(Q_i_c0, K_j_c0.T) + tl.dot(Q_i_c1, K_j_c1.T) + 
                 tl.dot(Q_i_c2, K_j_c2.T) + tl.dot(Q_i_c3, K_j_c3.T))
        
        mask_k_col = (j * BLOCK_N + k_row_offset) < S
        S_scaled = S_val * scale
        
        P = tl.exp(S_scaled - L_i)
        P = tl.where(mask_k_col[None, :], P, 0.0)
        
        dP = (tl.dot(dO_i_c0, V_j_c0.T) + tl.dot(dO_i_c1, V_j_c1.T) + 
              tl.dot(dO_i_c2, V_j_c2.T) + tl.dot(dO_i_c3, V_j_c3.T))
        
        dS = P * (dP - D_i) * scale
        
        acc_dV_c0 = tl.dot(P.T, dO_i_c0, acc_dV_c0)
        acc_dV_c1 = tl.dot(P.T, dO_i_c1, acc_dV_c1)
        acc_dV_c2 = tl.dot(P.T, dO_i_c2, acc_dV_c2)
        acc_dV_c3 = tl.dot(P.T, dO_i_c3, acc_dV_c3)
        
        acc_dK_c0 = tl.dot(dS.T, Q_i_c0, acc_dK_c0)
        acc_dK_c1 = tl.dot(dS.T, Q_i_c1, acc_dK_c1)
        acc_dK_c2 = tl.dot(dS.T, Q_i_c2, acc_dK_c2)
        acc_dK_c3 = tl.dot(dS.T, Q_i_c3, acc_dK_c3)
    
    base_dk = bid * stride_h + j * BLOCK_N * stride_s
    mask_store = (j * BLOCK_N + row_offset) < S
    tl.store(dK_ptr + base_dk + row_offset[:, None] * stride_s + col_offset[None, :] * stride_d, acc_dK_c0.to(tl.bfloat16), mask=mask_store[:, None])
    tl.store(dK_ptr + base_dk + row_offset[:, None] * stride_s + (col_offset + 32)[None, :] * stride_d, acc_dK_c1.to(tl.bfloat16), mask=mask_store[:, None])
    tl.store(dK_ptr + base_dk + row_offset[:, None] * stride_s + (col_offset + 64)[None, :] * stride_d, acc_dK_c2.to(tl.bfloat16), mask=mask_store[:, None])
    tl.store(dK_ptr + base_dk + row_offset[:, None] * stride_s + (col_offset + 96)[None, :] * stride_d, acc_dK_c3.to(tl.bfloat16), mask=mask_store[:, None])
    
    tl.store(dV_ptr + base_dk + row_offset[:, None] * stride_s + col_offset[None, :] * stride_d, acc_dV_c0.to(tl.bfloat16), mask=mask_store[:, None])
    tl.store(dV_ptr + base_dk + row_offset[:, None] * stride_s + (col_offset + 32)[None, :] * stride_d, acc_dV_c1.to(tl.bfloat16), mask=mask_store[:, None])
    tl.store(dV_ptr + base_dk + row_offset[:, None] * stride_s + (col_offset + 64)[None, :] * stride_d, acc_dV_c2.to(tl.bfloat16), mask=mask_store[:, None])
    tl.store(dV_ptr + base_dk + row_offset[:, None] * stride_s + (col_offset + 96)[None, :] * stride_d, acc_dV_c3.to(tl.bfloat16), mask=mask_store[:, None])


@triton.jit
def _dq_kernel(
    Q_ptr, dO_ptr, O_ptr, L_ptr, K_ptr, V_ptr, dQ_ptr,
    B, H, S, T_c, K,
    stride_h, stride_s, stride_d, stride_l_h,
    scale: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    i = tl.program_id(0)
    bid = tl.program_id(1)
    
    row_offset = tl.arange(0, BLOCK_N)
    col_offset = tl.arange(0, 32)
    
    base_q = bid * stride_h + i * BLOCK_N * stride_s
    mask_q = (i * BLOCK_N + row_offset) < S
    
    Q_i_c0 = tl.load(Q_ptr + base_q + row_offset[:, None] * stride_s + col_offset[None, :] * stride_d, mask=mask_q[:, None], other=0.0).to(tl.float32)
    Q_i_c1 = tl.load(Q_ptr + base_q + row_offset[:, None] * stride_s + (col_offset + 32)[None, :] * stride_d, mask=mask_q[:, None], other=0.0).to(tl.float32)
    Q_i_c2 = tl.load(Q_ptr + base_q + row_offset[:, None] * stride_s + (col_offset + 64)[None, :] * stride_d, mask=mask_q[:, None], other=0.0).to(tl.float32)
    Q_i_c3 = tl.load(Q_ptr + base_q + row_offset[:, None] * stride_s + (col_offset + 96)[None, :] * stride_d, mask=mask_q[:, None], other=0.0).to(tl.float32)
    
    base_do = bid * stride_h + i * BLOCK_N * stride_s
    dO_i_c0 = tl.load(dO_ptr + base_do + row_offset[:, None] * stride_s + col_offset[None, :] * stride_d, mask=mask_q[:, None], other=0.0).to(tl.float32)
    dO_i_c1 = tl.load(dO_ptr + base_do + row_offset[:, None] * stride_s + (col_offset + 32)[None, :] * stride_d, mask=mask_q[:, None], other=0.0).to(tl.float32)
    dO_i_c2 = tl.load(dO_ptr + base_do + row_offset[:, None] * stride_s + (col_offset + 64)[None, :] * stride_d, mask=mask_q[:, None], other=0.0).to(tl.float32)
    dO_i_c3 = tl.load(dO_ptr + base_do + row_offset[:, None] * stride_s + (col_offset + 96)[None, :] * stride_d, mask=mask_q[:, None], other=0.0).to(tl.float32)
                    
    base_o = bid * stride_h + i * BLOCK_N * stride_s
    O_i_c0 = tl.load(O_ptr + base_o + row_offset[:, None] * stride_s + col_offset[None, :] * stride_d, mask=mask_q[:, None], other=0.0).to(tl.float32)
    O_i_c1 = tl.load(O_ptr + base_o + row_offset[:, None] * stride_s + (col_offset + 32)[None, :] * stride_d, mask=mask_q[:, None], other=0.0).to(tl.float32)
    O_i_c2 = tl.load(O_ptr + base_o + row_offset[:, None] * stride_s + (col_offset + 64)[None, :] * stride_d, mask=mask_q[:, None], other=0.0).to(tl.float32)
    O_i_c3 = tl.load(O_ptr + base_o + row_offset[:, None] * stride_s + (col_offset + 96)[None, :] * stride_d, mask=mask_q[:, None], other=0.0).to(tl.float32)
    
    D_i = tl.sum(dO_i_c0 * O_i_c0 + dO_i_c1 * O_i_c1 + dO_i_c2 * O_i_c2 + dO_i_c3 * O_i_c3, axis=1)
    
    base_l = bid * stride_l_h + i * BLOCK_N
    L_i = tl.load(L_ptr + base_l + row_offset, mask=mask_q, other=1e20)
    
    L_i = L_i[:, None]
    D_i = D_i[:, None]
    
    acc_dQ_c0 = tl.zeros((BLOCK_N, 32), tl.float32)
    acc_dQ_c1 = tl.zeros((BLOCK_N, 32), tl.float32)
    acc_dQ_c2 = tl.zeros((BLOCK_N, 32), tl.float32)
    acc_dQ_c3 = tl.zeros((BLOCK_N, 32), tl.float32)
    
    k_row_offset = tl.arange(0, BLOCK_N)
    
    for j in range(T_c):
        base_k = bid * stride_h + j * BLOCK_N * stride_s
        mask_k = (j * BLOCK_N + row_offset) < S
        
        K_j_c0 = tl.load(K_ptr + base_k + row_offset[:, None] * stride_s + col_offset[None, :] * stride_d, mask=mask_k[:, None], other=0.0).to(tl.float32)
        K_j_c1 = tl.load(K_ptr + base_k + row_offset[:, None] * stride_s + (col_offset + 32)[None, :] * stride_d, mask=mask_k[:, None], other=0.0).to(tl.float32)
        K_j_c2 = tl.load(K_ptr + base_k + row_offset[:, None] * stride_s + (col_offset + 64)[None, :] * stride_d, mask=mask_k[:, None], other=0.0).to(tl.float32)
        K_j_c3 = tl.load(K_ptr + base_k + row_offset[:, None] * stride_s + (col_offset + 96)[None, :] * stride_d, mask=mask_k[:, None], other=0.0).to(tl.float32)
        
        base_v = bid * stride_h + j * BLOCK_N * stride_s
        V_j_c0 = tl.load(V_ptr + base_v + row_offset[:, None] * stride_s + col_offset[None, :] * stride_d, mask=mask_k[:, None], other=0.0).to(tl.float32)
        V_j_c1 = tl.load(V_ptr + base_v + row_offset[:, None] * stride_s + (col_offset + 32)[None, :] * stride_d, mask=mask_k[:, None], other=0.0).to(tl.float32)
        V_j_c2 = tl.load(V_ptr + base_v + row_offset[:, None] * stride_s + (col_offset + 64)[None, :] * stride_d, mask=mask_k[:, None], other=0.0).to(tl.float32)
        V_j_c3 = tl.load(V_ptr + base_v + row_offset[:, None] * stride_s + (col_offset + 96)[None, :] * stride_d, mask=mask_k[:, None], other=0.0).to(tl.float32)
        
        S_val = (tl.dot(Q_i_c0, K_j_c0.T) + tl.dot(Q_i_c1, K_j_c1.T) + 
                 tl.dot(Q_i_c2, K_j_c2.T) + tl.dot(Q_i_c3, K_j_c3.T))
        
        mask_k_col = (j * BLOCK_N + k_row_offset) < S
        S_scaled = S_val * scale
        
        P = tl.exp(S_scaled - L_i)
        P = tl.where(mask_k_col[None, :], P, 0.0)
        
        dP = (tl.dot(dO_i_c0, V_j_c0.T) + tl.dot(dO_i_c1, V_j_c1.T) + 
              tl.dot(dO_i_c2, V_j_c2.T) + tl.dot(dO_i_c3, V_j_c3.T))
        
        dS = P * (dP - D_i) * scale
        
        acc_dQ_c0 = tl.dot(dS, K_j_c0, acc_dQ_c0)
        acc_dQ_c1 = tl.dot(dS, K_j_c1, acc_dQ_c1)
        acc_dQ_c2 = tl.dot(dS, K_j_c2, acc_dQ_c2)
        acc_dQ_c3 = tl.dot(dS, K_j_c3, acc_dQ_c3)
    
    base_dq = bid * stride_h + i * BLOCK_N * stride_s
    mask_store = (i * BLOCK_N + row_offset) < S
    tl.store(dQ_ptr + base_dq + row_offset[:, None] * stride_s + col_offset[None, :] * stride_d, acc_dQ_c0.to(tl.bfloat16), mask=mask_store[:, None])
    tl.store(dQ_ptr + base_dq + row_offset[:, None] * stride_s + (col_offset + 32)[None, :] * stride_d, acc_dQ_c1.to(tl.bfloat16), mask=mask_store[:, None])
    tl.store(dQ_ptr + base_dq + row_offset[:, None] * stride_s + (col_offset + 64)[None, :] * stride_d, acc_dQ_c2.to(tl.bfloat16), mask=mask_store[:, None])
    tl.store(dQ_ptr + base_dq + row_offset[:, None] * stride_s + (col_offset + 96)[None, :] * stride_d, acc_dQ_c3.to(tl.bfloat16), mask=mask_store[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    
    scale = 1.0 / math.sqrt(d)
    
    BLOCK_N = 64
    T_r = triton.cdiv(S, BLOCK_N)
    T_c = triton.cdiv(S, BLOCK_N)
    
    stride_h = Q.stride()[1]
    stride_s = Q.stride()[2]
    stride_d = Q.stride()[3]
    stride_l_h = L.stride()[1]
    
    grid_dkdv = (T_c, B * H)
    _dk_dv_kernel[grid_dkdv](
        K, V, Q, dO, O, L, dK, dV,
        B, H, S, T_r, d,
        stride_h, stride_s, stride_d, stride_l_h,
        scale=scale,
        BLOCK_N=BLOCK_N,
        num_warps=8,
        num_stages=3,
    )
    
    grid_dq = (T_r, B * H)
    _dq_kernel[grid_dq](
        Q, dO, O, L, K, V, dQ,
        B, H, S, T_c, d,
        stride_h, stride_s, stride_d, stride_l_h,
        scale=scale,
        BLOCK_N=BLOCK_N,
        num_warps=8,
        num_stages=3,
    )