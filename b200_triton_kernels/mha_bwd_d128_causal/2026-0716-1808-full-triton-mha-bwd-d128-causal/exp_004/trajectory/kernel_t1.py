import math
import torch
import triton
import triton.language as tl


@triton.jit
def _bwd_dQ(
    Q, K, V, O, dO, L, dQ,
    S, stride_b, stride_h, stride_s, stride_d, H,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    """
    Computes the gradient w.r.t. the queries (dQ) with fully tiled inner reductions.
    """
    i_block = tl.program_id(0)
    bh = tl.program_id(1)
    h = bh // H
    b = bh % H
    
    offset_m = i_block * BLOCK_M
    if offset_m >= S:
        return
    
    base_ptr = b * stride_b + h * stride_h
    
    q_row = tl.arange(0, BLOCK_M)
    col = tl.arange(0, HEAD_DIM)
    
    Q_load = tl.load(Q + base_ptr + (offset_m + q_row[:, None]) * stride_s + col[None, :] * stride_d, 
                      mask=(offset_m + q_row[:, None]) < S, other=0.0)
    O_load = tl.load(O + base_ptr + (offset_m + q_row[:, None]) * stride_s + col[None, :] * stride_d, 
                      mask=(offset_m + q_row[:, None]) < S, other=0.0)
    dO_load = tl.load(dO + base_ptr + (offset_m + q_row[:, None]) * stride_s + col[None, :] * stride_d, 
                       mask=(offset_m + q_row[:, None]) < S, other=0.0)
    
    d_val = (dO_load * O_load).sum(axis=1)
    
    L_base = L + b * H * S + h * S
    l_load = tl.load(L_base + offset_m + q_row, mask=(offset_m + q_row) < S, other=0.0)
    
    acc_dQ = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)
    
    scale = 1.0 / math.sqrt(HEAD_DIM)
    
    k_row = tl.arange(0, BLOCK_N)
    
    j_max = (S - 1) // BLOCK_M if S > 0 else 0
    max_j = min(i_block, j_max)
    
    for j in range(max_j + 1):
        offset_n = j * BLOCK_N
        
        acc_S = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        acc_dP = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        
        for k0 in range(0, HEAD_DIM, BLOCK_K):
            col_k = k0 + tl.arange(0, BLOCK_K)
            
            a0 = tl.load(Q + base_ptr + (offset_m + q_row[:, None]) * stride_s + col_k[None, :] * stride_d, 
                          mask=(offset_m + q_row[:, None]) < S, other=0.0)
            b0 = tl.load(K + base_ptr + (offset_n + k_row[:, None]) * stride_s + col_k[None, :] * stride_d, 
                          mask=(offset_n + k_row[:, None]) < S, other=0.0)
            
            b0_T = b0.T
            acc_S = tl.dot(a0, b0_T, acc_S)
            
            dO0 = tl.load(dO + base_ptr + (offset_m + q_row[:, None]) * stride_s + col_k[None, :] * stride_d, 
                           mask=(offset_m + q_row[:, None]) < S, other=0.0)
            V0 = tl.load(V + base_ptr + (offset_n + k_row[:, None]) * stride_s + col_k[None, :] * stride_d, 
                          mask=(offset_n + k_row[:, None]) < S, other=0.0)
            
            V0_T = V0.T
            acc_dP = tl.dot(dO0, V0_T, acc_dP)
        
        p_unmasked = tl.exp(acc_S * scale - l_load[:, None])
        
        causal_mask = ((i_block * BLOCK_M + q_row[:, None]) >= (j * BLOCK_N + k_row[None, :])) & ((j * BLOCK_N + k_row[None, :]) < S)
        
        P = tl.where(causal_mask, p_unmasked, 0.0)
        P = tl.where((offset_m + q_row[:, None]) < S, P, 0.0)
        
        dS = tl.where((offset_m + q_row[:, None]) < S, P * (acc_dP - d_val[:, None]) * scale, 0.0)
        
        K_load = tl.load(K + base_ptr + (offset_n + k_row[:, None]) * stride_s + col[None, :] * stride_d, 
                          mask=(offset_n + k_row[:, None]) < S, other=0.0)
        
        for k0 in range(0, BLOCK_N, BLOCK_K):
            k_row_0 = k_row[k0:k0+BLOCK_K]
            dS0 = dS[:, k_row_0]
            K0 = K_load[k_row_0, :]
            acc_dQ = tl.dot(dS0, K0, acc_dQ)
        
    out_ptr = dQ + base_ptr + (offset_m + q_row[:, None]) * stride_s + col[None, :] * stride_d
    tl.store(out_ptr, acc_dQ, mask=(offset_m + q_row[:, None]) < S)


@triton.jit
def _bwd_dK_dV(
    Q, K, V, O, dO, L, dK, dV,
    S, stride_b, stride_h, stride_s, stride_d, H,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    """
    Resolves gradients w.r.t. the memory bank keys and values (dK, dV) with fully tiled inner reductions.
    """
    j_block = tl.program_id(0)
    bh = tl.program_id(1)
    h = bh // H
    b = bh % H
    
    offset_n = j_block * BLOCK_N
    if offset_n >= S:
        return
    
    base_ptr = b * stride_b + h * stride_h
    
    k_row = tl.arange(0, BLOCK_N)
    col = tl.arange(0, HEAD_DIM)
    
    K_load = tl.load(K + base_ptr + (offset_n + k_row[:, None]) * stride_s + col[None, :] * stride_d, 
                      mask=(offset_n + k_row[:, None]) < S, other=0.0)
    V_load = tl.load(V + base_ptr + (offset_n + k_row[:, None]) * stride_s + col[None, :] * stride_d, 
                      mask=(offset_n + k_row[:, None]) < S, other=0.0)
    
    acc_dK = tl.zeros((BLOCK_N, HEAD_DIM), tl.float32)
    acc_dV = tl.zeros((BLOCK_N, HEAD_DIM), tl.float32)
    
    scale = 1.0 / math.sqrt(HEAD_DIM)
    
    q_row = tl.arange(0, BLOCK_M)
    
    for i in range(j_block, num_blocks_per_head):
        offset_m = i * BLOCK_M
        
        Q_load = tl.load(Q + base_ptr + (offset_m + q_row[:, None]) * stride_s + col[None, :] * stride_d, 
                          mask=(offset_m + q_row[:, None]) < S, other=0.0)
        O_load = tl.load(O + base_ptr + (offset_m + q_row[:, None]) * stride_s + col[None, :] * stride_d, 
                          mask=(offset_m + q_row[:, None]) < S, other=0.0)
        dO_load = tl.load(dO + base_ptr + (offset_m + q_row[:, None]) * stride_s + col[None, :] * stride_d, 
                           mask=(offset_m + q_row[:, None]) < S, other=0.0)
        
        d_val = (dO_load * O_load).sum(axis=1)
        
        L_base = L + b * H * S + h * S
        l_load = tl.load(L_base + offset_m + q_row, mask=(offset_m + q_row) < S, other=0.0)
        
        acc_S = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        acc_dP = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        
        for k0 in range(0, HEAD_DIM, BLOCK_K):
            col_k = k0 + tl.arange(0, BLOCK_K)
            
            a0 = tl.load(Q + base_ptr + (offset_m + q_row[:, None]) * stride_s + col_k[None, :] * stride_d, 
                          mask=(offset_m + q_row[:, None]) < S, other=0.0)
            b0 = tl.load(K + base_ptr + (offset_n + k_row[:, None]) * stride_s + col_k[None, :] * stride_d, 
                          mask=(offset_n + k_row[:, None]) < S, other=0.0)
            
            b0_T = b0.T
            acc_S = tl.dot(a0, b0_T, acc_S)
            
            dO0 = tl.load(dO + base_ptr + (offset_m + q_row[:, None]) * stride_s + col_k[None, :] * stride_d, 
                           mask=(offset_m + q_row[:, None]) < S, other=0.0)
            V0 = tl.load(V + base_ptr + (offset_n + k_row[:, None]) * stride_s + col_k[None, :] * stride_d, 
                          mask=(offset_n + k_row[:, None]) < S, other=0.0)
            
            V0_T = V0.T
            acc_dP = tl.dot(dO0, V0_T, acc_dP)
        
        p_unmasked = tl.exp(acc_S * scale - l_load[:, None])
        
        causal_mask = ((i * BLOCK_M + q_row[:, None]) >= (j_block * BLOCK_N + k_row[None, :])) & ((j_block * BLOCK_N + k_row[None, :]) < S)
        
        P = tl.where(causal_mask, p_unmasked, 0.0)
        P = tl.where((offset_m + q_row[:, None]) < S, P, 0.0)
        
        dS = tl.where((offset_m + q_row[:, None]) < S, P * (acc_dP - d_val[:, None]) * scale, 0.0)
        
        dS_T = dS.T
        P_T = P.T
        
        q_row_transposed = q_row[None, :]
        dS_T_padded = tl.where((offset_m + q_row_transposed) < S, dS_T, 0.0)
        P_T_padded = tl.where((offset_m + q_row_transposed) < S, P_T, 0.0)
        
        for k0 in range(0, BLOCK_M, BLOCK_K):
            q_row_0 = q_row[k0:k0+BLOCK_K]
            dS_T0 = dS_T_padded[:, q_row_0]
            Q0 = Q_load[q_row_0, :]
            acc_dK = tl.dot(dS_T0, Q0, acc_dK)
            
            P_T0 = P_T_padded[:, q_row_0]
            dO0 = dO_load[q_row_0, :]
            acc_dV = tl.dot(P_T0, dO0, acc_dV)
        
    out_ptr_K = dK + base_ptr + (offset_n + k_row[:, None]) * stride_s + col[None, :] * stride_d
    tl.store(out_ptr_K, acc_dK, mask=(offset_n + k_row[:, None]) < S)
    
    out_ptr_V = dV + base_ptr + (offset_n + k_row[:, None]) * stride_s + col[None, :] * stride_d
    tl.store(out_ptr_V, acc_dV, mask=(offset_n + k_row[:, None]) < S)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Host bridge launching the two sequential Triton device-pass routines.
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, d = Q.shape
    
    stride_b = H * S * d
    stride_h = S * d
    stride_s = d
    stride_d = 1
    
    num_blocks_per_head = triton.cdiv(S, 16)
    grid = (num_blocks_per_head, B * H)
    
    _bwd_dQ[grid](Q, K, V, O, dO, L, dQ, S, stride_b, stride_h, stride_s, stride_d, H, BLOCK_M=16, BLOCK_N=16, BLOCK_K=16, HEAD_DIM=128)
    _bwd_dK_dV[grid](Q, K, V, O, dO, L, dK, dV, S, stride_b, stride_h, stride_s, stride_d, H, BLOCK_M=16, BLOCK_N=16, BLOCK_K=16, HEAD_DIM=128)