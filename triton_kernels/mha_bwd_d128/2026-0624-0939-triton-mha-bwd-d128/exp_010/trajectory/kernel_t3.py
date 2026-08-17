import torch
import triton
import triton.language as tl


@triton.jit
def _dk_dv_kernel(
    K_ptr, V_ptr, Q_ptr, dO_ptr, O_ptr, L_ptr, dK_ptr, dV_ptr,
    B, H, S, T_r,
    stride_h, stride_s, stride_d, stride_l_h,
    scale: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    j = tl.program_id(0)
    bid = tl.program_id(1)
    
    row_offset = tl.arange(0, BLOCK_N)
    col_offset = tl.arange(0, 64)
    
    # Load K and V once
    base_k = bid * stride_h + j * BLOCK_N * stride_s
    mask_k = (j * BLOCK_N + row_offset) < S
    K_j_l = tl.load(K_ptr + base_k + row_offset[:, None] * stride_s + col_offset[None, :] * stride_d, 
                  mask=mask_k[:, None], other=0.0).to(tl.float32)
    K_j_r = tl.load(K_ptr + base_k + row_offset[:, None] * stride_s + (col_offset + 64)[None, :] * stride_d, 
                  mask=mask_k[:, None], other=0.0).to(tl.float32)
    
    base_v = bid * stride_h + j * BLOCK_N * stride_s
    V_j_l = tl.load(V_ptr + base_v + row_offset[:, None] * stride_s + col_offset[None, :] * stride_d, 
                  mask=mask_k[:, None], other=0.0).to(tl.float32)
    V_j_r = tl.load(V_ptr + base_v + row_offset[:, None] * stride_s + (col_offset + 64)[None, :] * stride_d, 
                  mask=mask_k[:, None], other=0.0).to(tl.float32)
    
    acc_dK_l = tl.zeros((BLOCK_N, 64), tl.float32)
    acc_dK_r = tl.zeros((BLOCK_N, 64), tl.float32)
    acc_dV_l = tl.zeros((BLOCK_N, 64), tl.float32)
    acc_dV_r = tl.zeros((BLOCK_N, 64), tl.float32)
    
    for i in range(T_r):
        base_q = bid * stride_h + i * BLOCK_N * stride_s
        mask_q = (i * BLOCK_N + row_offset) < S
        Q_i_l = tl.load(Q_ptr + base_q + row_offset[:, None] * stride_s + col_offset[None, :] * stride_d,
                       mask=mask_q[:, None], other=0.0).to(tl.float32)
        Q_i_r = tl.load(Q_ptr + base_q + row_offset[:, None] * stride_s + (col_offset + 64)[None, :] * stride_d,
                       mask=mask_q[:, None], other=0.0).to(tl.float32)
        
        base_do = bid * stride_h + i * BLOCK_N * stride_s
        dO_i_l = tl.load(dO_ptr + base_do + row_offset[:, None] * stride_s + col_offset[None, :] * stride_d,
                        mask=mask_q[:, None], other=0.0).to(tl.float32)
        dO_i_r = tl.load(dO_ptr + base_do + row_offset[:, None] * stride_s + (col_offset + 64)[None, :] * stride_d,
                        mask=mask_q[:, None], other=0.0).to(tl.float32)
        
        base_o = bid * stride_h + i * BLOCK_N * stride_s
        O_i_l = tl.load(O_ptr + base_o + row_offset[:, None] * stride_s + col_offset[None, :] * stride_d,
                        mask=mask_q[:, None], other=0.0).to(tl.float32)
        O_i_r = tl.load(O_ptr + base_o + row_offset[:, None] * stride_s + (col_offset + 64)[None, :] * stride_d,
                        mask=mask_q[:, None], other=0.0).to(tl.float32)
        
        # Compute per-row invariant required by the backward math
        D_i = tl.sum(dO_i_l * O_i_l + dO_i_r * O_i_r, axis=1)
        
        base_l = bid * stride_l_h + i * BLOCK_N
        L_i = tl.load(L_ptr + base_l + row_offset, mask=mask_q, other=1e20)
        
        L_i = L_i[:, None]
        D_i = D_i[:, None]
        
        # S = Q_i @ K_j^T
        S_val = tl.dot(Q_i_l, K_j_l.T) + tl.dot(Q_i_r, K_j_r.T)
        
        # P = exp(S * scale - L_i)
        k_idx = j * BLOCK_N + row_offset
        mask_k_col = k_idx < S
        S_scaled = S_val * scale
        P = tl.exp(S_scaled - L_i)
        P = tl.where(mask_k_col[None, :], P, 0.0)
        
        # dP = dO_i @ V_j^T
        dP = tl.dot(dO_i_l, V_j_l.T) + tl.dot(dO_i_r, V_j_r.T)
        
        # dS = P * (dP - D_i) * scale
        dS = P * (dP - D_i) * scale
        
        # dV_j += P^T @ dO_i
        acc_dV_l = tl.dot(P.T, dO_i_l, acc_dV_l)
        acc_dV_r = tl.dot(P.T, dO_i_r, acc_dV_r)
        
        # dK_j += dS^T @ Q_i
        acc_dK_l = tl.dot(dS.T, Q_i_l, acc_dK_l)
        acc_dK_r = tl.dot(dS.T, Q_i_r, acc_dK_r)
    
    # Flush accumulated gradients back to HBM converted to bf16.
    base_dk = bid * stride_h + j * BLOCK_N * stride_s
    mask_store = (j * BLOCK_N + row_offset) < S
    tl.store(dK_ptr + base_dk + row_offset[:, None] * stride_s + col_offset[None, :] * stride_d, 
             acc_dK_l.to(tl.bfloat16), mask=mask_store[:, None])
    tl.store(dK_ptr + base_dk + row_offset[:, None] * stride_s + (col_offset + 64)[None, :] * stride_d, 
             acc_dK_r.to(tl.bfloat16), mask=mask_store[:, None])
    
    tl.store(dV_ptr + base_dk + row_offset[:, None] * stride_s + col_offset[None, :] * stride_d, 
             acc_dV_l.to(tl.bfloat16), mask=mask_store[:, None])
    tl.store(dV_ptr + base_dk + row_offset[:, None] * stride_s + (col_offset + 64)[None, :] * stride_d, 
             acc_dV_r.to(tl.bfloat16), mask=mask_store[:, None])


@triton.jit
def _dq_kernel(
    Q_ptr, dO_ptr, O_ptr, L_ptr, K_ptr, V_ptr, dQ_ptr,
    B, H, S, T_c,
    stride_h, stride_s, stride_d, stride_l_h,
    scale: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    i = tl.program_id(0)
    bid = tl.program_id(1)
    
    row_offset = tl.arange(0, BLOCK_N)
    col_offset = tl.arange(0, 64)
    
    # Load Q and dO once at the start of the program
    base_q = bid * stride_h + i * BLOCK_N * stride_s
    mask_q = (i * BLOCK_N + row_offset) < S
    Q_i_l = tl.load(Q_ptr + base_q + row_offset[:, None] * stride_s + col_offset[None, :] * stride_d,
                   mask=mask_q[:, None], other=0.0).to(tl.float32)
    Q_i_r = tl.load(Q_ptr + base_q + row_offset[:, None] * stride_s + (col_offset + 64)[None, :] * stride_d,
                   mask=mask_q[:, None], other=0.0).to(tl.float32)
    
    base_do = bid * stride_h + i * BLOCK_N * stride_s
    dO_i_l = tl.load(dO_ptr + base_do + row_offset[:, None] * stride_s + col_offset[None, :] * stride_d,
                    mask=mask_q[:, None], other=0.0).to(tl.float32)
    dO_i_r = tl.load(dO_ptr + base_do + row_offset[:, None] * stride_s + (col_offset + 64)[None, :] * stride_d,
                    mask=mask_q[:, None], other=0.0).to(tl.float32)
                    
    base_o = bid * stride_h + i * BLOCK_N * stride_s
    O_i_l = tl.load(O_ptr + base_o + row_offset[:, None] * stride_s + col_offset[None, :] * stride_d,
                    mask=mask_q[:, None], other=0.0).to(tl.float32)
    O_i_r = tl.load(O_ptr + base_o + row_offset[:, None] * stride_s + (col_offset + 64)[None, :] * stride_d,
                    mask=mask_q[:, None], other=0.0).to(tl.float32)
    
    # Compute per-row invariant required by the backward math
    D_i = tl.sum(dO_i_l * O_i_l + dO_i_r * O_i_r, axis=1)
    
    base_l = bid * stride_l_h + i * BLOCK_N
    L_i = tl.load(L_ptr + base_l + row_offset, mask=mask_q, other=1e20)
    
    L_i = L_i[:, None]
    D_i = D_i[:, None]
    
    acc_dQ_l = tl.zeros((BLOCK_N, 64), tl.float32)
    acc_dQ_r = tl.zeros((BLOCK_N, 64), tl.float32)
    
    for j in range(T_c):
        base_k = bid * stride_h + j * BLOCK_N * stride_s
        mask_k = (j * BLOCK_N + row_offset) < S
        K_j_l = tl.load(K_ptr + base_k + row_offset[:, None] * stride_s + col_offset[None, :] * stride_d,
                       mask=mask_k[:, None], other=0.0).to(tl.float32)
        K_j_r = tl.load(K_ptr + base_k + row_offset[:, None] * stride_s + (col_offset + 64)[None, :] * stride_d,
                       mask=mask_k[:, None], other=0.0).to(tl.float32)
        
        base_v = bid * stride_h + j * BLOCK_N * stride_s
        V_j_l = tl.load(V_ptr + base_v + row_offset[:, None] * stride_s + col_offset[None, :] * stride_d,
                       mask=mask_k[:, None], other=0.0).to(tl.float32)
        V_j_r = tl.load(V_ptr + base_v + row_offset[:, None] * stride_s + (col_offset + 64)[None, :] * stride_d,
                       mask=mask_k[:, None], other=0.0).to(tl.float32)
        
        # S = Q_i @ K_j^T
        S_val = tl.dot(Q_i_l, K_j_l.T) + tl.dot(Q_i_r, K_j_r.T)
        
        # P = exp(S * scale - L_i)
        k_idx = j * BLOCK_N + row_offset
        mask_k_col = k_idx < S
        S_scaled = S_val * scale
        P = tl.exp(S_scaled - L_i)
        P = tl.where(mask_k_col[None, :], P, 0.0)
        
        # dP = dO_i @ V_j^T
        dP = tl.dot(dO_i_l, V_j_l.T) + tl.dot(dO_i_r, V_j_r.T)
        
        # dS = P * (dP - D_i) * scale
        dS = P * (dP - D_i) * scale
        
        # dQ_i += dS @ K_j
        acc_dQ_l = tl.dot(dS, K_j_l, acc_dQ_l)
        acc_dQ_r = tl.dot(dS, K_j_r, acc_dQ_r)
    
    # Flush accumulated dQ gradients back to HBM converted to bf16.
    base_dq = bid * stride_h + i * BLOCK_N * stride_s
    mask_store = (i * BLOCK_N + row_offset) < S
    tl.store(dQ_ptr + base_dq + row_offset[:, None] * stride_s + col_offset[None, :] * stride_d, 
             acc_dQ_l.to(tl.bfloat16), mask=mask_store[:, None])
    tl.store(dQ_ptr + base_dq + row_offset[:, None] * stride_s + (col_offset + 64)[None, :] * stride_d, 
             acc_dQ_r.to(tl.bfloat16), mask=mask_store[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    
    scale = 1.0 / (d ** 0.5)
    
    BLOCK_N = 64
    T_r = triton.cdiv(S, BLOCK_N)
    T_c = triton.cdiv(S, BLOCK_N)
    
    stride_h = Q.stride()[1]
    stride_s = Q.stride()[2]
    stride_d = Q.stride()[3]
    stride_l_h = L.stride()[1]
    
    # Execute the dK and dV accumulation pass
    grid_dkdv = (T_c, B * H)
    _dk_dv_kernel[grid_dkdv](
        K, V, Q, dO, O, L, dK, dV,
        B, H, S, T_r,
        stride_h, stride_s, stride_d, stride_l_h,
        scale=scale,
        BLOCK_N=BLOCK_N,
        num_warps=8,
        num_stages=3,
    )
    
    # Execute the dQ accumulation pass
    grid_dq = (T_r, B * H)
    _dq_kernel[grid_dq](
        Q, dO, O, L, K, V, dQ,
        B, H, S, T_c,
        stride_h, stride_s, stride_d, stride_l_h,
        scale=scale,
        BLOCK_N=BLOCK_N,
        num_warps=8,
        num_stages=3,
    )