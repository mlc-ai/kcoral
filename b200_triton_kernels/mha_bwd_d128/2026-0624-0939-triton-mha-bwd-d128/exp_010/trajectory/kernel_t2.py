import torch
import triton
import triton.language as tl


@triton.jit
def _compute_D_kernel(
    dO_ptr, O_ptr, D_ptr, n_rows, stride_s, stride_d
):
    row_idx = tl.program_id(0)
    if row_idx >= n_rows:
        return
    col_offset = tl.arange(0, 128)
    base = row_idx * stride_s
    dO = tl.load(dO_ptr + base + col_offset * stride_d)
    O = tl.load(O_ptr + base + col_offset * stride_d)
    D = tl.sum(dO * O)
    tl.store(D_ptr + row_idx, D)


@triton.jit
def _dk_dv_kernel(
    K_ptr, V_ptr, Q_ptr, dO_ptr, L_ptr, D_ptr, dK_ptr, dV_ptr,
    B, H, S, T_r,
    stride_h, stride_s, stride_d, stride_l_h,
    scale: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    j = tl.program_id(0)
    bid = tl.program_id(1)
    
    row_offset = tl.arange(0, BLOCK_N)
    col_offset = tl.arange(0, 128)
    
    # Load K and V once
    base_k = bid * stride_h + j * BLOCK_N * stride_s
    mask_k = (j * BLOCK_N + row_offset) < S
    K_j = tl.load(K_ptr + base_k + row_offset[:, None] * stride_s + col_offset[None, :] * stride_d, 
                  mask=mask_k[:, None], other=0.0).to(tl.float32)
    
    base_v = bid * stride_h + j * BLOCK_N * stride_s
    V_j = tl.load(V_ptr + base_v + row_offset[:, None] * stride_s + col_offset[None, :] * stride_d, 
                  mask=mask_k[:, None], other=0.0).to(tl.float32)
    
    acc_dK = tl.zeros((BLOCK_N, 128), tl.float32)
    acc_dV = tl.zeros((BLOCK_N, 128), tl.float32)
    
    for i in range(T_r):
        base_q = bid * stride_h + i * BLOCK_N * stride_s
        mask_q = (i * BLOCK_N + row_offset) < S
        Q_i = tl.load(Q_ptr + base_q + row_offset[:, None] * stride_s + col_offset[None, :] * stride_d,
                       mask=mask_q[:, None], other=0.0).to(tl.float32)
        
        base_do = bid * stride_h + i * BLOCK_N * stride_s
        dO_i = tl.load(dO_ptr + base_do + row_offset[:, None] * stride_s + col_offset[None, :] * stride_d,
                        mask=mask_q[:, None], other=0.0).to(tl.float32)
        
        base_l = bid * stride_l_h + i * BLOCK_N
        L_i = tl.load(L_ptr + base_l + row_offset, mask=mask_q, other=1e20)
        D_i_val = tl.load(D_ptr + base_l + row_offset, mask=mask_q, other=0.0)
        
        L_i = L_i[:, None]
        D_i = D_i_val[:, None]
        
        # S = Q_i @ K_j^T
        S_val = tl.dot(Q_i, K_j.T)
        
        # P = exp(S * scale - L_i)
        k_idx = j * BLOCK_N + row_offset
        mask_k_col = k_idx < S
        S_scaled = S_val * scale
        S_masked = tl.where(mask_k_col[None, :], S_scaled, -1e20)
        P = tl.exp(S_masked - L_i)
        
        # dP = dO_i @ V_j^T
        dP = tl.dot(dO_i, V_j.T)
        
        # dS = P * (dP - D_i) * scale
        dS = P * (dP - D_i) * scale
        
        # dV_j += P^T @ dO_i
        acc_dV = tl.dot(P.T, dO_i, acc_dV)
        
        # dK_j += dS^T @ Q_i
        acc_dK = tl.dot(dS.T, Q_i, acc_dK)
    
    # Flush accumulated gradients back to HBM converted to bf16.
    base_dk = bid * stride_h + j * BLOCK_N * stride_s
    mask_store = (j * BLOCK_N + row_offset) < S
    tl.store(dK_ptr + base_dk + row_offset[:, None] * stride_s + col_offset[None, :] * stride_d, 
             acc_dK.to(tl.bfloat16), mask=mask_store[:, None])
    
    tl.store(dV_ptr + base_dk + row_offset[:, None] * stride_s + col_offset[None, :] * stride_d, 
             acc_dV.to(tl.bfloat16), mask=mask_store[:, None])


@triton.jit
def _dq_kernel(
    Q_ptr, dO_ptr, L_ptr, D_ptr, K_ptr, V_ptr, dQ_ptr,
    B, H, S, T_c,
    stride_h, stride_s, stride_d, stride_l_h,
    scale: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    i = tl.program_id(0)
    bid = tl.program_id(1)
    
    row_offset = tl.arange(0, BLOCK_N)
    col_offset = tl.arange(0, 128)
    
    # Load Q and dO once at the start of the program
    base_q = bid * stride_h + i * BLOCK_N * stride_s
    mask_q = (i * BLOCK_N + row_offset) < S
    Q_i = tl.load(Q_ptr + base_q + row_offset[:, None] * stride_s + col_offset[None, :] * stride_d,
                   mask=mask_q[:, None], other=0.0).to(tl.float32)
    
    base_do = bid * stride_h + i * BLOCK_N * stride_s
    dO_i = tl.load(dO_ptr + base_do + row_offset[:, None] * stride_s + col_offset[None, :] * stride_d,
                    mask=mask_q[:, None], other=0.0).to(tl.float32)
    
    base_l = bid * stride_l_h + i * BLOCK_N
    L_i = tl.load(L_ptr + base_l + row_offset, mask=mask_q, other=1e20)
    D_i_val = tl.load(D_ptr + base_l + row_offset, mask=mask_q, other=0.0)
    
    L_i = L_i[:, None]
    D_i = D_i_val[:, None]
    
    acc_dQ = tl.zeros((BLOCK_N, 128), tl.float32)
    
    for j in range(T_c):
        base_k = bid * stride_h + j * BLOCK_N * stride_s
        mask_k = (j * BLOCK_N + row_offset) < S
        K_j = tl.load(K_ptr + base_k + row_offset[:, None] * stride_s + col_offset[None, :] * stride_d,
                       mask=mask_k[:, None], other=0.0).to(tl.float32)
        
        base_v = bid * stride_h + j * BLOCK_N * stride_s
        V_j = tl.load(V_ptr + base_v + row_offset[:, None] * stride_s + col_offset[None, :] * stride_d,
                       mask=mask_k[:, None], other=0.0).to(tl.float32)
        
        # S = Q_i @ K_j^T
        S_val = tl.dot(Q_i, K_j.T)
        
        # P = exp(S * scale - L_i)
        k_idx = j * BLOCK_N + row_offset
        mask_k_col = k_idx < S
        S_scaled = S_val * scale
        S_masked = tl.where(mask_k_col[None, :], S_scaled, -1e20)
        P = tl.exp(S_masked - L_i)
        
        # dP = dO_i @ V_j^T
        dP = tl.dot(dO_i, V_j.T)
        
        # dS = P * (dP - D_i) * scale
        dS = P * (dP - D_i) * scale
        
        # dQ_i += dS @ K_j
        acc_dQ = tl.dot(dS, K_j, acc_dQ)
    
    # Flush accumulated dQ gradients back to HBM converted to bf16.
    base_dq = bid * stride_h + i * BLOCK_N * stride_s
    mask_store = (i * BLOCK_N + row_offset) < S
    tl.store(dQ_ptr + base_dq + row_offset[:, None] * stride_s + col_offset[None, :] * stride_d, 
             acc_dQ.to(tl.bfloat16), mask=mask_store[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    
    # 1. Pre-process per-row invariant required by the backward math
    D = torch.empty((B, H, S), dtype=torch.float32, device=Q.device)
    n_rows = B * H * S
    grid_D = (triton.cdiv(n_rows, 1),)
    _compute_D_kernel[grid_D](dO, O, D, n_rows, d, 1)
    
    scale = 1.0 / (d ** 0.5)
    
    BLOCK_N = 64
    T_r = triton.cdiv(S, BLOCK_N)
    T_c = triton.cdiv(S, BLOCK_N)
    
    stride_h = Q.stride()[1]
    stride_s = Q.stride()[2]
    stride_d = Q.stride()[3]
    stride_l_h = L.stride()[1]
    
    # 2. Execute the dK and dV accumulation pass
    grid_dkdv = (T_c, B * H)
    _dk_dv_kernel[grid_dkdv](
        K, V, Q, dO, L, D, dK, dV,
        B, H, S, T_r,
        stride_h, stride_s, stride_d, stride_l_h,
        scale=scale,
        BLOCK_N=BLOCK_N,
        num_warps=8,
        num_stages=3,
    )
    
    # 3. Execute the dQ accumulation pass
    grid_dq = (T_r, B * H)
    _dq_kernel[grid_dq](
        Q, dO, L, D, K, V, dQ,
        B, H, S, T_c,
        stride_h, stride_s, stride_d, stride_l_h,
        scale=scale,
        BLOCK_N=BLOCK_N,
        num_warps=8,
        num_stages=3,
    )