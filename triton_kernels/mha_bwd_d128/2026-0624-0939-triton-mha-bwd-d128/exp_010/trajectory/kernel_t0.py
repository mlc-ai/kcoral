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
    stride_b, stride_h, stride_s, stride_d,
    scale: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    j = tl.program_id(0)
    bid = tl.program_id(1)
    b = bid // H
    h = bid % H
    
    row_offset = tl.arange(0, BLOCK_N)
    col_offset = tl.arange(0, 64)
    
    # Load K and V at the beginning to allow SMEM reuse across the Q-iteration loop
    base_k = b * stride_b + h * stride_h + j * BLOCK_N * stride_s
    mask_k_rows = (j * BLOCK_N + row_offset) < S
    K_j_l = tl.load(K_ptr + base_k + row_offset[:, None] * stride_s + col_offset[None, :] * stride_d, 
                     mask=mask_k_rows[:, None], other=0.0)
    K_j_r = tl.load(K_ptr + base_k + row_offset[:, None] * stride_s + (col_offset + 64)[None, :] * stride_d, 
                     mask=mask_k_rows[:, None], other=0.0)
    
    base_v = b * stride_b + h * stride_h + j * BLOCK_N * stride_s
    V_j_l = tl.load(V_ptr + base_v + row_offset[:, None] * stride_s + col_offset[None, :] * stride_d, 
                     mask=mask_k_rows[:, None], other=0.0)
    V_j_r = tl.load(V_ptr + base_v + row_offset[:, None] * stride_s + (col_offset + 64)[None, :] * stride_d, 
                     mask=mask_k_rows[:, None], other=0.0)
    
    acc_dK_l = tl.zeros((BLOCK_N, 64), tl.float32)
    acc_dK_r = tl.zeros((BLOCK_N, 64), tl.float32)
    acc_dV_l = tl.zeros((BLOCK_N, 64), tl.float32)
    acc_dV_r = tl.zeros((BLOCK_N, 64), tl.float32)
    
    for i in range(T_r):
        base_q = b * stride_b + h * stride_h + i * BLOCK_N * stride_s
        mask_q_rows = (i * BLOCK_N + row_offset) < S
        Q_i_l = tl.load(Q_ptr + base_q + row_offset[:, None] * stride_s + col_offset[None, :] * stride_d,
                         mask=mask_q_rows[:, None], other=0.0)
        Q_i_r = tl.load(Q_ptr + base_q + row_offset[:, None] * stride_s + (col_offset + 64)[None, :] * stride_d,
                         mask=mask_q_rows[:, None], other=0.0)
        
        base_do = b * stride_b + h * stride_h + i * BLOCK_N * stride_s
        dO_i_l = tl.load(dO_ptr + base_do + row_offset[:, None] * stride_s + col_offset[None, :] * stride_d,
                          mask=mask_q_rows[:, None], other=0.0)
        dO_i_r = tl.load(dO_ptr + base_do + row_offset[:, None] * stride_s + (col_offset + 64)[None, :] * stride_d,
                          mask=mask_q_rows[:, None], other=0.0)
        
        base_l = b * (H * S) + h * S + i * BLOCK_N
        L_i = tl.load(L_ptr + base_l + row_offset, mask=mask_q_rows, other=1e20)
        D_i_val = tl.load(D_ptr + base_l + row_offset, mask=mask_q_rows, other=0.0)
        
        L_i = L_i[:, None]
        D_i = D_i_val[:, None]
        
        # S = Q_i @ K_j^T
        S = tl.dot(Q_i_l, K_j_l.T) + tl.dot(Q_i_r, K_j_r.T)
        
        # P = exp(S * scale - L_i)
        S_scaled = S * scale
        k_idx = j * BLOCK_N + col_offset
        mask_k = k_idx < S
        S_masked = tl.where(mask_k[None, :], S_scaled, -1e20)
        P = tl.exp(S_masked - L_i)
        
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
    base_dk = b * stride_b + h * stride_h + j * BLOCK_N * stride_s
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
    Q_ptr, dO_ptr, L_ptr, D_ptr, K_ptr, V_ptr, dQ_ptr,
    B, H, S, T_c,
    stride_b, stride_h, stride_s, stride_d,
    scale: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    i = tl.program_id(0)
    bid = tl.program_id(1)
    b = bid // H
    h = bid % H
    
    row_offset = tl.arange(0, BLOCK_N)
    col_offset = tl.arange(0, 64)
    
    # Load Q and dO once at the start of the program
    base_q = b * stride_b + h * stride_h + i * BLOCK_N * stride_s
    mask_q_rows = (i * BLOCK_N + row_offset) < S
    Q_i_l = tl.load(Q_ptr + base_q + row_offset[:, None] * stride_s + col_offset[None, :] * stride_d,
                     mask=mask_q_rows[:, None], other=0.0)
    Q_i_r = tl.load(Q_ptr + base_q + row_offset[:, None] * stride_s + (col_offset + 64)[None, :] * stride_d,
                     mask=mask_q_rows[:, None], other=0.0)
    
    base_do = b * stride_b + h * stride_h + i * BLOCK_N * stride_s
    dO_i_l = tl.load(dO_ptr + base_do + row_offset[:, None] * stride_s + col_offset[None, :] * stride_d,
                      mask=mask_q_rows[:, None], other=0.0)
    dO_i_r = tl.load(dO_ptr + base_do + row_offset[:, None] * stride_s + (col_offset + 64)[None, :] * stride_d,
                      mask=mask_q_rows[:, None], other=0.0)
    
    # Load invariant per-row normalization state for this Q-tile
    base_l = b * (H * S) + h * S + i * BLOCK_N
    L_i = tl.load(L_ptr + base_l + row_offset, mask=mask_q_rows, other=1e20)
    D_i_val = tl.load(D_ptr + base_l + row_offset, mask=mask_q_rows, other=0.0)
    
    L_i = L_i[:, None]
    D_i = D_i_val[:, None]
    
    acc_dQ_l = tl.zeros((BLOCK_N, 64), tl.float32)
    acc_dQ_r = tl.zeros((BLOCK_N, 64), tl.float32)
    
    for j in range(T_c):
        base_k = b * stride_b + h * stride_h + j * BLOCK_N * stride_s
        mask_k_rows = (j * BLOCK_N + row_offset) < S
        K_j_l = tl.load(K_ptr + base_k + row_offset[:, None] * stride_s + col_offset[None, :] * stride_d,
                         mask=mask_k_rows[:, None], other=0.0)
        K_j_r = tl.load(K_ptr + base_k + row_offset[:, None] * stride_s + (col_offset + 64)[None, :] * stride_d,
                         mask=mask_k_rows[:, None], other=0.0)
        
        base_v = b * stride_b + h * stride_h + j * BLOCK_N * stride_s
        V_j_l = tl.load(V_ptr + base_v + row_offset[:, None] * stride_s + col_offset[None, :] * stride_d,
                         mask=mask_k_rows[:, None], other=0.0)
        V_j_r = tl.load(V_ptr + base_v + row_offset[:, None] * stride_s + (col_offset + 64)[None, :] * stride_d,
                         mask=mask_k_rows[:, None], other=0.0)
        
        # S = Q_i @ K_j^T
        S = tl.dot(Q_i_l, K_j_l.T) + tl.dot(Q_i_r, K_j_r.T)
        
        # P = exp(S * scale - L_i)
        S_scaled = S * scale
        k_idx = j * BLOCK_N + col_offset
        mask_k = k_idx < S
        S_masked = tl.where(mask_k[None, :], S_scaled, -1e20)
        P = tl.exp(S_masked - L_i)
        
        # dP = dO_i @ V_j^T
        dP = tl.dot(dO_i_l, V_j_l.T) + tl.dot(dO_i_r, V_j_r.T)
        
        # dS = P * (dP - D_i) * scale
        dS = P * (dP - D_i) * scale
        
        # dQ_i += dS @ K_j
        acc_dQ_l = tl.dot(dS, K_j_l, acc_dQ_l)
        acc_dQ_r = tl.dot(dS, K_j_r, acc_dQ_r)
    
    # Flush accumulated dQ gradients back to HBM converted to bf16.
    base_dq = b * stride_b + h * stride_h + i * BLOCK_N * stride_s
    mask_store = (i * BLOCK_N + row_offset) < S
    tl.store(dQ_ptr + base_dq + row_offset[:, None] * stride_s + col_offset[None, :] * stride_d, 
             acc_dQ_l.to(tl.bfloat16), mask=mask_store[:, None])
    tl.store(dQ_ptr + base_dq + row_offset[:, None] * stride_s + (col_offset + 64)[None, :] * stride_d, 
             acc_dQ_r.to(tl.bfloat16), mask=mask_store[:, None])


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
    
    stride_b = Q.stride()[0]
    stride_h = Q.stride()[1]
    stride_s = Q.stride()[2]
    stride_d = Q.stride()[3]
    
    # 2. Execute the dK and dV accumulation pass
    grid_dkdv = (T_c, B * H)
    _dk_dv_kernel[grid_dkdv](
        K, V, Q, dO, L, D, dK, dV,
        B, H, S, T_r,
        stride_b, stride_h, stride_s, stride_d,
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
        stride_b, stride_h, stride_s, stride_d,
        scale=scale,
        BLOCK_N=BLOCK_N,
        num_warps=8,
        num_stages=3,
    )