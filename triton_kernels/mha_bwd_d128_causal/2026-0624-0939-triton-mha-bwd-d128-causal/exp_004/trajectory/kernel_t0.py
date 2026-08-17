import torch
import triton
import triton.language as tl
import math


@triton.jit
def _compute_D_kernel(
    dO_ptr,
    O_ptr,
    D_ptr,
    S,
    d,
    BLOCK_D: tl.constexpr,
):
    row_idx = tl.program_id(0)
    chunk = tl.program_id(2)
    
    lane_ids = tl.arange(0, 32)
    offsets = chunk * 32 + lane_ids
    idx = row_idx * d + offsets
    
    do_val = tl.load(dO_ptr + idx)
    o_val = tl.load(O_ptr + idx)
    
    sum_val = tl.sum(do_val * o_val)
    
    # Cross-lane reduction via atomic add
    if chunk == 0:
        # Direct store for first chunk
        pass
    
    # Use atomic add to combine partial sums
    if lane_ids[0] == 0:
        if chunk == 0:
            # Reset to 0 first if needed, but atomic add from 0 is fine if initialized to 0
            pass
        # Atomic add the partial sum
        tl.atomic_add(D_ptr + row_idx, sum_val)


@triton.jit
def _bwd_dKdV_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    dO_ptr,
    L_ptr,
    D_ptr,
    dK_ptr,
    dV_ptr,
    S,
    H,
    d,
    tau,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    j = tl.program_id(2)
    
    T_r = tl.cdiv(S, BLOCK_N)
    
    if j >= T_r:
        return
    
    batch_offset = (b * H + h) * S * d
    l_offset = (b * H + h) * S
    
    dK_acc = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)
    dV_acc = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)
    
    rows_k = tl.arange(0, BLOCK_N)
    cols_d = tl.arange(0, BLOCK_D)
    
    k_abs = j * BLOCK_N + rows_k
    K_j = tl.load(K_ptr + batch_offset + k_abs[:, None] * 128 + cols_d[None, :], 
                  mask=(k_abs[:, None] < S) & (cols_d[None, :] < d), other=0.0)
    V_j = tl.load(V_ptr + batch_offset + k_abs[:, None] * 128 + cols_d[None, :], 
                  mask=(k_abs[:, None] < S) & (cols_d[None, :] < d), other=0.0)
    
    for i in range(j, T_r):
        rows_q = tl.arange(0, BLOCK_N)
        q_abs = i * BLOCK_N + rows_q
        
        Q_i = tl.load(Q_ptr + batch_offset + q_abs[:, None] * 128 + cols_d[None, :], 
                      mask=(q_abs[:, None] < S) & (cols_d[None, :] < d), other=0.0)
        dO_i = tl.load(dO_ptr + batch_offset + q_abs[:, None] * 128 + cols_d[None, :], 
                       mask=(q_abs[:, None] < S) & (cols_d[None, :] < d), other=0.0)
        
        L_i = tl.load(L_ptr + l_offset + q_abs, mask=q_abs < S, other=0.0)
        D_i = tl.load(D_ptr + l_offset + q_abs, mask=q_abs < S, other=0.0)
        
        S_ij = tl.dot(Q_i, K_j.T) * tau
        if i == j:
            rows = rows_q[:, None]
            cols = tl.arange(0, BLOCK_N)[None, :]
            causal_mask = cols <= rows
            S_ij = tl.where(causal_mask, S_ij, -float('inf'))
        
        P_ij = tl.exp(S_ij - L_i[:, None])
        if i == j:
            P_ij = tl.where(causal_mask, P_ij, 0.0)
        
        dP_ij = tl.dot(dO_i, V_j.T)
        dS_ij = P_ij * (dP_ij - D_i[:, None]) * tau
        if i == j:
            dS_ij = tl.where(causal_mask, dS_ij, 0.0)
        
        dV_acc = tl.dot(P_ij.T, dO_i, acc=dV_acc)
        dK_acc = tl.dot(dS_ij.T, Q_i, acc=dK_acc)
    
    out_K = dK_ptr + batch_offset + k_abs[:, None] * 128 + cols_d[None, :]
    out_V = dV_ptr + batch_offset + k_abs[:, None] * 128 + cols_d[None, :]
    
    mask_k = (k_abs[:, None] < S) & (cols_d[None, :] < d)
    tl.store(out_K, dK_acc, mask=mask_k)
    tl.store(out_V, dV_acc, mask=mask_k)


@triton.jit
def _bwd_dQ_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    dO_ptr,
    L_ptr,
    D_ptr,
    dQ_ptr,
    S,
    H,
    d,
    tau,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    i = tl.program_id(2)
    
    T_c = tl.cdiv(S, BLOCK_N)
    
    if i >= T_c:
        return
    
    batch_offset = (b * H + h) * S * d
    l_offset = (b * H + h) * S
    
    dQ_acc = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)
    
    rows_q = tl.arange(0, BLOCK_N)
    cols_d = tl.arange(0, BLOCK_D)
    
    q_abs = i * BLOCK_N + rows_q
    Q_i = tl.load(Q_ptr + batch_offset + q_abs[:, None] * 128 + cols_d[None, :], 
                  mask=(q_abs[:, None] < S) & (cols_d[None, :] < d), other=0.0)
    dO_i = tl.load(dO_ptr + batch_offset + q_abs[:, None] * 128 + cols_d[None, :], 
                   mask=(q_abs[:, None] < S) & (cols_d[None, :] < d), other=0.0)
    
    L_i = tl.load(L_ptr + l_offset + q_abs, mask=q_abs < S, other=0.0)
    D_i = tl.load(D_ptr + l_offset + q_abs, mask=q_abs < S, other=0.0)
    
    for j in range(0, i + 1):
        rows_k = tl.arange(0, BLOCK_N)
        k_abs = j * BLOCK_N + rows_k
        
        K_j = tl.load(K_ptr + batch_offset + k_abs[:, None] * 128 + cols_d[None, :], 
                      mask=(k_abs[:, None] < S) & (cols_d[None, :] < d), other=0.0)
        V_j = tl.load(V_ptr + batch_offset + k_abs[:, None] * 128 + cols_d[None, :], 
                      mask=(k_abs[:, None] < S) & (cols_d[None, :] < d), other=0.0)
        
        S_ij = tl.dot(Q_i, K_j.T) * tau
        if i == j:
            rows = rows_q[:, None]
            cols = rows_k[None, :]
            causal_mask = cols <= rows
            S_ij = tl.where(causal_mask, S_ij, -float('inf'))
        
        P_ij = tl.exp(S_ij - L_i[:, None])
        if i == j:
            P_ij = tl.where(causal_mask, P_ij, 0.0)
        
        dP_ij = tl.dot(dO_i, V_j.T)
        dS_ij = P_ij * (dP_ij - D_i[:, None]) * tau
        if i == j:
            dS_ij = tl.where(causal_mask, dS_ij, 0.0)
        
        dQ_acc = tl.dot(dS_ij, K_j, acc=dQ_acc)
    
    out_Q = dQ_ptr + batch_offset + q_abs[:, None] * 128 + cols_d[None, :]
    mask_q = (q_abs[:, None] < S) & (cols_d[None, :] < d)
    tl.store(out_Q, dQ_acc, mask=mask_q)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    tau = 1.0 / math.sqrt(d)
    
    T_r = (S + 63) // 64
    
    D_ptr = torch.empty((B, H, S), dtype=torch.float32, device=Q.device)
    
    grid_D = (B * H * S, 1, d // 32)
    _compute_D_kernel[grid_D](dO, O, D_ptr, S, d, BLOCK_D=128)
    
    grid = (B, H, T_r)
    _bwd_dKdV_kernel[grid](
        Q, K, V, O, dO, L, D_ptr, dK, dV, S, H, d, tau,
        BLOCK_D=128, BLOCK_N=64, num_warps=4, num_stages=3
    )
    _bwd_dQ_kernel[grid](
        Q, K, V, O, dO, L, D_ptr, dQ, S, H, d, tau,
        BLOCK_D=128, BLOCK_N=64, num_warps=4, num_stages=3
    )