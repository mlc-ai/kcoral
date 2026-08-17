import torch
import triton
import triton.language as tl


@triton.jit
def _compute_D_kernel(
    dO_ptr,
    O_ptr,
    D_ptr,
    N,  
    HEAD_DIM,
    BLOCK: tl.constexpr,
):
    base_idx = tl.program_id(0) * BLOCK
    idx = base_idx + tl.arange(0, BLOCK)
    mask = idx < N
    
    sum_val = tl.zeros((BLOCK,), tl.float32)
    for chunk in range(0, HEAD_DIM, 64):
        off = idx[:, None] * HEAD_DIM + chunk + tl.arange(0, 64)[None, :]
        val_do = tl.load(dO_ptr + off, mask=(idx[:, None] < N), other=0.0)
        val_o = tl.load(O_ptr + off, mask=(idx[:, None] < N), other=0.0)
        sum_val += (val_do * val_o).sum(axis=1)
    
    tl.store(D_ptr + idx, sum_val, mask=mask)


@triton.jit
def _bwd_Q_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, D_ptr, dQ_ptr,
    B, H, S, scale,
    BLOCK_Q: tl.constexpr, BLOCK_KV: tl.constexpr,
):
    b_h_idx = tl.program_id(1)
    q_idx_base = tl.program_id(0) * BLOCK_Q
    
    row_offs = tl.arange(0, BLOCK_Q)
    s_offs = q_idx_base + row_offs
    mask_q = s_offs < S
    
    d_offs = tl.arange(0, 128)
    load_offs = (b_h_idx * S + s_offs[:, None]) * 128 + d_offs[None, :]
    
    Q_half = tl.load(Q_ptr + load_offs, mask=mask_q[:, None], other=0.0)
    Q_half = tl.cast(Q_half, tl.float32)
    
    dO_half = tl.load(dO_ptr + load_offs, mask=mask_q[:, None], other=0.0)
    dO_half = tl.cast(dO_half, tl.float32)
    
    l_offs = b_h_idx * S + s_offs
    L_val = tl.load(L_ptr + l_offs, mask=mask_q, other=0.0)
    D_val = tl.load(D_ptr + l_offs, mask=mask_q, other=0.0)
    
    acc_dQ = tl.zeros((BLOCK_Q, 128), tl.float32)
    
    for kv_idx_base in range(0, S, BLOCK_KV):
        kv_offs = kv_idx_base + row_offs
        mask_kv = kv_offs < S
        
        k_load_offs = (b_h_idx * S + kv_offs[:, None]) * 128 + d_offs[None, :]
        K_half = tl.load(K_ptr + k_load_offs, mask=mask_kv[:, None], other=0.0)
        K_half = tl.cast(K_half, tl.float32)
        
        v_load_offs = (b_h_idx * S + kv_offs[:, None]) * 128 + d_offs[None, :]
        V_half = tl.load(V_ptr + v_load_offs, mask=mask_kv[:, None], other=0.0)
        V_half = tl.cast(V_half, tl.float32)
        
        S = tl.dot(Q_half, K_half.T) * scale
        P = tl.exp(S - L_val[:, None])
        
        dP = tl.dot(dO_half, V_half.T)
        dS = P * (dP - D_val[:, None]) * scale
        
        acc_dQ += tl.dot(dS, K_half)
        
    store_offs = (b_h_idx * S + s_offs[:, None]) * 128 + d_offs[None, :]
    tl.store(dQ_ptr + store_offs, acc_dQ.to(tl.bfloat16), mask=mask_q[:, None])


@triton.jit
def _bwd_KV_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, D_ptr, dK_ptr, dV_ptr,
    B, H, S, scale,
    BLOCK_Q: tl.constexpr, BLOCK_KV: tl.constexpr,
):
    b_h_idx = tl.program_id(1)
    kv_idx_base = tl.program_id(0) * BLOCK_KV
    
    row_offs = tl.arange(0, BLOCK_KV)
    s_offs = kv_idx_base + row_offs
    mask_kv = s_offs < S
    
    d_offs = tl.arange(0, 128)
    load_offs = (b_h_idx * S + s_offs[:, None]) * 128 + d_offs[None, :]
    K_half = tl.load(K_ptr + load_offs, mask=mask_kv[:, None], other=0.0)
    K_half = tl.cast(K_half, tl.float32)
    
    V_half = tl.load(V_ptr + load_offs, mask=mask_kv[:, None], other=0.0)
    V_half = tl.cast(V_half, tl.float32)
    
    acc_dK = tl.zeros((BLOCK_KV, 128), tl.float32)
    acc_dV = tl.zeros((BLOCK_KV, 128), tl.float32)
    
    for q_idx_base in range(0, S, BLOCK_Q):
        q_row_offs = tl.arange(0, BLOCK_Q)
        q_s_offs = q_idx_base + q_row_offs
        mask_q = q_s_offs < S
        
        q_load_offs = (b_h_idx * S + q_s_offs[:, None]) * 128 + d_offs[None, :]
        Q_half = tl.load(Q_ptr + q_load_offs, mask=mask_q[:, None], other=0.0)
        Q_half = tl.cast(Q_half, tl.float32)
        
        dO_half = tl.load(dO_ptr + q_load_offs, mask=mask_q[:, None], other=0.0)
        dO_half = tl.cast(dO_half, tl.float32)
        
        l_offs = b_h_idx * S + q_s_offs
        L_val = tl.load(L_ptr + l_offs, mask=mask_q, other=0.0)
        D_val = tl.load(D_ptr + l_offs, mask=mask_q, other=0.0)
        
        S = tl.dot(Q_half, K_half.T) * scale
        P = tl.exp(S - L_val[None, :])
        
        dP = tl.dot(dO_half, V_half.T)
        dS = P * (dP - D_val[None, :]) * scale
        
        dS_T = dS.T
        P_T = P.T
        
        acc_dK += tl.dot(dS_T, Q_half)
        acc_dV += tl.dot(P_T, dO_half)
        
    store_offs = (b_h_idx * S + s_offs[:, None]) * 128 + d_offs[None, :]
    tl.store(dK_ptr + store_offs, acc_dK.to(tl.bfloat16), mask=mask_kv[:, None])
    tl.store(dV_ptr + store_offs, acc_dV.to(tl.bfloat16), mask=mask_kv[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Execute optimized multi-head attention backward pass."""
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    scale = 1.0 / (d ** 0.5)
    
    D = torch.empty((B, H, S), dtype=torch.float32, device=Q.device)
    
    grid_D = (triton.cdiv(B * H * S, 64),)
    _compute_D_kernel[grid_D](
        dO, O, D, B * H * S, 128, BLOCK=64
    )
    
    grid = (triton.cdiv(S, 128), B * H)
    
    _bwd_Q_kernel[grid](
        Q, K, V, dO, L, D, dQ,
        B, H, S, scale,
        BLOCK_Q=128, BLOCK_KV=128,
        num_warps=8, num_stages=3
    )
    
    _bwd_KV_kernel[grid](
        Q, K, V, dO, L, D, dK, dV,
        B, H, S, scale,
        BLOCK_Q=128, BLOCK_KV=128,
        num_warps=8, num_stages=3
    )