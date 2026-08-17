import torch
import triton
import triton.language as tl


@triton.jit
def _bwd_Q_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dQ_ptr,
    B, H, S, scale,
    BLOCK_Q: tl.constexpr, BLOCK_KV: tl.constexpr,
):
    b_h_idx = tl.program_id(1)
    q_idx_base = tl.program_id(0) * BLOCK_Q
    
    row_offs = tl.arange(0, BLOCK_Q)
    s_offs = q_idx_base + row_offs
    mask_q = s_offs < S
    
    d_offs_a = tl.arange(0, 64)
    d_offs_b = 64 + tl.arange(0, 64)
    
    load_offs_a = (b_h_idx * S + s_offs[:, None]) * 128 + d_offs_a[None, :]
    Q_half_a = tl.load(Q_ptr + load_offs_a, mask=mask_q[:, None], other=0.0)
    dO_half_a = tl.load(dO_ptr + load_offs_a, mask=mask_q[:, None], other=0.0)
    
    load_offs_b = (b_h_idx * S + s_offs[:, None]) * 128 + d_offs_b[None, :]
    Q_half_b = tl.load(Q_ptr + load_offs_b, mask=mask_q[:, None], other=0.0)
    dO_half_b = tl.load(dO_ptr + load_offs_b, mask=mask_q[:, None], other=0.0)
    
    D_val = tl.zeros((BLOCK_Q,), tl.float32)
    O_half_a = tl.load(O_ptr + load_offs_a, mask=mask_q[:, None], other=0.0)
    D_val += (dO_half_a * O_half_a).sum(axis=1)
    O_half_b = tl.load(O_ptr + load_offs_b, mask=mask_q[:, None], other=0.0)
    D_val += (dO_half_b * O_half_b).sum(axis=1)
    
    l_offs = b_h_idx * S + s_offs
    L_val = tl.load(L_ptr + l_offs, mask=mask_q, other=0.0)
    
    acc_dQ_a = tl.zeros((BLOCK_Q, 64), tl.float32)
    acc_dQ_b = tl.zeros((BLOCK_Q, 64), tl.float32)
    
    for kv_idx_base in range(0, S, BLOCK_KV):
        kv_offs = kv_idx_base + row_offs
        mask_kv = kv_offs < S
        
        k_load_offs_a = (b_h_idx * S + kv_offs[:, None]) * 128 + d_offs_a[None, :]
        K_half_a = tl.load(K_ptr + k_load_offs_a, mask=mask_kv[:, None], other=0.0)
        
        k_load_offs_b = (b_h_idx * S + kv_offs[:, None]) * 128 + d_offs_b[None, :]
        K_half_b = tl.load(K_ptr + k_load_offs_b, mask=mask_kv[:, None], other=0.0)
        
        v_load_offs_a = (b_h_idx * S + kv_offs[:, None]) * 128 + d_offs_a[None, :]
        V_half_a = tl.load(V_ptr + v_load_offs_a, mask=mask_kv[:, None], other=0.0)
        
        v_load_offs_b = (b_h_idx * S + kv_offs[:, None]) * 128 + d_offs_b[None, :]
        V_half_b = tl.load(V_ptr + v_load_offs_b, mask=mask_kv[:, None], other=0.0)
        
        S = (tl.dot(Q_half_a, K_half_a.T) + tl.dot(Q_half_b, K_half_b.T)) * scale
        P = tl.exp(S - L_val[:, None])
        
        dP = (tl.dot(dO_half_a, V_half_a.T) + tl.dot(dO_half_b, V_half_b.T))
        dS = P * (dP - D_val[:, None]) * scale
        
        acc_dQ_a += tl.dot(dS, K_half_a)
        acc_dQ_b += tl.dot(dS, K_half_b)
        
    store_offs_a = (b_h_idx * S + s_offs[:, None]) * 128 + d_offs_a[None, :]
    tl.store(dQ_ptr + store_offs_a, acc_dQ_a.to(tl.bfloat16), mask=mask_q[:, None])
    
    store_offs_b = (b_h_idx * S + s_offs[:, None]) * 128 + d_offs_b[None, :]
    tl.store(dQ_ptr + store_offs_b, acc_dQ_b.to(tl.bfloat16), mask=mask_q[:, None])


@triton.jit
def _bwd_KV_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dK_ptr, dV_ptr,
    B, H, S, scale,
    BLOCK_Q: tl.constexpr, BLOCK_KV: tl.constexpr,
):
    b_h_idx = tl.program_id(1)
    kv_idx_base = tl.program_id(0) * BLOCK_KV
    
    row_offs = tl.arange(0, BLOCK_KV)
    s_offs = kv_idx_base + row_offs
    mask_kv = s_offs < S
    
    d_offs_a = tl.arange(0, 64)
    d_offs_b = 64 + tl.arange(0, 64)
    
    load_offs_a = (b_h_idx * S + s_offs[:, None]) * 128 + d_offs_a[None, :]
    K_half_a = tl.load(K_ptr + load_offs_a, mask=mask_kv[:, None], other=0.0)
    V_half_a = tl.load(V_ptr + load_offs_a, mask=mask_kv[:, None], other=0.0)
    
    load_offs_b = (b_h_idx * S + s_offs[:, None]) * 128 + d_offs_b[None, :]
    K_half_b = tl.load(K_ptr + load_offs_b, mask=mask_kv[:, None], other=0.0)
    V_half_b = tl.load(V_ptr + load_offs_b, mask=mask_kv[:, None], other=0.0)
    
    acc_dK_a = tl.zeros((BLOCK_KV, 64), tl.float32)
    acc_dK_b = tl.zeros((BLOCK_KV, 64), tl.float32)
    acc_dV_a = tl.zeros((BLOCK_KV, 64), tl.float32)
    acc_dV_b = tl.zeros((BLOCK_KV, 64), tl.float32)
    
    for q_idx_base in range(0, S, BLOCK_Q):
        q_row_offs = tl.arange(0, BLOCK_Q)
        q_s_offs = q_idx_base + q_row_offs
        mask_q = q_s_offs < S
        
        q_load_offs_a = (b_h_idx * S + q_s_offs[:, None]) * 128 + d_offs_a[None, :]
        Q_half_a = tl.load(Q_ptr + q_load_offs_a, mask=mask_q[:, None], other=0.0)
        dO_half_a = tl.load(dO_ptr + q_load_offs_a, mask=mask_q[:, None], other=0.0)
        
        q_load_offs_b = (b_h_idx * S + q_s_offs[:, None]) * 128 + d_offs_b[None, :]
        Q_half_b = tl.load(Q_ptr + q_load_offs_b, mask=mask_q[:, None], other=0.0)
        dO_half_b = tl.load(dO_ptr + q_load_offs_b, mask=mask_q[:, None], other=0.0)
        
        D_val = tl.zeros((BLOCK_Q,), tl.float32)
        O_half_a = tl.load(O_ptr + q_load_offs_a, mask=mask_q[:, None], other=0.0)
        D_val += (dO_half_a * O_half_a).sum(axis=1)
        O_half_b = tl.load(O_ptr + q_load_offs_b, mask=mask_q[:, None], other=0.0)
        D_val += (dO_half_b * O_half_b).sum(axis=1)
        
        l_offs = b_h_idx * S + q_s_offs
        L_val = tl.load(L_ptr + l_offs, mask=mask_q, other=0.0)
        
        S = (tl.dot(Q_half_a, K_half_a.T) + tl.dot(Q_half_b, K_half_b.T)) * scale
        P = tl.exp(S - L_val[None, :])
        
        dP = (tl.dot(dO_half_a, V_half_a.T) + tl.dot(dO_half_b, V_half_b.T))
        dS = P * (dP - D_val[None, :]) * scale
        
        dS_T = dS.T
        P_T = P.T
        
        acc_dK_a += tl.dot(dS_T, Q_half_a)
        acc_dK_b += tl.dot(dS_T, Q_half_b)
        acc_dV_a += tl.dot(P_T, dO_half_a)
        acc_dV_b += tl.dot(P_T, dO_half_b)
        
    store_offs_a = (b_h_idx * S + s_offs[:, None]) * 128 + d_offs_a[None, :]
    tl.store(dK_ptr + store_offs_a, acc_dK_a.to(tl.bfloat16), mask=mask_kv[:, None])
    tl.store(dV_ptr + store_offs_a, acc_dV_a.to(tl.bfloat16), mask=mask_kv[:, None])
    
    store_offs_b = (b_h_idx * S + s_offs[:, None]) * 128 + d_offs_b[None, :]
    tl.store(dK_ptr + store_offs_b, acc_dK_b.to(tl.bfloat16), mask=mask_kv[:, None])
    tl.store(dV_ptr + store_offs_b, acc_dV_b.to(tl.bfloat16), mask=mask_kv[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Execute optimized multi-head attention backward pass."""
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    scale = 1.0 / (d ** 0.5)
    
    grid = (triton.cdiv(S, 64), B * H)
    
    _bwd_Q_kernel[grid](
        Q, K, V, O, dO, L, dQ,
        B, H, S, scale,
        BLOCK_Q=64, BLOCK_KV=64,
        num_warps=8, num_stages=3
    )
    
    _bwd_KV_kernel[grid](
        Q, K, V, O, dO, L, dK, dV,
        B, H, S, scale,
        BLOCK_Q=64, BLOCK_KV=64,
        num_warps=8, num_stages=3
    )