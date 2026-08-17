import torch
import triton
import triton.language as tl


@triton.jit
def _bwd_Q_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, dQ_ptr,
    B, H, S_len, scale,
    BLOCK_Q: tl.constexpr, BLOCK_KV: tl.constexpr,
):
    q_idx_base = tl.program_id(0) * BLOCK_Q
    
    row_offs = tl.arange(0, 128)
    
    for b_h in range(B * H):
        s_offs = q_idx_base + row_offs
        mask_q = s_offs < S_len
        
        load_offs_0 = (b_h * S_len + s_offs[:, None]) * 128 + tl.arange(0, 64)[None, :]
        Q_0 = tl.load(Q_ptr + load_offs_0, mask=mask_q[:, None], other=0.0)
        Q_0 = tl.cast(Q_0, tl.float32)
        dO_0 = tl.load(dO_ptr + load_offs_0, mask=mask_q[:, None], other=0.0)
        dO_0 = tl.cast(dO_0, tl.float32)
        O_0 = tl.load(O_ptr + load_offs_0, mask=mask_q[:, None], other=0.0)
        O_0 = tl.cast(O_0, tl.float32)
        
        load_offs_1 = (b_h * S_len + s_offs[:, None]) * 128 + (64 + tl.arange(0, 64))[None, :]
        Q_1 = tl.load(Q_ptr + load_offs_1, mask=mask_q[:, None], other=0.0)
        Q_1 = tl.cast(Q_1, tl.float32)
        dO_1 = tl.load(dO_ptr + load_offs_1, mask=mask_q[:, None], other=0.0)
        dO_1 = tl.cast(dO_1, tl.float32)
        O_1 = tl.load(O_ptr + load_offs_1, mask=mask_q[:, None], other=0.0)
        O_1 = tl.cast(O_1, tl.float32)
        
        D_val = (dO_0 * O_0 + dO_1 * O_1).sum(axis=1)
        
        l_offs = b_h * S_len + s_offs
        L_val = tl.load(L_ptr + l_offs, mask=mask_q, other=0.0)
        
        acc_dQ_0 = tl.zeros((128, 64), tl.float32)
        acc_dQ_1 = tl.zeros((128, 64), tl.float32)
        
        for kv_idx_base in range(0, S_len, 128):
            kv_offs = kv_idx_base + row_offs
            mask_kv = kv_offs < S_len
            
            k_load_offs_0 = (b_h * S_len + kv_offs[:, None]) * 128 + tl.arange(0, 64)[None, :]
            K_0 = tl.load(K_ptr + k_load_offs_0, mask=mask_kv[:, None], other=0.0)
            K_0 = tl.cast(K_0, tl.float32)
            v_load_offs_0 = (b_h * S_len + kv_offs[:, None]) * 128 + tl.arange(0, 64)[None, :]
            V_0 = tl.load(V_ptr + v_load_offs_0, mask=mask_kv[:, None], other=0.0)
            V_0 = tl.cast(V_0, tl.float32)
            
            k_load_offs_1 = (b_h * S_len + kv_offs[:, None]) * 128 + (64 + tl.arange(0, 64))[None, :]
            K_1 = tl.load(K_ptr + k_load_offs_1, mask=mask_kv[:, None], other=0.0)
            K_1 = tl.cast(K_1, tl.float32)
            v_load_offs_1 = (b_h * S_len + kv_offs[:, None]) * 128 + (64 + tl.arange(0, 64))[None, :]
            V_1 = tl.load(V_ptr + v_load_offs_1, mask=mask_kv[:, None], other=0.0)
            V_1 = tl.cast(V_1, tl.float32)
            
            S_scores = (tl.dot(Q_0, K_0.T) + tl.dot(Q_1, K_1.T)) * scale
            P = tl.exp(S_scores - L_val[:, None])
            
            dP = tl.dot(dO_0, V_0.T) + tl.dot(dO_1, V_1.T)
            dS = P * (dP - D_val[:, None]) * scale
            
            acc_dQ_0 += tl.dot(dS, K_0)
            acc_dQ_1 += tl.dot(dS, K_1)
            
        store_offs_0 = (b_h * S_len + s_offs[:, None]) * 128 + tl.arange(0, 64)[None, :]
        tl.store(dQ_ptr + store_offs_0, acc_dQ_0.to(tl.bfloat16), mask=mask_q[:, None])
        store_offs_1 = (b_h * S_len + s_offs[:, None]) * 128 + (64 + tl.arange(0, 64))[None, :]
        tl.store(dQ_ptr + store_offs_1, acc_dQ_1.to(tl.bfloat16), mask=mask_q[:, None])


@triton.jit
def _bwd_KV_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, dK_ptr, dV_ptr,
    B, H, S_len, scale,
    BLOCK_Q: tl.constexpr, BLOCK_KV: tl.constexpr,
):
    kv_idx_base = tl.program_id(0) * BLOCK_KV
    
    row_offs = tl.arange(0, 128)
    
    for b_h in range(B * H):
        s_offs = kv_idx_base + row_offs
        mask_kv = s_offs < S_len
        
        load_offs_0 = (b_h * S_len + s_offs[:, None]) * 128 + tl.arange(0, 64)[None, :]
        K_0 = tl.load(K_ptr + load_offs_0, mask=mask_kv[:, None], other=0.0)
        K_0 = tl.cast(K_0, tl.float32)
        V_0 = tl.load(V_ptr + load_offs_0, mask=mask_kv[:, None], other=0.0)
        V_0 = tl.cast(V_0, tl.float32)
        
        load_offs_1 = (b_h * S_len + s_offs[:, None]) * 128 + (64 + tl.arange(0, 64))[None, :]
        K_1 = tl.load(K_ptr + load_offs_1, mask=mask_kv[:, None], other=0.0)
        K_1 = tl.cast(K_1, tl.float32)
        V_1 = tl.load(V_ptr + load_offs_1, mask=mask_kv[:, None], other=0.0)
        V_1 = tl.cast(V_1, tl.float32)
        
        acc_dK_0 = tl.zeros((128, 64), tl.float32)
        acc_dK_1 = tl.zeros((128, 64), tl.float32)
        acc_dV_0 = tl.zeros((128, 64), tl.float32)
        acc_dV_1 = tl.zeros((128, 64), tl.float32)
        
        for q_idx_base in range(0, S_len, 128):
            q_offs = q_idx_base + row_offs
            mask_q = q_offs < S_len
            
            q_load_offs_0 = (b_h * S_len + q_offs[:, None]) * 128 + tl.arange(0, 64)[None, :]
            Q_0 = tl.load(Q_ptr + q_load_offs_0, mask=mask_q[:, None], other=0.0)
            Q_0 = tl.cast(Q_0, tl.float32)
            dO_0 = tl.load(dO_ptr + q_load_offs_0, mask=mask_q[:, None], other=0.0)
            dO_0 = tl.cast(dO_0, tl.float32)
            O_0 = tl.load(O_ptr + q_load_offs_0, mask=mask_q[:, None], other=0.0)
            O_0 = tl.cast(O_0, tl.float32)
            
            q_load_offs_1 = (b_h * S_len + q_offs[:, None]) * 128 + (64 + tl.arange(0, 64))[None, :]
            Q_1 = tl.load(Q_ptr + q_load_offs_1, mask=mask_q[:, None], other=0.0)
            Q_1 = tl.cast(Q_1, tl.float32)
            dO_1 = tl.load(dO_ptr + q_load_offs_1, mask=mask_q[:, None], other=0.0)
            dO_1 = tl.cast(dO_1, tl.float32)
            O_1 = tl.load(O_ptr + q_load_offs_1, mask=mask_q[:, None], other=0.0)
            O_1 = tl.cast(O_1, tl.float32)
            
            D_val = (dO_0 * O_0 + dO_1 * O_1).sum(axis=1)
            
            l_offs = b_h * S_len + q_offs
            L_val = tl.load(L_ptr + l_offs, mask=mask_q, other=0.0)
            
            S_scores = (tl.dot(Q_0, K_0.T) + tl.dot(Q_1, K_1.T)) * scale
            P = tl.exp(S_scores - L_val[:, None])
            
            dP = tl.dot(dO_0, V_0.T) + tl.dot(dO_1, V_1.T)
            dS = P * (dP - D_val[:, None]) * scale
            
            dS_T = dS.T
            P_T = P.T
            
            acc_dK_0 += tl.dot(dS_T, Q_0)
            acc_dK_1 += tl.dot(dS_T, Q_1)
            acc_dV_0 += tl.dot(P_T, dO_0)
            acc_dV_1 += tl.dot(P_T, dO_1)
            
        store_offs_0 = (b_h * S_len + s_offs[:, None]) * 128 + tl.arange(0, 64)[None, :]
        tl.store(dK_ptr + store_offs_0, acc_dK_0.to(tl.bfloat16), mask=mask_kv[:, None])
        tl.store(dV_ptr + store_offs_0, acc_dV_0.to(tl.bfloat16), mask=mask_kv[:, None])
        
        store_offs_1 = (b_h * S_len + s_offs[:, None]) * 128 + (64 + tl.arange(0, 64))[None, :]
        tl.store(dK_ptr + store_offs_1, acc_dK_1.to(tl.bfloat16), mask=mask_kv[:, None])
        tl.store(dV_ptr + store_offs_1, acc_dV_1.to(tl.bfloat16), mask=mask_kv[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Execute optimized multi-head attention backward pass."""
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    scale = 1.0 / (d ** 0.5)
    
    grid = (triton.cdiv(S, 128),)
    
    _bwd_Q_kernel[grid](
        Q, K, V, dO, L, dQ,
        B, H, S, scale,
        BLOCK_Q=128, BLOCK_KV=128,
        num_warps=8, num_stages=3
    )
    
    _bwd_KV_kernel[grid](
        Q, K, V, dO, L, dK, dV,
        B, H, S, scale,
        BLOCK_Q=128, BLOCK_KV=128,
        num_warps=8, num_stages=3
    )