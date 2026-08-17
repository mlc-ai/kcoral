import torch
import triton
import triton.language as tl


@triton.jit
def _bwd_dKdV_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, O_ptr, L_ptr, dK_ptr, dV_ptr,
    s, tau,
):
    batch_head_idx = tl.program_id(1)
    j = tl.program_id(0)
    
    row_idx = tl.arange(0, 128)
    col_idx_0 = tl.arange(0, 64)
    col_idx_1 = tl.arange(64, 128)
    
    k_base = (batch_head_idx * s * 128 + j * 128 * 128).to(tl.int64)
    K_j_0 = tl.load(K_ptr + k_base + row_idx[:, None] * 128 + col_idx_0[None, :], mask=(row_idx[:, None] < s), other=0.0)
    K_j_1 = tl.load(K_ptr + k_base + row_idx[:, None] * 128 + col_idx_1[None, :], mask=(row_idx[:, None] < s), other=0.0)
    
    v_base = (batch_head_idx * s * 128 + j * 128 * 128).to(tl.int64)
    V_j_0 = tl.load(V_ptr + v_base + row_idx[:, None] * 128 + col_idx_0[None, :], mask=(row_idx[:, None] < s), other=0.0)
    V_j_1 = tl.load(V_ptr + v_base + row_idx[:, None] * 128 + col_idx_1[None, :], mask=(row_idx[:, None] < s), other=0.0)
    
    dk_0 = tl.zeros((128, 64), dtype=tl.float32)
    dk_1 = tl.zeros((128, 64), dtype=tl.float32)
    dv_0 = tl.zeros((128, 64), dtype=tl.float32)
    dv_1 = tl.zeros((128, 64), dtype=tl.float32)
    
    num_s_blocks = tl.cdiv(s, 128)
    
    for i in range(j, num_s_blocks):
        q_base = (batch_head_idx * s * 128 + i * 128 * 128).to(tl.int64)
        Q_i_0 = tl.load(Q_ptr + q_base + row_idx[:, None] * 128 + col_idx_0[None, :], mask=(row_idx[:, None] < s), other=0.0)
        Q_i_1 = tl.load(Q_ptr + q_base + row_idx[:, None] * 128 + col_idx_1[None, :], mask=(row_idx[:, None] < s), other=0.0)
        
        do_base = (batch_head_idx * s * 128 + i * 128 * 128).to(tl.int64)
        dO_i_0 = tl.load(dO_ptr + do_base + row_idx[:, None] * 128 + col_idx_0[None, :], mask=(row_idx[:, None] < s), other=0.0)
        dO_i_1 = tl.load(dO_ptr + do_base + row_idx[:, None] * 128 + col_idx_1[None, :], mask=(row_idx[:, None] < s), other=0.0)
        
        o_base = (batch_head_idx * s * 128 + i * 128 * 128).to(tl.int64)
        O_i_0 = tl.load(O_ptr + o_base + row_idx[:, None] * 128 + col_idx_0[None, :], mask=(row_idx[:, None] < s), other=0.0)
        O_i_1 = tl.load(O_ptr + o_base + row_idx[:, None] * 128 + col_idx_1[None, :], mask=(row_idx[:, None] < s), other=0.0)
        
        D_i = tl.sum(dO_i_0 * O_i_0 + dO_i_1 * O_i_1, axis=1)
        
        r_idx = (i * 128 + row_idx).to(tl.int32)
        L_i = tl.load(L_ptr + batch_head_idx * s + r_idx, mask=(r_idx < s), other=0.0)
        
        S_0 = tl.dot(Q_i_0, K_j_0.T)
        S_1 = tl.dot(Q_i_1, K_j_1.T)
        S = S_0 + S_1
        
        c_idx = (j * 128 + row_idx).to(tl.int32)
        mask = (r_idx[None, :] >= c_idx[:, None]) & (r_idx[None, :] < s) & (c_idx[:, None] < s)
        
        P_unmasked = tl.exp(S * tau - L_i.unsqueeze(-1))
        P = tl.where(mask, P_unmasked, 0.0)
        
        dP_0 = tl.dot(dO_i_0, V_j_0.T)
        dP_1 = tl.dot(dO_i_1, V_j_1.T)
        dP = dP_0 + dP_1
        
        dS = P * (dP - D_i.unsqueeze(-1)) * tau
        
        p_bf16 = P.to(tl.bfloat16)
        ds_bf16 = dS.to(tl.bfloat16)
        
        dv_0 += tl.dot(p_bf16.T, dO_i_0)
        dv_1 += tl.dot(p_bf16.T, dO_i_1)
        
        dk_0 += tl.dot(ds_bf16.T, Q_i_0)
        dk_1 += tl.dot(ds_bf16.T, Q_i_1)
    
    r_idx_store = (j * 128 + row_idx).to(tl.int32)
    base_dk = dK_ptr + batch_head_idx * s * 128 + j * 128 * 128
    base_dv = dV_ptr + batch_head_idx * s * 128 + j * 128 * 128
    
    mask_store = (r_idx_store[:, None] < s)
    
    tl.store(base_dk + row_idx[:, None] * 128 + col_idx_0[None, :], dk_0.to(tl.bfloat16), mask=mask_store)
    tl.store(base_dk + row_idx[:, None] * 128 + col_idx_1[None, :], dk_1.to(tl.bfloat16), mask=mask_store)
    tl.store(base_dv + row_idx[:, None] * 128 + col_idx_0[None, :], dv_0.to(tl.bfloat16), mask=mask_store)
    tl.store(base_dv + row_idx[:, None] * 128 + col_idx_1[None, :], dv_1.to(tl.bfloat16), mask=mask_store)


@triton.jit
def _bwd_dQ_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, O_ptr, L_ptr, dQ_ptr,
    s, tau,
):
    batch_head_idx = tl.program_id(1)
    i = tl.program_id(0)
    
    row_idx = tl.arange(0, 128)
    col_idx_0 = tl.arange(0, 64)
    col_idx_1 = tl.arange(64, 128)
    
    q_base = (batch_head_idx * s * 128 + i * 128 * 128).to(tl.int64)
    Q_i_0 = tl.load(Q_ptr + q_base + row_idx[:, None] * 128 + col_idx_0[None, :], mask=(row_idx[:, None] < s), other=0.0)
    Q_i_1 = tl.load(Q_ptr + q_base + row_idx[:, None] * 128 + col_idx_1[None, :], mask=(row_idx[:, None] < s), other=0.0)
    
    do_base = (batch_head_idx * s * 128 + i * 128 * 128).to(tl.int64)
    dO_i_0 = tl.load(dO_ptr + do_base + row_idx[:, None] * 128 + col_idx_0[None, :], mask=(row_idx[:, None] < s), other=0.0)
    dO_i_1 = tl.load(dO_ptr + do_base + row_idx[:, None] * 128 + col_idx_1[None, :], mask=(row_idx[:, None] < s), other=0.0)
    
    o_base = (batch_head_idx * s * 128 + i * 128 * 128).to(tl.int64)
    O_i_0 = tl.load(O_ptr + o_base + row_idx[:, None] * 128 + col_idx_0[None, :], mask=(row_idx[:, None] < s), other=0.0)
    O_i_1 = tl.load(O_ptr + o_base + row_idx[:, None] * 128 + col_idx_1[None, :], mask=(row_idx[:, None] < s), other=0.0)
    
    D_i = tl.sum(dO_i_0 * O_i_0 + dO_i_1 * O_i_1, axis=1)
    
    r_idx = (i * 128 + row_idx).to(tl.int32)
    L_i = tl.load(L_ptr + batch_head_idx * s + r_idx, mask=(r_idx < s), other=0.0)
    
    dq_0 = tl.zeros((128, 64), dtype=tl.float32)
    dq_1 = tl.zeros((128, 64), dtype=tl.float32)
    
    for j in range(0, i + 1):
        k_base = (batch_head_idx * s * 128 + j * 128 * 128).to(tl.int64)
        K_j_0 = tl.load(K_ptr + k_base + row_idx[:, None] * 128 + col_idx_0[None, :], mask=(row_idx[:, None] < s), other=0.0)
        K_j_1 = tl.load(K_ptr + k_base + row_idx[:, None] * 128 + col_idx_1[None, :], mask=(row_idx[:, None] < s), other=0.0)
        
        v_base = (batch_head_idx * s * 128 + j * 128 * 128).to(tl.int64)
        V_j_0 = tl.load(V_ptr + v_base + row_idx[:, None] * 128 + col_idx_0[None, :], mask=(row_idx[:, None] < s), other=0.0)
        V_j_1 = tl.load(V_ptr + v_base + row_idx[:, None] * 128 + col_idx_1[None, :], mask=(row_idx[:, None] < s), other=0.0)
        
        S_0 = tl.dot(Q_i_0, K_j_0.T)
        S_1 = tl.dot(Q_i_1, K_j_1.T)
        S = S_0 + S_1
        
        c_idx = (j * 128 + row_idx).to(tl.int32)
        mask = (r_idx[None, :] >= c_idx[:, None]) & (r_idx[None, :] < s) & (c_idx[:, None] < s)
        
        P_unmasked = tl.exp(S * tau - L_i.unsqueeze(-1))
        P = tl.where(mask, P_unmasked, 0.0)
        
        dP_0 = tl.dot(dO_i_0, V_j_0.T)
        dP_1 = tl.dot(dO_i_1, V_j_1.T)
        dP = dP_0 + dP_1
        
        dS = P * (dP - D_i.unsqueeze(-1)) * tau
        
        ds_bf16 = dS.to(tl.bfloat16)
        
        dq_0 += tl.dot(ds_bf16, K_j_0)
        dq_1 += tl.dot(ds_bf16, K_j_1)
    
    base_dq = dQ_ptr + batch_head_idx * s * 128 + i * 128 * 128
    mask_store = (r_idx[:, None] < s)
    
    tl.store(base_dq + row_idx[:, None] * 128 + col_idx_0[None, :], dq_0.to(tl.bfloat16), mask=mask_store)
    tl.store(base_dq + row_idx[:, None] * 128 + col_idx_1[None, :], dq_1.to(tl.bfloat16), mask=mask_store)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    b, h, s, d = Q.shape
    
    if s == 0:
        return
    
    tau = 1.0 / (d ** 0.5)
    
    grid_KV = (triton.cdiv(s, 128), b * h)
    _bwd_dKdV_kernel[grid_KV](
        Q, K, V, dO, O, L, dK, dV,
        s, tau,
        num_warps=8,
        maxnreg=128,
    )
    
    grid_Q = (triton.cdiv(s, 128), b * h)
    _bwd_dQ_kernel[grid_Q](
        Q, K, V, dO, O, L, dQ,
        s, tau,
        num_warps=8,
        maxnreg=128,
    )