import torch
import triton
import triton.language as tl


@triton.jit
def _kernel_dVk(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
    dK_ptr, dV_ptr,
    S_len, d_dim, H_total,
    sqrt_d,
    BLOCK: tl.constexpr
):
    j = tl.program_id(0)
    b_h = tl.program_id(1)
    
    num_blocks = tl.cdiv(S_len, BLOCK)
    if j >= num_blocks:
        return
        
    base_offset_bh = b_h * S_len * d_dim
    
    dV_acc = tl.zeros((BLOCK, 128), tl.float32)
    dK_acc = tl.zeros((BLOCK, 128), tl.float32)
    
    cols_d = tl.arange(0, 128)
    
    k_base = base_offset_bh + j * BLOCK * d_dim
    rows_k = tl.arange(0, BLOCK)
    k_ptrs = K_ptr + k_base + rows_k[:, None] * d_dim + cols_d[None, :]
    k_mask = (j * BLOCK + rows_k[:, None]) < S_len
    K_j = tl.load(k_ptrs, mask=k_mask, other=0.0)
    
    v_ptrs = V_ptr + k_base + rows_k[:, None] * d_dim + cols_d[None, :]
    V_j = tl.load(v_ptrs, mask=k_mask, other=0.0)
    
    cols_k = tl.arange(0, BLOCK)
    
    for i in range(j, num_blocks):
        q_base = base_offset_bh + i * BLOCK * d_dim
        o_base = base_offset_bh + i * BLOCK * d_dim
        do_base = base_offset_bh + i * BLOCK * d_dim
        
        rows_q = tl.arange(0, BLOCK)
        q_ptrs = Q_ptr + q_base + rows_q[:, None] * d_dim + cols_d[None, :]
        q_mask = (i * BLOCK + rows_q[:, None]) < S_len
        Q_i = tl.load(q_ptrs, mask=q_mask, other=0.0)
        
        o_ptrs = O_ptr + o_base + rows_q[:, None] * d_dim + cols_d[None, :]
        O_i = tl.load(o_ptrs, mask=q_mask, other=0.0)
        
        do_ptrs = dO_ptr + do_base + rows_q[:, None] * d_dim + cols_d[None, :]
        dO_i = tl.load(do_ptrs, mask=q_mask, other=0.0)
        
        l_base = b_h * S_len + i * BLOCK
        l_ptrs = L_ptr + l_base + rows_q
        l_mask = (i * BLOCK + rows_q) < S_len
        L_i = tl.load(l_ptrs, mask=l_mask, other=0.0)
        
        D_i = tl.sum(dO_i * O_i, axis=1)
        
        S = tl.dot(Q_i, K_j.T)
        
        P = tl.exp(S / sqrt_d - L_i[:, None])
        
        mask = ((j * BLOCK + cols_k[None, :]) <= (i * BLOCK + rows_q[:, None])) & \
               ((i * BLOCK + rows_q[:, None]) < S_len) & \
               ((j * BLOCK + cols_k[None, :]) < S_len)
        P = P * mask
        
        dV_acc = tl.dot(P.T, dO_i, acc=dV_acc)
        
        dS_raw = tl.dot(dO_i, V_j.T)
        dS = P * (dS_raw - D_i[:, None]) / sqrt_d
        
        dK_acc = tl.dot(dS.T, Q_i, acc=dK_acc)
    
    out_k_base = base_offset_bh + j * BLOCK * d_dim
    out_v_base = base_offset_bh + j * BLOCK * d_dim
    
    rows_out = tl.arange(0, BLOCK)
    out_ptrs_k = dK_ptr + out_k_base + rows_out[:, None] * d_dim + cols_d[None, :]
    out_mask_k = (j * BLOCK + rows_out[:, None]) < S_len
    tl.store(out_ptrs_k, dK_acc.to(tl.bfloat16), mask=out_mask_k)
    
    out_ptrs_v = dV_ptr + out_v_base + rows_out[:, None] * d_dim + cols_d[None, :]
    out_mask_v = (j * BLOCK + rows_out[:, None]) < S_len
    tl.store(out_ptrs_v, dV_acc.to(tl.bfloat16), mask=out_mask_v)


@triton.jit
def _kernel_dQ(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
    dQ_ptr,
    S_len, d_dim, H_total,
    sqrt_d,
    BLOCK: tl.constexpr
):
    i = tl.program_id(0)
    b_h = tl.program_id(1)
    
    num_blocks = tl.cdiv(S_len, BLOCK)
    if i >= num_blocks:
        return
        
    base_offset_bh = b_h * S_len * d_dim
    
    q_base = base_offset_bh + i * BLOCK * d_dim
    o_base = base_offset_bh + i * BLOCK * d_dim
    do_base = base_offset_bh + i * BLOCK * d_dim
    
    cols_d = tl.arange(0, 128)
    rows = tl.arange(0, BLOCK)
    
    q_ptrs = Q_ptr + q_base + rows[:, None] * d_dim + cols_d[None, :]
    q_mask = (i * BLOCK + rows[:, None]) < S_len
    Q_i = tl.load(q_ptrs, mask=q_mask, other=0.0)
    
    o_ptrs = O_ptr + o_base + rows[:, None] * d_dim + cols_d[None, :]
    O_i = tl.load(o_ptrs, mask=q_mask, other=0.0)
    
    do_ptrs = dO_ptr + do_base + rows[:, None] * d_dim + cols_d[None, :]
    dO_i = tl.load(do_ptrs, mask=q_mask, other=0.0)
    
    l_base = b_h * S_len + i * BLOCK
    l_ptrs = L_ptr + l_base + rows
    l_mask = (i * BLOCK + rows) < S_len
    L_i = tl.load(l_ptrs, mask=l_mask, other=0.0)
    
    D_i = tl.sum(dO_i * O_i, axis=1)
    
    dQ_acc = tl.zeros((BLOCK, 128), tl.float32)
    
    cols_k = tl.arange(0, BLOCK)
    
    for j in range(0, i + 1):
        k_base = base_offset_bh + j * BLOCK * d_dim
        v_base = base_offset_bh + j * BLOCK * d_dim
        
        rows_k = tl.arange(0, BLOCK)
        k_ptrs = K_ptr + k_base + rows_k[:, None] * d_dim + cols_d[None, :]
        k_mask = (j * BLOCK + rows_k[:, None]) < S_len
        K_j = tl.load(k_ptrs, mask=k_mask, other=0.0)
        
        v_ptrs = V_ptr + v_base + rows_k[:, None] * d_dim + cols_d[None, :]
        V_j = tl.load(v_ptrs, mask=k_mask, other=0.0)
        
        S = tl.dot(Q_i, K_j.T)
        
        P = tl.exp(S / sqrt_d - L_i[:, None])
        
        mask = ((j * BLOCK + cols_k[None, :]) <= (i * BLOCK + rows[:, None])) & \
               ((i * BLOCK + rows[:, None]) < S_len) & \
               ((j * BLOCK + cols_k[None, :]) < S_len)
        P = P * mask
        
        dS_raw = tl.dot(dO_i, V_j.T)
        dS = P * (dS_raw - D_i[:, None]) / sqrt_d
        
        dQ_acc = tl.dot(dS, K_j, acc=dQ_acc)
    
    out_ptrs = dQ_ptr + q_base + rows[:, None] * d_dim + cols_d[None, :]
    out_mask = (i * BLOCK + rows[:, None]) < S_len
    tl.store(out_ptrs, dQ_acc.to(tl.bfloat16), mask=out_mask)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S_len, d_dim = Q.shape
    
    sqrt_d = 1.0 / (d_dim ** 0.5)
    
    BLOCK = 64
    num_blocks = triton.cdiv(S_len, BLOCK)
    H_total = B * H
    
    grid = (num_blocks, H_total)
    
    _kernel_dVk[grid](
        Q, K, V, O, dO, L, dK, dV,
        S_len, d_dim, H_total, sqrt_d,
        BLOCK=BLOCK,
        num_warps=4, num_stages=2
    )
    
    _kernel_dQ[grid](
        Q, K, V, O, dO, L, dQ,
        S_len, d_dim, H_total, sqrt_d,
        BLOCK=BLOCK,
        num_warps=4, num_stages=2
    )