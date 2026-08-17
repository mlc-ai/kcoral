import torch
import triton
import triton.language as tl
import math

D = 128
BLOCK_S = 64


@triton.jit
def load_gathered(ptr, s_offset, bh, outer_base, S):
    base_ptr = ptr + outer_base + s_offset * D
    rows = tl.arange(0, BLOCK_S)
    cols = tl.arange(0, 64)
    ptrs_0 = base_ptr + rows[:, None] * D + cols[None, :]
    ptrs_1 = base_ptr + rows[:, None] * D + cols[None, :] + 64
    max_rows = S - s_offset
    vals_0 = tl.load(ptrs_0, mask=(rows[:, None] < max_rows), other=0.0)
    vals_1 = tl.load(ptrs_1, mask=(rows[:, None] < max_rows), other=0.0)
    return vals_0, vals_1


@triton.jit
def load_scaled(L_ptr, s_offset, bh, outer_base_L, S):
    l_i = L_ptr + outer_base_L + s_offset
    rows = tl.arange(0, BLOCK_S)
    max_rows = S - s_offset
    vals = tl.load(l_i + rows, mask=(rows < max_rows), other=0.0)
    return vals


@triton.jit
def store_gathered(ptr, s_off, bh, outer_base, val_0, val_1, S):
    base_ptr = ptr + outer_base + s_off * D
    rows = tl.arange(0, BLOCK_S)
    cols = tl.arange(0, 64)
    ptrs_0 = base_ptr + rows[:, None] * D + cols[None, :]
    ptrs_1 = base_ptr + rows[:, None] * D + cols[None, :] + 64
    val_0_bf16 = val_0.to(ptr.element_ty)
    val_1_bf16 = val_1.to(ptr.element_ty)
    max_rows = S - s_off
    mask = rows[:, None] < max_rows
    tl.store(ptrs_0, val_0_bf16, mask=mask)
    tl.store(ptrs_1, val_1_bf16, mask=mask)


@triton.jit
def _bwd_dq(Q_ptr, K_ptr, V_ptr, dO_ptr, O_ptr, L_ptr, dQ_ptr, S, scale):
    i = tl.program_id(0)
    bh = tl.program_id(1)
    
    outer_base = bh * (S * D)
    outer_base_L = bh * S
    
    q_i_0, q_i_1 = load_gathered(Q_ptr, i * BLOCK_S, bh, outer_base, S)
    do_i_0, do_i_1 = load_gathered(dO_ptr, i * BLOCK_S, bh, outer_base, S)
    o_i_0, o_i_1 = load_gathered(O_ptr, i * BLOCK_S, bh, outer_base, S)
    
    d_val_i = ((do_i_0 * o_i_0) + (do_i_1 * o_i_1)).to(tl.float32).sum(axis=1)
    l_i = load_scaled(L_ptr, i * BLOCK_S, bh, outer_base_L, S)
    
    acc_dQ_0 = tl.zeros((BLOCK_S, 64), tl.float32)
    acc_dQ_1 = tl.zeros((BLOCK_S, 64), tl.float32)
    
    row_idx = tl.arange(BLOCK_S)
    col_idx = tl.arange(BLOCK_S)
    
    for j in range(tl.cdiv(S, BLOCK_S)):
        k_j_0, k_j_1 = load_gathered(K_ptr, j * BLOCK_S, bh, outer_base, S)
        v_j_0, v_j_1 = load_gathered(V_ptr, j * BLOCK_S, bh, outer_base, S)
        
        s_ij_0 = tl.dot(q_i_0, k_j_0.T, acc=None)
        s_ij_1 = tl.dot(q_i_1, k_j_1.T, acc=None)
        s_ij = s_ij_0 + s_ij_1
        
        mask_ij = ((i * BLOCK_S + row_idx[:, None]) < S) & ((j * BLOCK_S + col_idx[None, :]) < S)
        p_ij = tl.exp(s_ij * scale - l_i[:, None])
        p_ij = p_ij * mask_ij
        
        acc_dP_0 = tl.dot(do_i_0, v_j_0.T, acc=None)
        acc_dP_1 = tl.dot(do_i_1, v_j_1.T, acc=None)
        
        acc_dP = acc_dP_0 + acc_dP_1
        dp_ij = (acc_dP - d_val_i[:, None]) * p_ij
        
        acc_dQ_0 = tl.dot(dp_ij, k_j_0, acc=acc_dQ_0)
        acc_dQ_1 = tl.dot(dp_ij, k_j_1, acc=acc_dQ_1)
        
    acc_dQ_0 = acc_dQ_0 * scale
    acc_dQ_1 = acc_dQ_1 * scale
    
    store_gathered(dQ_ptr, i * BLOCK_S, bh, outer_base, acc_dQ_0, acc_dQ_1, S)


@triton.jit
def _bwd_dk(Q_ptr, K_ptr, V_ptr, dO_ptr, O_ptr, L_ptr, dK_ptr, S, scale):
    j = tl.program_id(0)
    bh = tl.program_id(1)
    
    outer_base = bh * (S * D)
    outer_base_L = bh * S
    
    k_j_0, k_j_1 = load_gathered(K_ptr, j * BLOCK_S, bh, outer_base, S)
    v_j_0, v_j_1 = load_gathered(V_ptr, j * BLOCK_S, bh, outer_base, S)
    
    acc_dK_0 = tl.zeros((BLOCK_S, 64), tl.float32)
    acc_dK_1 = tl.zeros((BLOCK_S, 64), tl.float32)
    
    row_idx = tl.arange(BLOCK_S)
    col_idx = tl.arange(BLOCK_S)
    
    for i in range(tl.cdiv(S, BLOCK_S)):
        q_i_0, q_i_1 = load_gathered(Q_ptr, i * BLOCK_S, bh, outer_base, S)
        do_i_0, do_i_1 = load_gathered(dO_ptr, i * BLOCK_S, bh, outer_base, S)
        o_i_0, o_i_1 = load_gathered(O_ptr, i * BLOCK_S, bh, outer_base, S)
        
        d_val_i = ((do_i_0 * o_i_0) + (do_i_1 * o_i_1)).to(tl.float32).sum(axis=1)
        l_i = load_scaled(L_ptr, i * BLOCK_S, bh, outer_base_L, S)
        
        s_ij_0 = tl.dot(q_i_0, k_j_0.T, acc=None)
        s_ij_1 = tl.dot(q_i_1, k_j_1.T, acc=None)
        s_ij = s_ij_0 + s_ij_1
        
        mask_ij = ((i * BLOCK_S + row_idx[:, None]) < S) & ((j * BLOCK_S + col_idx[None, :]) < S)
        p_ij = tl.exp(s_ij * scale - l_i[:, None]) 
        p_ij = p_ij * mask_ij
        
        acc_dP_0 = tl.dot(do_i_0, v_j_0.T, acc=None)
        acc_dP_1 = tl.dot(do_i_1, v_j_1.T, acc=None)
        
        acc_dP = acc_dP_0 + acc_dP_1
        dp_ij = (acc_dP - d_val_i[:, None]) * p_ij
        
        acc_dK_0 = tl.dot(dp_ij.T, q_i_0, acc=acc_dK_0)
        acc_dK_1 = tl.dot(dp_ij.T, q_i_1, acc=acc_dK_1)
        
    acc_dK_0 = acc_dK_0 * scale
    acc_dK_1 = acc_dK_1 * scale
    
    store_gathered(dK_ptr, j * BLOCK_S, bh, outer_base, acc_dK_0, acc_dK_1, S)


@triton.jit
def _bwd_dv(Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, dV_ptr, S, scale):
    j = tl.program_id(0)
    bh = tl.program_id(1)
    
    outer_base = bh * (S * D)
    outer_base_L = bh * S
    
    k_j_0, k_j_1 = load_gathered(K_ptr, j * BLOCK_S, bh, outer_base, S)
    
    acc_dV_0 = tl.zeros((BLOCK_S, 64), tl.float32)
    acc_dV_1 = tl.zeros((BLOCK_S, 64), tl.float32)
    
    row_idx = tl.arange(BLOCK_S)
    col_idx = tl.arange(BLOCK_S)
    
    for i in range(tl.cdiv(S, BLOCK_S)):
        q_i_0, q_i_1 = load_gathered(Q_ptr, i * BLOCK_S, bh, outer_base, S)
        do_i_0, do_i_1 = load_gathered(dO_ptr, i * BLOCK_S, bh, outer_base, S)
        
        s_ij_0 = tl.dot(q_i_0, k_j_0.T, acc=None)
        s_ij_1 = tl.dot(q_i_1, k_j_1.T, acc=None)
        s_ij = s_ij_0 + s_ij_1
        
        l_i = load_scaled(L_ptr, i * BLOCK_S, bh, outer_base_L, S)
        
        mask_ij = ((i * BLOCK_S + row_idx[:, None]) < S) & ((j * BLOCK_S + col_idx[None, :]) < S)
        p_ij = tl.exp(s_ij * scale - l_i[:, None]) 
        p_ij = p_ij * mask_ij
        
        acc_dV_0 = tl.dot(p_ij.T, do_i_0, acc=acc_dV_0)
        acc_dV_1 = tl.dot(p_ij.T, do_i_1, acc=acc_dV_1)
        
    store_gathered(dV_ptr, j * BLOCK_S, bh, outer_base, acc_dV_0, acc_dV_1, S)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    b, h, s, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    
    grid_1 = (triton.cdiv(S, BLOCK_S), b * h)
    
    _bwd_dq[grid_1](
        Q, K, V, dO, O, L, dQ,
        s, scale,
        num_warps=4, num_stages=2
    )
    
    _bwd_dk[grid_1](
        Q, K, V, dO, O, L, dK,
        s, scale,
        num_warps=4, num_stages=2
    )
    
    _bwd_dv[grid_1](
        Q, K, V, dO, L, dV,
        s, scale,
        num_warps=4, num_stages=2
    )