import torch
import triton
import triton.language as tl
import math

BLOCK_S = 64


@triton.jit
def load_half_tile(ptr, row_start, bh, S):
    outer_base = bh * (S * 128)
    base_ptr = ptr + outer_base + row_start * 128
    rows = tl.arange(0, BLOCK_S)
    cols = tl.arange(0, 64)
    ptrs_0 = base_ptr + rows[:, None] * 128 + cols[None, :]
    ptrs_1 = base_ptr + rows[:, None] * 128 + cols[None, :] + 64
    max_rows = S - row_start
    vals_0 = tl.load(ptrs_0, mask=(rows[:, None] < max_rows), other=0.0)
    vals_1 = tl.load(ptrs_1, mask=(rows[:, None] < max_rows), other=0.0)
    return vals_0, vals_1


@triton.jit
def load_scaled(L_ptr, row_start, bh, S):
    l_i = L_ptr + bh * S + row_start
    rows = tl.arange(0, BLOCK_S)
    max_rows = S - row_start
    vals = tl.load(l_i + rows, mask=(rows < max_rows), other=0.0)
    return vals


@triton.jit
def store_half_tile(ptr, row_start, bh, val_0, val_1, S):
    outer_base = bh * (S * 128)
    base_ptr = ptr + outer_base + row_start * 128
    rows = tl.arange(0, BLOCK_S)
    cols = tl.arange(0, 64)
    ptrs_0 = base_ptr + rows[:, None] * 128 + cols[None, :]
    ptrs_1 = base_ptr + rows[:, None] * 128 + cols[None, :] + 64
    
    val_0_bf16 = val_0.to(ptr.element_ty)
    val_1_bf16 = val_1.to(ptr.element_ty)
    
    max_rows = S - row_start
    mask = rows[:, None] < max_rows
    tl.store(ptrs_0, val_0_bf16, mask=mask)
    tl.store(ptrs_1, val_1_bf16, mask=mask)


@triton.jit
def _bwd_dq(Q, K, V, O, dO, L, dQ, S, scale):
    i = tl.program_id(0)
    bh = tl.program_id(1)
    
    q_i_0, q_i_1 = load_half_tile(Q, i * BLOCK_S, bh, S)
    do_i_0, do_i_1 = load_half_tile(dO, i * BLOCK_S, bh, S)
    o_i_0, o_i_1 = load_half_tile(O, i * BLOCK_S, bh, S)
    
    d_val_i = ((do_i_0 * o_i_0) + (do_i_1 * o_i_1)).to(tl.float32).sum(axis=1)
    l_i = load_scaled(L, i * BLOCK_S, bh, S)
    
    acc_dQ_0 = tl.zeros((BLOCK_S, 64), tl.float32)
    acc_dQ_1 = tl.zeros((BLOCK_S, 64), tl.float32)
    
    row_idx = tl.arange(BLOCK_S)[:, None]
    col_idx = tl.arange(BLOCK_S)[None, :]
    
    for j in range(tl.cdiv(S, BLOCK_S)):
        k_j_0, k_j_1 = load_half_tile(K, j * BLOCK_S, bh, S)
        v_j_0, v_j_1 = load_half_tile(V, j * BLOCK_S, bh, S)
        
        s_ij = tl.dot(q_i_0, k_j_0.T, acc=None) + tl.dot(q_i_1, k_j_1.T, acc=None)
        
        mask_ij = ((i * BLOCK_S + row_idx) < S) & ((j * BLOCK_S + col_idx) < S)
        p_ij = tl.exp(s_ij * scale - l_i[:, None])
        p_ij = p_ij * mask_ij
        
        acc_dP = tl.dot(do_i_0, v_j_0.T, acc=None) + tl.dot(do_i_1, v_j_1.T, acc=None)
        dp_ij = (acc_dP - d_val_i[:, None]) * p_ij
        
        acc_dQ_0 = tl.dot(dp_ij, k_j_0, acc=acc_dQ_0)
        acc_dQ_1 = tl.dot(dp_ij, k_j_1, acc=acc_dQ_1)
        
    acc_dQ_0 = acc_dQ_0 * scale
    acc_dQ_1 = acc_dQ_1 * scale
    
    store_half_tile(dQ, i * BLOCK_S, bh, acc_dQ_0, acc_dQ_1, S)


@triton.jit
def _bwd_dk(Q, K, V, O, dO, L, dK, S, scale):
    j = tl.program_id(0)
    bh = tl.program_id(1)
    
    k_j_0, k_j_1 = load_half_tile(K, j * BLOCK_S, bh, S)
    v_j_0, v_j_1 = load_half_tile(V, j * BLOCK_S, bh, S)
    
    acc_dK_0 = tl.zeros((BLOCK_S, 64), tl.float32)
    acc_dK_1 = tl.zeros((BLOCK_S, 64), tl.float32)
    
    row_idx = tl.arange(BLOCK_S)[:, None]
    col_idx = tl.arange(BLOCK_S)[None, :]
    
    for i in range(tl.cdiv(S, BLOCK_S)):
        q_i_0, q_i_1 = load_half_tile(Q, i * BLOCK_S, bh, S)
        do_i_0, do_i_1 = load_half_tile(dO, i * BLOCK_S, bh, S)
        o_i_0, o_i_1 = load_half_tile(O, i * BLOCK_S, bh, S)
        
        d_val_i = ((do_i_0 * o_i_0) + (do_i_1 * o_i_1)).to(tl.float32).sum(axis=1)
        l_i = load_scaled(L, i * BLOCK_S, bh, S)
        
        s_ij = tl.dot(q_i_0, k_j_0.T, acc=None) + tl.dot(q_i_1, k_j_1.T, acc=None)
        
        mask_ij = ((i * BLOCK_S + row_idx) < S) & ((j * BLOCK_S + col_idx) < S)
        p_ij = tl.exp(s_ij * scale - l_i[:, None])
        p_ij = p_ij * mask_ij
        
        acc_dP = tl.dot(do_i_0, v_j_0.T, acc=None) + tl.dot(do_i_1, v_j_1.T, acc=None)
        dp_ij = (acc_dP - d_val_i[:, None]) * p_ij
        
        acc_dK_0 = tl.dot(dp_ij.T, q_i_0, acc=acc_dK_0)
        acc_dK_1 = tl.dot(dp_ij.T, q_i_1, acc=acc_dK_1)
        
    acc_dK_0 = acc_dK_0 * scale
    acc_dK_1 = acc_dK_1 * scale
    
    store_half_tile(dK, j * BLOCK_S, bh, acc_dK_0, acc_dK_1, S)


@triton.jit
def _bwd_dv(Q, K, V, dO, L, dV, S, scale):
    j = tl.program_id(0)
    bh = tl.program_id(1)
    
    k_j_0, k_j_1 = load_half_tile(K, j * BLOCK_S, bh, S)
    
    acc_dV_0 = tl.zeros((BLOCK_S, 64), tl.float32)
    acc_dV_1 = tl.zeros((BLOCK_S, 64), tl.float32)
    
    row_idx = tl.arange(BLOCK_S)[:, None]
    col_idx = tl.arange(BLOCK_S)[None, :]
    
    for i in range(tl.cdiv(S, BLOCK_S)):
        q_i_0, q_i_1 = load_half_tile(Q, i * BLOCK_S, bh, S)
        do_i_0, do_i_1 = load_half_tile(dO, i * BLOCK_S, bh, S)
        
        s_ij = tl.dot(q_i_0, k_j_0.T, acc=None) + tl.dot(q_i_1, k_j_1.T, acc=None)
        
        l_i = load_scaled(L, i * BLOCK_S, bh, S)
        
        mask_ij = ((i * BLOCK_S + row_idx) < S) & ((j * BLOCK_S + col_idx) < S)
        p_ij = tl.exp(s_ij * scale - l_i[:, None])
        p_ij = p_ij * mask_ij
        
        acc_dV_0 = tl.dot(p_ij.T, do_i_0, acc=acc_dV_0)
        acc_dV_1 = tl.dot(p_ij.T, do_i_1, acc=acc_dV_1)
        
    store_half_tile(dV, j * BLOCK_S, bh, acc_dV_0, acc_dV_1, S)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    b, h, s, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    
    grid_1 = (triton.cdiv(s, BLOCK_S), b * h)
    
    _bwd_dq[grid_1](
        Q, K, V, O, dO, L, dQ,
        s, scale,
        num_warps=4, num_stages=3
    )
    
    _bwd_dk[grid_1](
        Q, K, V, O, dO, L, dK,
        s, scale,
        num_warps=4, num_stages=3
    )
    
    _bwd_dv[grid_1](
        Q, K, V, dO, L, dV,
        s, scale,
        num_warps=4, num_stages=3
    )