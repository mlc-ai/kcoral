import torch
import triton
import triton.language as tl
import math


@triton.jit
def load_gathered(ptr, s_offset, bh, d, outer_base, S):
    base_ptr = ptr + outer_base + s_offset * d
    rows = tl.arange(0, BLOCK_S)
    cols = tl.arange(0, BLOCK_D)
    ptrs = base_ptr + rows[:, None] * d + cols[None, :]
    max_rows = S - s_offset
    vals = tl.load(ptrs, mask=(rows[:, None] < max_rows), other=0.0)
    return vals


@triton.jit
def load_scaled(L_ptr, s_offset, bh, outer_base_L, S):
    l_i = L_ptr + outer_base_L + s_offset
    rows = tl.arange(0, BLOCK_S)
    max_rows = S - s_offset
    vals = tl.load(l_i + rows, mask=(rows < max_rows), other=0.0)
    return vals


@triton.jit
def store_gathered(ptr, s_off, bh, d, outer_base, val, S):
    rows = tl.arange(0, BLOCK_S)
    cols = tl.arange(0, BLOCK_D)
    base_ptr = ptr + outer_base + s_off * d
    ptrs = base_ptr + rows[:, None] * d + cols[None, :]
    val_bf16 = val.to(ptr.element_ty)
    max_rows = S - s_off
    tl.store(ptrs, val_bf16, mask=(rows[:, None] < max_rows))


@triton.jit
def _bwd_dq(
    Q_ptr, K_ptr, V_ptr, dO_ptr, O_ptr, L_ptr, dQ_ptr,
    S, d, scale, BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr
):
    i = tl.program_id(0)
    bh = tl.program_id(1)
    
    outer_base = bh * (S * d)
    outer_base_L = bh * S
    
    q_i_0 = load_gathered(Q_ptr, i * BLOCK_S, bh, d, outer_base, S)
    do_i_0 = load_gathered(dO_ptr, i * BLOCK_S, bh, d, outer_base, S)
    o_i_0 = load_gathered(O_ptr, i * BLOCK_S, bh, d, outer_base, S)
    
    q_i_1 = load_gathered(Q_ptr, i * BLOCK_S, bh, d, outer_base + BLOCK_D, S)
    do_i_1 = load_gathered(dO_ptr, i * BLOCK_S, bh, d, outer_base + BLOCK_D, S)
    o_i_1 = load_gathered(O_ptr, i * BLOCK_S, bh, d, outer_base + BLOCK_D, S)
    
    d_val_i = ((do_i_0 * o_i_0) + (do_i_1 * o_i_1)).to(tl.float32).sum(axis=1)
    l_i = load_scaled(L_ptr, i * BLOCK_S, bh, outer_base_L, S)
    
    acc_dQ_0 = tl.zeros((BLOCK_S, BLOCK_D), tl.float32)
    acc_dQ_1 = tl.zeros((BLOCK_S, BLOCK_D), tl.float32)
    
    row_idx_f32 = tl.arange(BLOCK_S)[:, None]
    col_idx_f32 = tl.arange(BLOCK_S)[None, :]
    
    for j in range(tl.cdiv(S, BLOCK_S)):
        k_j_0 = load_gathered(K_ptr, j * BLOCK_S, bh, d, outer_base, S)
        v_j_0 = load_gathered(V_ptr, j * BLOCK_S, bh, d, outer_base, S)
        
        k_j_1 = load_gathered(K_ptr, j * BLOCK_S, bh, d, outer_base + BLOCK_D, S)
        v_j_1 = load_gathered(V_ptr, j * BLOCK_S, bh, d, outer_base + BLOCK_D, S)
        
        acc_dP_0 = tl.dot(do_i_0, v_j_0.T, acc=None)
        acc_dP_1 = tl.dot(do_i_1, v_j_1.T, acc=None)
        
        s_ij_0 = tl.dot(q_i_0, k_j_0.T, acc=None)
        s_ij_1 = tl.dot(q_i_1, k_j_1.T, acc=None)
        s_ij = s_ij_0 + s_ij_1
        
        mask_ij = ((i * BLOCK_S + row_idx_f32) < S) & ((j * BLOCK_S + col_idx_f32) < S)
        p_ij = tl.exp(s_ij * scale - l_i[:, None])
        p_ij = p_ij * mask_ij
        
        dp_ij_0 = (acc_dP_0 - d_val_i[:, None])
        dp_ij_1 = (acc_dP_1 - d_val_i[:, None])
        dp_ij_0 = dp_ij_0 * p_ij
        dp_ij_1 = dp_ij_1 * p_ij
        
        acc_dQ_0 = tl.dot(dp_ij_0, k_j_0, acc=acc_dQ_0)
        acc_dQ_1 = tl.dot(dp_ij_1, k_j_1, acc=acc_dQ_1)
        
    acc_dQ_0 = acc_dQ_0 * scale
    acc_dQ_1 = acc_dQ_1 * scale
    
    store_gathered(dQ_ptr, i * BLOCK_S, bh, d, outer_base, acc_dQ_0, S)
    store_gathered(dQ_ptr, i * BLOCK_S, bh, d, outer_base + BLOCK_D, acc_dQ_1, S)


@triton.jit
def _bwd_dk(
    Q_ptr, K_ptr, V_ptr, dO_ptr, O_ptr, L_ptr, dK_ptr,
    S, d, scale, BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr
):
    j = tl.program_id(0)
    bh = tl.program_id(1)
    
    outer_base = bh * (S * d)
    outer_base_L = bh * S
    
    k_j_0 = load_gathered(K_ptr, j * BLOCK_S, bh, d, outer_base, S)
    v_j_0 = load_gathered(V_ptr, j * BLOCK_S, bh, d, outer_base, S)
    
    k_j_1 = load_gathered(K_ptr, j * BLOCK_S, bh, d, outer_base + BLOCK_D, S)
    v_j_1 = load_gathered(V_ptr, j * BLOCK_S, bh, d, outer_base + BLOCK_D, S)
    
    do_j_0 = load_gathered(dO_ptr, j * BLOCK_S, bh, d, outer_base, S)
    o_j_0 = load_gathered(O_ptr, j * BLOCK_S, bh, d, outer_base, S)
    
    do_j_1 = load_gathered(dO_ptr, j * BLOCK_S, bh, d, outer_base + BLOCK_D, S)
    o_j_1 = load_gathered(O_ptr, j * BLOCK_S, bh, d, outer_base + BLOCK_D, S)
    
    d_val_j = ((do_j_0 * o_j_0) + (do_j_1 * o_j_1)).to(tl.float32).sum(axis=1)
    l_j = load_scaled(L_ptr, j * BLOCK_S, bh, outer_base_L, S)
    
    acc_dK_0 = tl.zeros((BLOCK_S, BLOCK_D), tl.float32)
    acc_dK_1 = tl.zeros((BLOCK_S, BLOCK_D), tl.float32)
    
    row_idx_f32 = tl.arange(BLOCK_S)[:, None]
    col_idx_f32 = tl.arange(BLOCK_S)[None, :]
    
    for i in range(tl.cdiv(S, BLOCK_S)):
        q_i_0 = load_gathered(Q_ptr, i * BLOCK_S, bh, d, outer_base, S)
        do_i_0 = load_gathered(dO_ptr, i * BLOCK_S, bh, d, outer_base, S)
        
        q_i_1 = load_gathered(Q_ptr, i * BLOCK_S, bh, d, outer_base + BLOCK_D, S)
        do_i_1 = load_gathered(dO_ptr, i * BLOCK_S, bh, d, outer_base + BLOCK_D, S)
        
        acc_dP_0 = tl.dot(do_i_0, v_j_0.T, acc=None)
        acc_dP_1 = tl.dot(do_i_1, v_j_1.T, acc=None)
        
        s_ij_0 = tl.dot(q_i_0, k_j_0.T, acc=None)
        s_ij_1 = tl.dot(q_i_1, k_j_1.T, acc=None)
        s_ij = s_ij_0 + s_ij_1
        
        mask_ij = ((i * BLOCK_S + row_idx_f32) < S) & ((j * BLOCK_S + col_idx_f32) < S)
        p_ij = tl.exp(s_ij * scale - l_j[:, None]) 
        p_ij = p_ij * mask_ij
        
        dp_ij_0 = (acc_dP_0 - d_val_j[:, None])
        dp_ij_1 = (acc_dP_1 - d_val_j[:, None])
        dp_ij_0 = dp_ij_0 * p_ij
        dp_ij_1 = dp_ij_1 * p_ij
        
        acc_dK_0 = tl.dot(dp_ij_0.T, q_i_0, acc=acc_dK_0)
        acc_dK_1 = tl.dot(dp_ij_1.T, q_i_1, acc=acc_dK_1)
        
    acc_dK_0 = acc_dK_0 * scale
    acc_dK_1 = acc_dK_1 * scale
    
    store_gathered(dK_ptr, j * BLOCK_S, bh, d, outer_base, acc_dK_0, S)
    store_gathered(dK_ptr, j * BLOCK_S, bh, d, outer_base + BLOCK_D, acc_dK_1, S)


@triton.jit
def _bwd_dv(
    Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, dV_ptr,
    S, d, scale, BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr
):
    j = tl.program_id(0)
    bh = tl.program_id(1)
    
    outer_base = bh * (S * d)
    outer_base_L = bh * S
    
    k_j_0 = load_gathered(K_ptr, j * BLOCK_S, bh, d, outer_base, S)
    k_j_1 = load_gathered(K_ptr, j * BLOCK_S, bh, d, outer_base + BLOCK_D, S)
    
    acc_dV_0 = tl.zeros((BLOCK_S, BLOCK_D), tl.float32)
    acc_dV_1 = tl.zeros((BLOCK_S, BLOCK_D), tl.float32)
    
    row_idx_f32 = tl.arange(BLOCK_S)[:, None]
    col_idx_f32 = tl.arange(BLOCK_S)[None, :]
    
    for i in range(tl.cdiv(S, BLOCK_S)):
        q_i_0 = load_gathered(Q_ptr, i * BLOCK_S, bh, d, outer_base, S)
        do_i_0 = load_gathered(dO_ptr, i * BLOCK_S, bh, d, outer_base, S)
        
        q_i_1 = load_gathered(Q_ptr, i * BLOCK_S, bh, d, outer_base + BLOCK_D, S)
        do_i_1 = load_gathered(dO_ptr, i * BLOCK_S, bh, d, outer_base + BLOCK_D, S)
        
        s_ij_0 = tl.dot(q_i_0, k_j_0.T, acc=None)
        s_ij_1 = tl.dot(q_i_1, k_j_1.T, acc=None)
        s_ij = s_ij_0 + s_ij_1
        
        l_i = load_scaled(L_ptr, i * BLOCK_S, bh, outer_base_L, S)
        
        mask_ij = ((i * BLOCK_S + row_idx_f32) < S) & ((j * BLOCK_S + col_idx_f32) < S)
        p_ij = tl.exp(s_ij * scale - l_i[:, None]) 
        p_ij = p_ij * mask_ij
        
        acc_dV_0 = tl.dot(p_ij.T, do_i_0, acc=acc_dV_0)
        acc_dV_1 = tl.dot(p_ij.T, do_i_1, acc=acc_dV_1)
        
    store_gathered(dV_ptr, j * BLOCK_S, bh, d, outer_base, acc_dV_0, S)
    store_gathered(dV_ptr, j * BLOCK_S, bh, d, outer_base + BLOCK_D, acc_dV_1, S)


BLOCK_S = 64
BLOCK_D = 64


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    b, h, s, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    
    dQ.fill_(0.0)
    dK.fill_(0.0)
    dV.fill_(0.0)
    
    grid = (triton.cdiv(s, BLOCK_S), triton.cdiv(s, BLOCK_S), b * h)
    
    _bwd_dq[grid](
        Q, K, V, dO, O, L, dQ,
        s, d, scale, BLOCK_S, BLOCK_D,
        num_warps=4, num_stages=2
    )
    
    _bwd_dk[grid](
        Q, K, V, dO, O, L, dK,
        s, d, scale, BLOCK_S, BLOCK_D,
        num_warps=4, num_stages=2
    )
    
    _bwd_dv[grid](
        Q, K, V, dO, L, dV,
        s, d, scale, BLOCK_S, BLOCK_D,
        num_warps=4, num_stages=2
    )