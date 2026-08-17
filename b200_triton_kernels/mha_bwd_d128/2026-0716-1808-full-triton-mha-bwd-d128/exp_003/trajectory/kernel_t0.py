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
def store_tile(ptr, s_off, bh, d, outer_base, val, S):
    rows = tl.arange(0, BLOCK_S)
    cols = tl.arange(0, BLOCK_D)
    base_ptr = ptr + outer_base + s_off * d
    ptrs = base_ptr + rows[:, None] * d + cols[None, :]
    val_bf16 = val.to(ptr.element_ty)
    max_rows = S - s_off
    tl.store(ptrs, val_bf16, mask=(rows[:, None] < max_rows))


@triton.jit
def _bwd_dq_dk(
    Q_ptr, K_ptr, V_ptr, dO_ptr, O_ptr, L_ptr,
    dQ_ptr, dK_ptr,
    S, d, scale, BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr
):
    s_off = tl.program_id(0) * BLOCK_S
    bh = tl.program_id(1)
    outer_base = bh * (S * d)
    outer_base_L = bh * S
    
    q_i = load_gathered(Q_ptr, s_off, bh, d, outer_base, S)
    do_i = load_gathered(dO_ptr, s_off, bh, d, outer_base, S)
    o_i = load_gathered(O_ptr, s_off, bh, d, outer_base, S)
    d_val_i = (do_i * o_i).to(tl.float32).sum(axis=1)
    l_i = load_scaled(L_ptr, s_off, bh, outer_base_L, S)
    
    acc_dQ = tl.zeros((BLOCK_S, BLOCK_D), tl.float32)
    
    for j in range(tl.cdiv(S, BLOCK_S)):
        s_j = j * BLOCK_S
        
        k_j = load_gathered(K_ptr, s_j, bh, d, outer_base, S)
        v_j = load_gathered(V_ptr, s_j, bh, d, outer_base, S)
        
        acc_dP = tl.dot(do_i, v_j.T, acc=None)
        
        s_ij = tl.dot(q_i, k_j.T, acc=None)
        mask_ij = (s_off + tl.arange(BLOCK_S)[:, None] < S) & (s_j + tl.arange(BLOCK_S)[None, :] < S)
        p_ij = tl.exp(s_ij * scale - l_i[:, None])
        p_ij = p_ij * mask_ij
        
        dp_ij = acc_dP - d_val_i[:, None]
        dp_ij = dp_ij * p_ij
        
        acc_dQ = tl.dot(dp_ij, k_j, acc=acc_dQ)
        
    acc_dQ = acc_dQ * scale
    
    store_tile(dQ_ptr, s_off, bh, d, outer_base, acc_dQ, S)
    
    acc_dK = tl.zeros((BLOCK_S, BLOCK_D), tl.float32)
    
    for j in range(tl.cdiv(S, BLOCK_S)):
        s_j = j * BLOCK_S
        
        do_j = load_gathered(dO_ptr, s_j, bh, d, outer_base, S)
        o_j = load_gathered(O_ptr, s_j, bh, d, outer_base, S)
        d_val_j = (do_j * o_j).to(tl.float32).sum(axis=1)
        l_j = load_scaled(L_ptr, s_j, bh, outer_base_L, S)
        
        v_j = load_gathered(V_ptr, s_j, bh, d, outer_base, S)
        
        acc_dP = tl.dot(do_j, v_i.T, acc=None)
        
        q_j = load_gathered(Q_ptr, s_j, bh, d, outer_base, S)
        k_j_j = load_gathered(K_ptr, s_j, bh, d, outer_base, S)
        
        s_ij = tl.dot(q_j, k_j_j.T, acc=None)
        mask_ij = (s_j + tl.arange(BLOCK_S)[:, None] < S) & (s_off + tl.arange(BLOCK_S)[None, :] < S)
        p_ij = tl.exp(s_ij * scale - l_j[:, None])
        p_ij = p_ij * mask_ij
        
        dp_ij = acc_dP - d_val_j[:, None]
        dp_ij = dp_ij * p_ij
        
        acc_dK = tl.dot(dp_ij.T, q_j, acc=acc_dK)
        
    acc_dK = acc_dK * scale
    
    store_tile(dK_ptr, s_off, bh, d, outer_base, acc_dK, S)


@triton.jit
def _bwd_dv(
    Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr,
    dV_ptr, S, d, scale, BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr
):
    s_off = tl.program_id(0) * BLOCK_S
    bh = tl.program_id(1)
    outer_base = bh * (S * d)
    outer_base_L = bh * S
    
    v_j = load_gathered(V_ptr, s_off, bh, d, outer_base, S)
    k_j = load_gathered(K_ptr, s_off, bh, d, outer_base, S)
    
    acc_dV = tl.zeros((BLOCK_S, BLOCK_D), tl.float32)
    
    for i in range(tl.cdiv(S, BLOCK_S)):
        s_i = i * BLOCK_S
        
        do_i = load_gathered(dO_ptr, s_i, bh, d, outer_base, S)
        
        acc_dP = tl.dot(do_i, v_j.T, acc=None)
        
        q_i = load_gathered(Q_ptr, s_i, bh, d, outer_base, S)
        
        s_ij = tl.dot(q_i, k_j.T, acc=None)
        mask_ij = (s_i + tl.arange(BLOCK_S)[:, None] < S) & (s_off + tl.arange(BLOCK_S)[None, :] < S)
        p_ij = tl.exp(s_ij * scale - l_i[:, None])
        p_ij = p_ij * mask_ij
        
        acc_dV = tl.dot(p_ij.T, do_i, acc=acc_dV)
        
    store_tile(dV_ptr, s_off, bh, d, outer_base, acc_dV, S)


BLOCK_S = 64
BLOCK_D = 128


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    b, h, s, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    
    grid = (triton.cdiv(s, BLOCK_S), b * h)
    
    _bwd_dq_dk[grid](
        Q, K, V, dO, O, L, dQ, dK,
        s, d, scale, BLOCK_S, BLOCK_D,
        num_warps=4, num_stages=2
    )
    
    _bwd_dv[grid](
        Q, K, V, dO, L, dV,
        s, d, scale, BLOCK_S, BLOCK_D,
        num_warps=4, num_stages=2
    )