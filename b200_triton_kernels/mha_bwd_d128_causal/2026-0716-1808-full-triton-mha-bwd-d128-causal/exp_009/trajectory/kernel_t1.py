import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
import math


@triton.jit
def wg_do(acc, A, B, K_dim, step):
    """Perform a series of dot products accumulating into acc."""
    for i in range(0, K_dim // step):
        a_part = A[:, i*step:(i+1)*step]
        b_part = B[:, i*step:(i+1)*step]
        acc = tl.dot(a_part, b_part.T, acc)
    return acc


@triton.jit
def wg_dot(acc, A, B, K_dim, step):
    """Perform a series of dot products accumulating into acc."""
    for i in range(0, K_dim // step):
        a_part = A[:, i*step:(i+1)*step]
        b_part = B[i*step:(i+1)*step, :]
        acc = tl.dot(a_part, b_part, acc)
    return acc


@triton.jit
def _bwd_dq_kernel(
    Q_desc, K_desc, V_desc, O_desc, dO_desc, L, dQ,
    S, scale,
):
    m_blk = tl.program_id(0)
    bh_id = tl.program_id(1)
    
    off_m = tl.arange(0, 128)
    off_n = tl.arange(0, 128)
    
    q = Q_desc.load([bh_id * S + m_blk * 128, 0])
    do_ = dO_desc.load([bh_id * S + m_blk * 128, 0])
    o = O_desc.load([bh_id * S + m_blk * 128, 0])
    
    L_ptr = L + bh_id * S + m_blk * 128
    l = tl.load(L_ptr + off_m, mask=off_m < S, other=-float('inf'))
    
    d = tl.sum(do_ * o, axis=1)
    d_exp = d[:, None]
    
    dq = tl.zeros((128, 128), dtype=tl.float32)
    
    for n_blk in range(0, m_blk + 1):
        k = K_desc.load([bh_id * S + n_blk * 128, 0])
        v = V_desc.load([bh_id * S + n_blk * 128, 0])
        
        s_gem = wg_do(tl.zeros((128, 128), dtype=tl.float32), q, k, 128, 64)
        dp_gem = wg_do(tl.zeros((128, 128), dtype=tl.float32), do_, v, 128, 64)
        
        s = s_gem * scale
        
        if n_blk == m_blk:
            valid = (off_n[None, :] <= off_m[:, None])
            s = tl.where(valid, s, -float('inf'))
            
        p = tl.exp(s - l[:, None])
        
        if n_blk == m_blk:
            p = tl.where(valid, p, 0.0)
            
        ds = p * (dp_gem - d_exp) * scale
        
        dq = wg_dot(dq, ds, k, 128, 64)
        
    off_row = bh_id * S + m_blk * 128 + off_m
    mask = off_row < S
    
    dQ_ptr = dQ + off_row[:, None] * 128 + off_n[None, :]
    tl.store(dQ_ptr, dq.to(tl.bfloat16), mask=mask[:, None])


@triton.jit
def _bwd_dk_dv_kernel(
    Q_desc, K_desc, V_desc, O_desc, dO_desc, L, dK, dV,
    S, scale,
):
    n_blk = tl.program_id(0)
    bh_id = tl.program_id(1)
    
    off_m = tl.arange(0, 128)
    off_n = tl.arange(0, 128)
    
    k = K_desc.load([bh_id * S + n_blk * 128, 0])
    v = V_desc.load([bh_id * S + n_blk * 128, 0])
    
    dk = tl.zeros((128, 128), dtype=tl.float32)
    dv = tl.zeros((128, 128), dtype=tl.float32)
    
    total_blks = tl.cdiv(S, 128)
    
    for i_blk in range(n_blk, total_blks):
        q = Q_desc.load([bh_id * S + i_blk * 128, 0])
        do_ = dO_desc.load([bh_id * S + i_blk * 128, 0])
        o = O_desc.load([bh_id * S + i_blk * 128, 0])
        
        L_ptr = L + bh_id * S + i_blk * 128
        l = tl.load(L_ptr + off_m, mask=off_m < S, other=-float('inf'))
        
        d = tl.sum(do_ * o, axis=1)
        d_exp = d[:, None]
        
        s_gem = wg_do(tl.zeros((128, 128), dtype=tl.float32), q, k, 128, 64)
        dp_gem = wg_do(tl.zeros((128, 128), dtype=tl.float32), do_, v, 128, 64)
        
        s = s_gem * scale
        
        if i_blk == n_blk:
            valid = (off_n[None, :] <= off_m[:, None])
            s = tl.where(valid, s, -float('inf'))
            
        p = tl.exp(s - l[:, None])
        
        if i_blk == n_blk:
            p = tl.where(valid, p, 0.0)
            
        ds = p * (dp_gem - d_exp) * scale
        
        dk = wg_dot(dk, ds.T, q, 128, 64)
        dv = wg_dot(dv, p.T, do_, 128, 64)
        
    off_row = bh_id * S + n_blk * 128 + off_n
    mask = off_row < S
    
    dK_ptr = dK + off_row[:, None] * 128 + off_m[None, :]
    tl.store(dK_ptr, dk.to(tl.bfloat16), mask=mask[:, None])
    
    dV_ptr = dV + off_row[:, None] * 128 + off_m[None, :]
    tl.store(dV_ptr, dv.to(tl.bfloat16), mask=mask[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute FlashAttention backward pass for causal multi-head attention."""
    torch.cuda.set_device(Q.device)
    
    B = Q.shape[0]
    H = Q.shape[1]
    S = Q.shape[2]
    d = Q.shape[3]
    
    scale = 1.0 / math.sqrt(d)
    
    Q_desc = TensorDescriptor.from_tensor(Q.view(B * H * S, d), [128, 128])
    K_desc = TensorDescriptor.from_tensor(K.view(B * H * S, d), [128, 128])
    V_desc = TensorDescriptor.from_tensor(V.view(B * H * S, d), [128, 128])
    O_desc = TensorDescriptor.from_tensor(O.view(B * H * S, d), [128, 128])
    dO_desc = TensorDescriptor.from_tensor(dO.view(B * H * S, d), [128, 128])
    
    grid = (
        triton.cdiv(S, 128),
        B * H,
    )
    
    _bwd_dk_dv_kernel[grid](
        Q_desc, K_desc, V_desc, O_desc, dO_desc, L, dK, dV,
        S, scale,
        num_warps=4, num_stages=2,
    )
    
    _bwd_dq_kernel[grid](
        Q_desc, K_desc, V_desc, O_desc, dO_desc, L, dQ,
        S, scale,
        num_warps=4, num_stages=2,
    )