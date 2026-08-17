import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
import math


@triton.jit
def _bwd_dq_kernel(
    Q_desc, K_desc, V_desc, O_desc, dO_desc, L, dQ,
    S, scale,
):
    m_blk = tl.program_id(0)
    bh_id = tl.program_id(1)
    
    s_idx = m_blk * 128
    
    q = tensor_map(Q_desc, [bh_id, s_idx, 0])
    do_ = tensor_map(dO_desc, [bh_id, s_idx, 0])
    o = tensor_map(O_desc, [bh_id, s_idx, 0])
    
    off_m = tl.arange(0, 128)
    L_ptr = L + bh_id * S + s_idx
    l = tl.load(L_ptr + off_m, mask=(s_idx + off_m) < S, other=-float('inf'))
    
    d = tl.sum(do_ * o, axis=1)
    d_exp = d[:, None]
    
    dq = tl.zeros((128, 128), dtype=tl.float32)
    
    for n_blk in range(0, m_blk + 1):
        k_idx = n_blk * 128
        k = tensor_map(K_desc, [bh_id, k_idx, 0])
        v = tensor_map(V_desc, [bh_id, k_idx, 0])
        
        s_gem = tl.zeros((128, 128), dtype=tl.float32)
        s_gem = tl.dot(q, k.T, s_gem)
        s = s_gem * scale
        
        if n_blk == m_blk:
            off_n = tl.arange(0, 128)
            valid = (k_idx + off_n[None, :]) <= (s_idx + off_m[:, None])
            s = tl.where(valid, s, -float('inf'))
            
        p = tl.exp(s - l[:, None])
        
        if n_blk == m_blk:
            p = tl.where(valid, p, 0.0)
            
        dp_gem = tl.zeros((128, 128), dtype=tl.float32)
        dp_gem = tl.dot(do_, v.T, dp_gem)
        
        ds = p * (dp_gem - d_exp) * scale
        
        dq = tl.dot(ds, k, dq)
        
    off_row = bh_id * S + s_idx + off_m
    mask = off_row < S
    
    off_n = tl.arange(0, 128)
    dQ_ptr = dQ + off_row[:, None] * 128 + off_n[None, :]
    tl.store(dQ_ptr, dq.to(tl.bfloat16), mask=mask[:, None])


@triton.jit
def _bwd_dk_dv_kernel(
    Q_desc, K_desc, V_desc, O_desc, dO_desc, L, dK, dV,
    S, scale,
):
    n_blk = tl.program_id(0)
    bh_id = tl.program_id(1)
    
    k_idx = n_blk * 128
    k = tensor_map(K_desc, [bh_id, k_idx, 0])
    v = tensor_map(V_desc, [bh_id, k_idx, 0])
    
    dk = tl.zeros((128, 128), dtype=tl.float32)
    dv = tl.zeros((128, 128), dtype=tl.float32)
    
    total_blks = tl.cdiv(S, 128)
    
    for i_blk in range(n_blk, total_blks):
        s_idx = i_blk * 128
        q = tensor_map(Q_desc, [bh_id, s_idx, 0])
        do_ = tensor_map(dO_desc, [bh_id, s_idx, 0])
        o = tensor_map(O_desc, [bh_id, s_idx, 0])
        
        off_m = tl.arange(0, 128)
        L_ptr = L + bh_id * S + s_idx
        l = tl.load(L_ptr + off_m, mask=(s_idx + off_m) < S, other=-float('inf'))
        
        d = tl.sum(do_ * o, axis=1)
        d_exp = d[:, None]
        
        s_gem = tl.zeros((128, 128), dtype=tl.float32)
        s_gem = tl.dot(q, k.T, s_gem)
        s = s_gem * scale
        
        if i_blk == n_blk:
            off_n = tl.arange(0, 128)
            valid = (k_idx + off_n[None, :]) <= (s_idx + off_m[:, None])
            s = tl.where(valid, s, -float('inf'))
            
        p = tl.exp(s - l[:, None])
        
        if i_blk == n_blk:
            p = tl.where(valid, p, 0.0)
            
        dp_gem = tl.zeros((128, 128), dtype=tl.float32)
        dp_gem = tl.dot(do_, v.T, dp_gem)
        
        ds = p * (dp_gem - d_exp) * scale
        
        dk = tl.dot(ds.T, q, dk)
        dv = tl.dot(p.T, do_, dv)
        
    off_row = bh_id * S + k_idx + off_m
    mask = off_row < S
    
    off_n = tl.arange(0, 128)
    dK_ptr = dK + off_row[:, None] * 128 + off_n[None, :]
    tl.store(dK_ptr, dk.to(tl.bfloat16), mask=mask[:, None])
    
    dV_ptr = dV + off_row[:, None] * 128 + off_n[None, :]
    tl.store(dV_ptr, dv.to(tl.bfloat16), mask=mask[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute FlashAttention backward pass for causal multi-head attention."""
    torch.cuda.set_device(Q.device)
    
    B = Q.shape[0]
    H = Q.shape[1]
    S = Q.shape[2]
    d = Q.shape[3]
    
    scale = 1.0 / math.sqrt(d)
    
    Q_desc = TensorDescriptor.from_tensor(Q.view(B * H, S, d), shape=[B * H, S, d], block_shape=[1, 128, 128])
    K_desc = TensorDescriptor.from_tensor(K.view(B * H, S, d), shape=[B * H, S, d], block_shape=[1, 128, 128])
    V_desc = TensorDescriptor.from_tensor(V.view(B * H, S, d), shape=[B * H, S, d], block_shape=[1, 128, 128])
    O_desc = TensorDescriptor.from_tensor(O.view(B * H, S, d), shape=[B * H, S, d], block_shape=[1, 128, 128])
    dO_desc = TensorDescriptor.from_tensor(dO.view(B * H, S, d), shape=[B * H, S, d], block_shape=[1, 128, 128])
    
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