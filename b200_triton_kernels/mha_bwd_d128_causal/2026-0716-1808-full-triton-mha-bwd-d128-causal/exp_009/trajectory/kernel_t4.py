import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
import math


@triton.jit
def _bwd_dq_kernel(
    Q_desc, K_desc, V_desc, O_desc, dO_desc, L, dQ,
    S, scale, total_S, L_s1,
    dQ_s2, dQ_s3,
):
    m_blk = tl.program_id(0)
    bh_id = tl.program_id(1)
    
    s_idx = m_blk * 128
    
    q = Q_desc.load([bh_id, s_idx, 0]).squeeze(0)
    do_ = dO_desc.load([bh_id, s_idx, 0]).squeeze(0)
    o = O_desc.load([bh_id, s_idx, 0]).squeeze(0)
    
    off_m = tl.arange(0, 128)
    L_ptr = L + bh_id * L_s1 + s_idx
    l = tl.load(L_ptr + off_m, mask=(bh_id * S + s_idx + off_m) < total_S, other=-float('inf'))
    
    d_val = tl.sum(do_ * o, axis=1)
    d_exp = d_val[:, None]
    
    dq = tl.zeros((128, 128), dtype=tl.float32)
    
    for n_blk in range(0, m_blk + 1):
        k_idx = n_blk * 128
        k = K_desc.load([bh_id, k_idx, 0]).squeeze(0)
        v = V_desc.load([bh_id, k_idx, 0]).squeeze(0)
        
        s_gem = tl.zeros((128, 128), dtype=tl.float32)
        s_gem = tl.dot(q, k.T, s_gem)
        s = s_gem * scale
        
        off_n = tl.arange(0, 128)
        valid = (k_idx + off_n[None, :]) <= (s_idx + off_m[:, None])
        s = tl.where(valid, s, -float('inf'))
            
        p = tl.exp(s - l[:, None])
        p = tl.where(valid, p, 0.0)
            
        dp_gem = tl.zeros((128, 128), dtype=tl.float32)
        dp_gem = tl.dot(do_, v.T, dp_gem)
        
        ds = p * (dp_gem - d_exp) * scale
        
        dq = tl.dot(ds, k, dq)
        
    off_row = bh_id * S + s_idx + off_m
    mask = off_row < (bh_id * S + S) # Mask ensures we do not write out of bounds for a specific BH slice
    
    off_n = tl.arange(0, 128)
    dQ_ptr = dQ + off_row[:, None] * dQ_s2 + off_n[None, :] * dQ_s3
    tl.store(dQ_ptr, dq.to(tl.bfloat16), mask=mask[:, None])


@triton.jit
def _bwd_dk_dv_kernel(
    Q_desc, K_desc, V_desc, O_desc, dO_desc, L, dK, dV,
    S, scale, total_S, L_s1,
    dK_s2, dK_s3, dV_s2, dV_s3,
):
    n_blk = tl.program_id(0)
    bh_id = tl.program_id(1)
    
    k_idx = n_blk * 128
    k = K_desc.load([bh_id, k_idx, 0]).squeeze(0)
    v = V_desc.load([bh_id, k_idx, 0]).squeeze(0)
    
    dk = tl.zeros((128, 128), dtype=tl.float32)
    dv = tl.zeros((128, 128), dtype=tl.float32)
    
    total_blks = tl.cdiv(S, 128)
    
    for i_blk in range(n_blk, total_blks):
        s_idx = i_blk * 128
        q = Q_desc.load([bh_id, s_idx, 0]).squeeze(0)
        do_ = dO_desc.load([bh_id, s_idx, 0]).squeeze(0)
        o = O_desc.load([bh_id, s_idx, 0]).squeeze(0)
        
        off_m = tl.arange(0, 128)
        L_ptr = L + bh_id * L_s1 + s_idx
        l = tl.load(L_ptr + off_m, mask=(bh_id * S + s_idx + off_m) < total_S, other=-float('inf'))
        
        d_val = tl.sum(do_ * o, axis=1)
        d_exp = d_val[:, None]
        
        s_gem = tl.zeros((128, 128), dtype=tl.float32)
        s_gem = tl.dot(q, k.T, s_gem)
        s = s_gem * scale
        
        off_n = tl.arange(0, 128)
        valid = (k_idx + off_n[None, :]) <= (s_idx + off_m[:, None])
        s = tl.where(valid, s, -float('inf'))
            
        p = tl.exp(s - l[:, None])
        p = tl.where(valid, p, 0.0)
            
        dp_gem = tl.zeros((128, 128), dtype=tl.float32)
        dp_gem = tl.dot(do_, v.T, dp_gem)
        
        ds = p * (dp_gem - d_exp) * scale
        
        dk = tl.dot(ds.T, q, dk)
        dv = tl.dot(p.T, do_, dv)
        
    off_row = bh_id * S + k_idx + off_n
    mask = off_row < (bh_id * S + S)
    
    dK_ptr = dK + off_row[:, None] * dK_s2 + off_m[None, :] * dK_s3
    tl.store(dK_ptr, dk.to(tl.bfloat16), mask=mask[:, None])
    
    dV_ptr = dV + off_row[:, None] * dV_s2 + off_m[None, :] * dV_s3
    tl.store(dV_ptr, dv.to(tl.bfloat16), mask=mask[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute FlashAttention backward pass for causal multi-head attention."""
    torch.cuda.set_device(Q.device)
    
    B = Q.shape[0]
    H = Q.shape[1]
    S = Q.shape[2]
    d = Q.shape[3]
    
    scale = 1.0 / math.sqrt(d)
    
    Q_desc = TensorDescriptor.from_tensor(Q.view(B * H, S, d), block_shape=[1, 128, 128])
    K_desc = TensorDescriptor.from_tensor(K.view(B * H, S, d), block_shape=[1, 128, 128])
    V_desc = TensorDescriptor.from_tensor(V.view(B * H, S, d), block_shape=[1, 128, 128])
    O_desc = TensorDescriptor.from_tensor(O.view(B * H, S, d), block_shape=[1, 128, 128])
    dO_desc = TensorDescriptor.from_tensor(dO.view(B * H, S, d), block_shape=[1, 128, 128])
    
    grid = (
        triton.cdiv(S, 128),
        B * H,
    )
    
    total_S = B * H * S
    L_s1 = L.stride(1)
    
    dQ_s2 = dQ.stride(2)
    dQ_s3 = dQ.stride(3)
    
    dK_s2 = dK.stride(2)
    dK_s3 = dK.stride(3)
    
    dV_s2 = dV.stride(2)
    dV_s3 = dV.stride(3)
    
    _bwd_dk_dv_kernel[grid](
        Q_desc, K_desc, V_desc, O_desc, dO_desc, L, dK, dV,
        S, scale, total_S, L_s1,
        dK_s2, dK_s3, dV_s2, dV_s3,
        num_warps=4, num_stages=3,
    )
    
    _bwd_dq_kernel[grid](
        Q_desc, K_desc, V_desc, O_desc, dO_desc, L, dQ,
        S, scale, total_S, L_s1,
        dQ_s2, dQ_s3,
        num_warps=4, num_stages=3,
    )