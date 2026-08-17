import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
import math


@triton.jit
def _bwd_dq_kernel(
    Q_desc, K_desc, V_desc, O_desc, dO_desc, L, dQ,
    S, scale, total_S, L_s1,
    dQ_s2, dQ_s3, S_times_dQ_s2, dQ_s2_times_64,
):
    m_blk = tl.program_id(0)
    bh_id = tl.program_id(1)
    
    s_idx = m_blk * 128
    row_q = bh_id * S_times_dQ_s2 + s_idx * dQ_s2
    
    q_0 = Q_desc.load([row_q, 0]).to(tl.float32)
    q_1 = Q_desc.load([row_q, 64]).to(tl.float32)
    
    do_0 = dO_desc.load([row_q, 0]).to(tl.float32)
    do_1 = dO_desc.load([row_q, 64]).to(tl.float32)
    
    o_0 = O_desc.load([row_q, 0]).to(tl.float32)
    o_1 = O_desc.load([row_q, 64]).to(tl.float32)
    
    off_m = tl.arange(0, 128)
    L_ptr = L + bh_id * L_s1 + s_idx
    l = tl.load(L_ptr + off_m, mask=(bh_id * S + s_idx + off_m) < total_S, other=-float('inf'))
    
    d_val = tl.sum(do_0 * o_0, axis=1) + tl.sum(do_1 * o_1, axis=1)
    d_exp = d_val[:, None]
    
    dq_0 = tl.zeros((128, 64), dtype=tl.float32)
    dq_1 = tl.zeros((128, 64), dtype=tl.float32)
    
    skip_blks = max(0, (m_blk * 128 - 127) // 128)
    n_blk_start = (m_blk * 128 - 127) // 128
    skip_blks_step = 1
    
    for n_blk in range(n_blk_start, min(m_blk, total_blks - 1) + 1, skip_blks_step):
        k_idx = n_blk * 128
        
        if k_idx + 127 < s_idx:
            continue
            
        row_k = bh_id * S_times_dQ_s2 + k_idx * dQ_s2
        
        k_0 = K_desc.load([row_k, 0]).to(tl.float32)
        k_1 = K_desc.load([row_k, 64]).to(tl.float32)
        
        v_0 = V_desc.load([row_k, 0]).to(tl.float32)
        v_1 = V_desc.load([row_k, 64]).to(tl.float32)
        
        s_gem = tl.zeros((128, 128), dtype=tl.float32)
        s_gem = tl.dot(q_0, k_0.T, s_gem)
        s_gem = tl.dot(q_1, k_1.T, s_gem)
        
        s = s_gem * scale
        
        off_n = tl.arange(0, 128)
        if k_idx <= s_idx:
            valid = (k_idx + off_n[None, :]) <= (s_idx + off_m[:, None])
            s = tl.where(valid, s, -float('inf'))
            
        p = tl.exp(s - l[:, None])
        if k_idx <= s_idx:
            p = tl.where(valid, p, 0.0)
            
        dp_gem = tl.zeros((128, 128), dtype=tl.float32)
        dp_gem = tl.dot(do_0, v_0.T, dp_gem)
        dp_gem = tl.dot(do_1, v_1.T, dp_gem)
        
        ds = p * (dp_gem - d_exp) * scale
        
        dq_0 = tl.dot(ds, k_0, dq_0)
        dq_1 = tl.dot(ds, k_1, dq_1)
        
    off_row = bh_id * S + s_idx + off_m
    mask = off_row < total_S
    
    off_n = tl.arange(0, 64)
    
    dQ_ptr_0 = dQ + off_row[:, None] * dQ_s2 + off_n[None, :] * dQ_s3
    tl.store(dQ_ptr_0, dq_0.to(tl.bfloat16), mask=mask[:, None])
    
    dQ_ptr_1 = dQ + off_row[:, None] * dQ_s2 + (off_n[None, :] + 64) * dQ_s3
    tl.store(dQ_ptr_1, dq_1.to(tl.bfloat16), mask=mask[:, None])


@triton.jit
def _bwd_dk_dv_kernel(
    Q_desc, K_desc, V_desc, O_desc, dO_desc, L, dK, dV,
    S, scale, total_S, L_s1,
    dK_s2, dK_s3, dV_s2, dV_s3,
    S_times_dK_s2, dK_s2_times_64, S_times_dV_s2, dV_s2_times_64,
):
    n_blk = tl.program_id(0)
    bh_id = tl.program_id(1)
    
    k_idx = n_blk * 128
    row_k = bh_id * S_times_dK_s2 + k_idx * dK_s2
    
    k_0 = K_desc.load([row_k, 0]).to(tl.float32)
    k_1 = K_desc.load([row_k, 64]).to(tl.float32)
    
    v_0 = V_desc.load([row_k, 0]).to(tl.float32)
    v_1 = V_desc.load([row_k, 64]).to(tl.float32)
    
    dk_0 = tl.zeros((128, 64), dtype=tl.float32)
    dk_1 = tl.zeros((128, 64), dtype=tl.float32)
    dv_0 = tl.zeros((128, 64), dtype=tl.float32)
    dv_1 = tl.zeros((128, 64), dtype=tl.float32)
    
    total_blks = tl.cdiv(S, 128)
    
    for i_blk in range(n_blk, total_blks):
        s_idx = i_blk * 128
        
        if s_idx < k_idx:
            continue
            
        row_q = bh_id * S_times_dK_s2 + s_idx * dK_s2
        
        q_0 = Q_desc.load([row_q, 0]).to(tl.float32)
        q_1 = Q_desc.load([row_q, 64]).to(tl.float32)
        
        do_0 = dO_desc.load([row_q, 0]).to(tl.float32)
        do_1 = dO_desc.load([row_q, 64]).to(tl.float32)
        
        o_0 = O_desc.load([row_q, 0]).to(tl.float32)
        o_1 = O_desc.load([row_q, 64]).to(tl.float32)
        
        off_m = tl.arange(0, 128)
        L_ptr = L + bh_id * L_s1 + s_idx
        l = tl.load(L_ptr + off_m, mask=(bh_id * S + s_idx + off_m) < total_S, other=-float('inf'))
        
        d_val = tl.sum(do_0 * o_0, axis=1) + tl.sum(do_1 * o_1, axis=1)
        d_exp = d_val[:, None]
        
        s_gem = tl.zeros((128, 128), dtype=tl.float32)
        s_gem = tl.dot(q_0, k_0.T, s_gem)
        s_gem = tl.dot(q_1, k_1.T, s_gem)
        
        s = s_gem * scale
        
        off_n = tl.arange(0, 128)
        if s_idx <= k_idx + 127:
            valid = (k_idx + off_n[None, :]) <= (s_idx + off_m[:, None])
            s = tl.where(valid, s, -float('inf'))
            
        p = tl.exp(s - l[:, None])
        if s_idx <= k_idx + 127:
            p = tl.where(valid, p, 0.0)
            
        dp_gem = tl.zeros((128, 128), dtype=tl.float32)
        dp_gem = tl.dot(do_0, v_0.T, dp_gem)
        dp_gem = tl.dot(do_1, v_1.T, dp_gem)
        
        ds = p * (dp_gem - d_exp) * scale
        
        dk_0 = tl.dot(ds.T, q_0, dk_0)
        dk_1 = tl.dot(ds.T, q_1, dk_1)
        
        dv_0 = tl.dot(p.T, do_0, dv_0)
        dv_1 = tl.dot(p.T, do_1, dv_1)
        
    off_row = bh_id * S + k_idx + off_n
    mask = off_row < total_S
    
    off_m = tl.arange(0, 64)
    
    dK_ptr_0 = dK + off_row[:, None] * dK_s2 + off_m[None, :] * dK_s3
    tl.store(dK_ptr_0, dk_0.to(tl.bfloat16), mask=mask[:, None])
    
    dK_ptr_1 = dK + off_row[:, None] * dK_s2 + (off_m[None, :] + 64) * dK_s3
    tl.store(dK_ptr_1, dk_1.to(tl.bfloat16), mask=mask[:, None])
    
    dV_ptr_0 = dV + off_row[:, None] * dV_s2 + off_m[None, :] * dV_s3
    tl.store(dV_ptr_0, dv_0.to(tl.bfloat16), mask=mask[:, None])
    
    dV_ptr_1 = dV + off_row[:, None] * dV_s2 + (off_m[None, :] + 64) * dV_s3
    tl.store(dV_ptr_1, dv_1.to(tl.bfloat16), mask=mask[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute FlashAttention backward pass for causal multi-head attention."""
    torch.cuda.set_device(Q.device)
    
    B = Q.shape[0]
    H = Q.shape[1]
    S = Q.shape[2]
    d = Q.shape[3]
    
    scale = 1.0 / math.sqrt(d)
    
    Q_desc = TensorDescriptor.from_tensor(Q.view(B * H * S, d), [128, 64])
    K_desc = TensorDescriptor.from_tensor(K.view(B * H * S, d), [128, 64])
    V_desc = TensorDescriptor.from_tensor(V.view(B * H * S, d), [128, 64])
    O_desc = TensorDescriptor.from_tensor(O.view(B * H * S, d), [128, 64])
    dO_desc = TensorDescriptor.from_tensor(dO.view(B * H * S, d), [128, 64])
    
    grid = (
        triton.cdiv(S, 128),
        B * H,
    )
    
    total_S = B * H * S
    L_s1 = L.stride(1)
    
    dQ_s2 = dQ.stride(2)
    dQ_s3 = dQ.stride(3)
    S_times_dQ_s2 = S * dQ_s2
    dQ_s2_times_64 = dQ_s2 * 64
    
    dK_s2 = dK.stride(2)
    dK_s3 = dK.stride(3)
    S_times_dK_s2 = S * dK_s2
    dK_s2_times_64 = dK_s2 * 64
    
    dV_s2 = dV.stride(2)
    dV_s3 = dV.stride(3)
    S_times_dV_s2 = S * dV_s2
    dV_s2_times_64 = dV_s2 * 64
    
    _bwd_dk_dv_kernel[grid](
        Q_desc, K_desc, V_desc, O_desc, dO_desc, L, dK, dV,
        S, scale, total_S, L_s1,
        dK_s2, dK_s3, dV_s2, dV_s3,
        S_times_dK_s2, dK_s2_times_64, S_times_dV_s2, dV_s2_times_64,
        num_warps=4, num_stages=3,
    )
    
    _bwd_dq_kernel[grid](
        Q_desc, K_desc, V_desc, O_desc, dO_desc, L, dQ,
        S, scale, total_S, L_s1,
        dQ_s2, dQ_s3, S_times_dQ_s2, dQ_s2_times_64,
        num_warps=4, num_stages=3,
    )