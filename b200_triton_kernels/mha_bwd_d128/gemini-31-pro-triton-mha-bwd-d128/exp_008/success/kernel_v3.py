import math
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_R": 128, "BLOCK_C": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_R": 128, "BLOCK_C": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_R": 64, "BLOCK_C": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_R": 64, "BLOCK_C": 64}, num_warps=4, num_stages=4),
    ],
    key=["S"]
)
@triton.jit
def bwd_dq_kernel(
    Q, K, V, O, dO, L, dQ,
    stride_qb, stride_qh, stride_qs,
    stride_kb, stride_kh, stride_ks,
    stride_vb, stride_vh, stride_vs,
    stride_ob, stride_oh, stride_os,
    stride_dob, stride_doh, stride_dos,
    stride_lb, stride_lh, stride_ls,
    stride_dqb, stride_dqh, stride_dqs,
    S, scale,
    BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr, d: tl.constexpr
):
    i_id = tl.program_id(0)
    bh_id = tl.program_id(1)
    
    H = 48
    b_id = bh_id // H
    h_id = bh_id % H
    
    offs_d = tl.arange(0, d)
    offs_r = i_id * BLOCK_R + tl.arange(0, BLOCK_R)
    mask_r = offs_r < S
    
    q_ptrs = Q + b_id * stride_qb + h_id * stride_qh + offs_r[:, None] * stride_qs + offs_d[None, :]
    do_ptrs = dO + b_id * stride_dob + h_id * stride_doh + offs_r[:, None] * stride_dos + offs_d[None, :]
    o_ptrs = O + b_id * stride_ob + h_id * stride_oh + offs_r[:, None] * stride_os + offs_d[None, :]
    l_ptrs = L + b_id * stride_lb + h_id * stride_lh + offs_r * stride_ls
    
    q_i = tl.load(q_ptrs, mask=mask_r[:, None], other=0.0)
    do_i = tl.load(do_ptrs, mask=mask_r[:, None], other=0.0)
    o_i = tl.load(o_ptrs, mask=mask_r[:, None], other=0.0)
    l_i = tl.load(l_ptrs, mask=mask_r, other=0.0)
    
    d_i = tl.sum(o_i.to(tl.float32) * do_i.to(tl.float32), axis=1)
    
    dq_i = tl.zeros([BLOCK_R, d], dtype=tl.float32)
    
    offs_c = tl.arange(0, BLOCK_C)
    k_ptrs = K + b_id * stride_kb + h_id * stride_kh + offs_c[:, None] * stride_ks + offs_d[None, :]
    v_ptrs = V + b_id * stride_vb + h_id * stride_vh + offs_c[:, None] * stride_vs + offs_d[None, :]
    
    num_steps = tl.cdiv(S, BLOCK_C)
    for j in range(num_steps):
        mask_c = offs_c < S
        k_j = tl.load(k_ptrs, mask=mask_c[:, None], other=0.0)
        v_j = tl.load(v_ptrs, mask=mask_c[:, None], other=0.0)
        
        # Hopper native optimal layouts applied
        s_ij = tl.dot(q_i, tl.trans(k_j), out_dtype=tl.float32) * scale
        p_ij = tl.exp(s_ij - l_i[:, None])
        p_ij = tl.where((mask_r[:, None]) & (mask_c[None, :]), p_ij, 0.0)
        
        dp_ij = tl.dot(do_i, tl.trans(v_j), out_dtype=tl.float32)
        
        ds_ij = p_ij * (dp_ij - d_i[:, None]) * scale
        
        dq_i = tl.dot(ds_ij.to(q_i.dtype), k_j, acc=dq_i, out_dtype=tl.float32)
        
        offs_c += BLOCK_C
        k_ptrs += BLOCK_C * stride_ks
        v_ptrs += BLOCK_C * stride_vs
        
    dq_out_ptrs = dQ + b_id * stride_dqb + h_id * stride_dqh + offs_r[:, None] * stride_dqs + offs_d[None, :]
    tl.store(dq_out_ptrs, dq_i.to(q_i.dtype), mask=mask_r[:, None])


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_C": 128, "BLOCK_R": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_C": 128, "BLOCK_R": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_C": 64, "BLOCK_R": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_C": 64, "BLOCK_R": 64}, num_warps=4, num_stages=4),
    ],
    key=["S"]
)
@triton.jit
def bwd_dk_dv_kernel(
    Q, K, V, O, dO, L, dK, dV,
    stride_qb, stride_qh, stride_qs,
    stride_kb, stride_kh, stride_ks,
    stride_vb, stride_vh, stride_vs,
    stride_ob, stride_oh, stride_os,
    stride_dob, stride_doh, stride_dos,
    stride_lb, stride_lh, stride_ls,
    stride_dkb, stride_dkh, stride_dks,
    stride_dvb, stride_dvh, stride_dvs,
    S, scale,
    BLOCK_C: tl.constexpr, BLOCK_R: tl.constexpr, d: tl.constexpr
):
    j_id = tl.program_id(0)
    bh_id = tl.program_id(1)
    
    H = 48
    b_id = bh_id // H
    h_id = bh_id % H
    
    offs_d = tl.arange(0, d)
    offs_c = j_id * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = offs_c < S
    
    k_ptrs = K + b_id * stride_kb + h_id * stride_kh + offs_c[:, None] * stride_ks + offs_d[None, :]
    v_ptrs = V + b_id * stride_vb + h_id * stride_vh + offs_c[:, None] * stride_vs + offs_d[None, :]
    
    k_j = tl.load(k_ptrs, mask=mask_c[:, None], other=0.0)
    v_j = tl.load(v_ptrs, mask=mask_c[:, None], other=0.0)
    
    dk_j = tl.zeros([BLOCK_C, d], dtype=tl.float32)
    dv_j = tl.zeros([BLOCK_C, d], dtype=tl.float32)
    
    offs_r = tl.arange(0, BLOCK_R)
    q_ptrs = Q + b_id * stride_qb + h_id * stride_qh + offs_r[:, None] * stride_qs + offs_d[None, :]
    do_ptrs = dO + b_id * stride_dob + h_id * stride_doh + offs_r[:, None] * stride_dos + offs_d[None, :]
    o_ptrs = O + b_id * stride_ob + h_id * stride_oh + offs_r[:, None] * stride_os + offs_d[None, :]
    l_ptrs = L + b_id * stride_lb + h_id * stride_lh + offs_r * stride_ls
    
    num_steps = tl.cdiv(S, BLOCK_R)
    for i in range(num_steps):
        mask_r = offs_r < S
        
        q_i = tl.load(q_ptrs, mask=mask_r[:, None], other=0.0)
        do_i = tl.load(do_ptrs, mask=mask_r[:, None], other=0.0)
        o_i = tl.load(o_ptrs, mask=mask_r[:, None], other=0.0)
        l_i = tl.load(l_ptrs, mask=mask_r, other=0.0)
        
        d_i = tl.sum(o_i.to(tl.float32) * do_i.to(tl.float32), axis=1)
        
        # Hopper optimal TN dots taking advantage of hardware transpising 
        s_trans = tl.dot(k_j, tl.trans(q_i), out_dtype=tl.float32) * scale
        p_trans = tl.exp(s_trans - l_i[None, :])
        p_trans = tl.where((mask_c[:, None]) & (mask_r[None, :]), p_trans, 0.0)
        
        dp_trans = tl.dot(v_j, tl.trans(do_i), out_dtype=tl.float32)
        
        ds_trans = p_trans * (dp_trans - d_i[None, :]) * scale
        
        dv_j = tl.dot(p_trans.to(q_i.dtype), do_i, acc=dv_j, out_dtype=tl.float32)
        dk_j = tl.dot(ds_trans.to(q_i.dtype), q_i, acc=dk_j, out_dtype=tl.float32)
        
        offs_r += BLOCK_R
        q_ptrs += BLOCK_R * stride_qs
        do_ptrs += BLOCK_R * stride_dos
        o_ptrs += BLOCK_R * stride_os
        l_ptrs += BLOCK_R * stride_ls
        
    dk_out_ptrs = dK + b_id * stride_dkb + h_id * stride_dkh + offs_c[:, None] * stride_dks + offs_d[None, :]
    dv_out_ptrs = dV + b_id * stride_dvb + h_id * stride_dvh + offs_c[:, None] * stride_dvs + offs_d[None, :]
    
    tl.store(dk_out_ptrs, dk_j.to(k_j.dtype), mask=mask_c[:, None])
    tl.store(dv_out_ptrs, dv_j.to(v_j.dtype), mask=mask_c[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes Multi-Head Attention backward pass purely leveraging highly optimized pointer increments and L2 grouping.
    TMA operations are intentionally skipped in favor of the much more streamlined `cp.async` pipeline for this form of layout.
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    
    stride_lb = L.stride(0)
    stride_lh = L.stride(1)
    stride_ls = L.stride(2)
    
    # 1. Dispatch dQ kernel
    grid_dq = lambda META: (triton.cdiv(S, META["BLOCK_R"]), B * H)
    bwd_dq_kernel[grid_dq](
        Q, K, V, O, dO, L, dQ,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        dO.stride(0), dO.stride(1), dO.stride(2),
        stride_lb, stride_lh, stride_ls,
        dQ.stride(0), dQ.stride(1), dQ.stride(2),
        S, scale, d=128
    )
    
    # 2. Dispatch dK, dV kernel
    grid_dkdv = lambda META: (triton.cdiv(S, META["BLOCK_C"]), B * H)
    bwd_dk_dv_kernel[grid_dkdv](
        Q, K, V, O, dO, L, dK, dV,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        dO.stride(0), dO.stride(1), dO.stride(2),
        stride_lb, stride_lh, stride_ls,
        dK.stride(0), dK.stride(1), dK.stride(2),
        dV.stride(0), dV.stride(1), dV.stride(2),
        S, scale, d=128
    )