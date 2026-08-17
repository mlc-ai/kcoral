import math
import torch
import triton
import triton.language as tl

@triton.jit
def _bwd_dq_kernel(
    Q, K, V, O, dO, dQ, L,
    seqlen, scale,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    stride_lb, stride_lh, stride_ls,
    H,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    D_HEAD: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    b_id = pid_bh // H
    h_id = pid_bh % H
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D_HEAD)
    
    q_ptrs = Q + b_id * stride_qb + h_id * stride_qh + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    o_ptrs = O + b_id * stride_ob + h_id * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    do_ptrs = dO + b_id * stride_dob + h_id * stride_doh + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod
    dq_ptrs = dQ + b_id * stride_dqb + h_id * stride_dqh + offs_m[:, None] * stride_dqs + offs_d[None, :] * stride_dqd
    
    l_ptrs = L + b_id * stride_lb + h_id * stride_lh + offs_m * stride_ls
    
    mask_m = offs_m < seqlen
    
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
    do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
    l = tl.load(l_ptrs, mask=mask_m, other=0.0)
    
    # Precompute D_i = sum(dO_i * O_i) for the current Q block
    o_f32 = o.to(tl.float32)
    do_f32 = do.to(tl.float32)
    d_i = tl.sum(o_f32 * do_f32, axis=1)  # shape (BLOCK_M,)
    
    dq = tl.zeros([BLOCK_M, D_HEAD], dtype=tl.float32)
    
    k_ptrs = K + b_id * stride_kb + h_id * stride_kh + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = V + b_id * stride_vb + h_id * stride_vh + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
    
    for start_n in range(0, seqlen, BLOCK_N):
        offs_n_curr = start_n + offs_n
        mask_n = offs_n_curr < seqlen
        
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        
        # S = Q @ K^T
        qk = tl.dot(q, k.T) * scale
        
        # P = exp(S - L)
        p = tl.exp(qk - l[:, None])
        p = tl.where((mask_m[:, None]) & (mask_n[None, :]), p, 0.0)
        
        # dP = dO @ V^T
        dp = tl.dot(do, v.T)
        
        # dS = P * (dP - D)
        ds = p * (dp - d_i[:, None]) * scale
        
        # dQ += dS @ K
        ds = ds.to(k.dtype)
        dq += tl.dot(ds, k)
        
        k_ptrs += BLOCK_N * stride_ks
        v_ptrs += BLOCK_N * stride_vs
        
    tl.store(dq_ptrs, dq.to(dQ.dtype.element_ty), mask=mask_m[:, None])


@triton.jit
def _bwd_dk_dv_kernel(
    Q, K, V, O, dO, dK, dV, L,
    seqlen, scale,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    stride_lb, stride_lh, stride_ls,
    H,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    D_HEAD: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    b_id = pid_bh // H
    h_id = pid_bh % H
    
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D_HEAD)
    
    mask_n = offs_n < seqlen
    
    k_ptrs = K + b_id * stride_kb + h_id * stride_kh + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = V + b_id * stride_vb + h_id * stride_vh + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
    dk_ptrs = dK + b_id * stride_dkb + h_id * stride_dkh + offs_n[:, None] * stride_dks + offs_d[None, :] * stride_dkd
    dv_ptrs = dV + b_id * stride_dvb + h_id * stride_dvh + offs_n[:, None] * stride_dvs + offs_d[None, :] * stride_dvd
    
    k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
    
    dk = tl.zeros([BLOCK_N, D_HEAD], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, D_HEAD], dtype=tl.float32)
    
    q_ptrs = Q + b_id * stride_qb + h_id * stride_qh + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    o_ptrs = O + b_id * stride_ob + h_id * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    do_ptrs = dO + b_id * stride_dob + h_id * stride_doh + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod
    
    l_ptrs = L + b_id * stride_lb + h_id * stride_lh + offs_m * stride_ls
    
    for start_m in range(0, seqlen, BLOCK_M):
        offs_m_curr = start_m + offs_m
        mask_m = offs_m_curr < seqlen
        
        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
        o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
        do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
        l = tl.load(l_ptrs, mask=mask_m, other=0.0)
        
        o_f32 = o.to(tl.float32)
        do_f32 = do.to(tl.float32)
        d_i = tl.sum(o_f32 * do_f32, axis=1)  # shape (BLOCK_M,)
        
        # S = Q @ K^T
        qk = tl.dot(q, k.T) * scale
        
        # P = exp(S - L)
        p = tl.exp(qk - l[:, None])
        p = tl.where((mask_m[:, None]) & (mask_n[None, :]), p, 0.0)
        
        # dP = dO @ V^T
        dp = tl.dot(do, v.T)
        
        # dS = P * (dP - D)
        ds = p * (dp - d_i[:, None]) * scale
        
        # dV += P^T @ dO
        p_t = p.T.to(do.dtype)
        dv += tl.dot(p_t, do)
        
        # dK += dS^T @ Q
        ds_t = ds.T.to(q.dtype)
        dk += tl.dot(ds_t, q)
        
        q_ptrs += BLOCK_M * stride_qs
        o_ptrs += BLOCK_M * stride_os
        do_ptrs += BLOCK_M * stride_dos
        l_ptrs += BLOCK_M * stride_ls
        
    tl.store(dk_ptrs, dk.to(dK.dtype.element_ty), mask=mask_n[:, None])
    tl.store(dv_ptrs, dv.to(dV.dtype.element_ty), mask=mask_n[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes the backward pass for multi-head attention (non-causal).
    Input tensors are (B, H, S, d) and L is (B, H, S) or (B, H, S, 1).
    Writes output into preallocated dQ, dK, dV tensors.
    """
    with torch.cuda.device(Q.device):
        B, H_dim, seqlen, D_HEAD = Q.shape
        scale = 1.0 / math.sqrt(D_HEAD)
        
        BLOCK_M = 64
        BLOCK_N = 64
        
        grid_dq = (triton.cdiv(seqlen, BLOCK_M), B * H_dim)
        grid_dk_dv = (triton.cdiv(seqlen, BLOCK_N), B * H_dim)
        
        # Handle L tensor strides gracefully regardless of whether it was unsqueezed or not
        if L.dim() == 4:
            stride_lb, stride_lh, stride_ls, _ = L.stride()
        else:
            stride_lb, stride_lh, stride_ls = L.stride()
        
        _bwd_dq_kernel[grid_dq](
            Q, K, V, O, dO, dQ, L,
            seqlen, scale,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
            dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
            stride_lb, stride_lh, stride_ls,
            H_dim,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            D_HEAD=D_HEAD,
            num_warps=4,
            num_stages=3
        )
        
        _bwd_dk_dv_kernel[grid_dk_dv](
            Q, K, V, O, dO, dK, dV, L,
            seqlen, scale,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
            dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
            dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
            stride_lb, stride_lh, stride_ls,
            H_dim,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            D_HEAD=D_HEAD,
            num_warps=4,
            num_stages=3
        )