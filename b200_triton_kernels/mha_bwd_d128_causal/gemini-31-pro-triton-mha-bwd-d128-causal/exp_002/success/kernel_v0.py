import math
import torch
import triton
import triton.language as tl

configs_dq = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
]

configs_dk = [
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
]


@triton.autotune(configs=configs_dq, key=['S'])
@triton.jit
def _bwd_kernel_dq(
    Q, K, V, O, DO, DQ, L,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    stride_lb, stride_lh, stride_ls,
    H, S, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D_HEAD: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    b = pid_bh // H
    h = pid_bh % H
    
    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = off_m < S
    
    off_n_base = tl.arange(0, BLOCK_N)
    off_d = tl.arange(0, D_HEAD)
    
    q_ptr = Q + b * stride_qb + h * stride_qh + off_m[:, None] * stride_qs + off_d[None, :] * stride_qd
    do_ptr = DO + b * stride_dob + h * stride_doh + off_m[:, None] * stride_dos + off_d[None, :] * stride_dod
    o_ptr = O + b * stride_ob + h * stride_oh + off_m[:, None] * stride_os + off_d[None, :] * stride_od
    dq_ptr = DQ + b * stride_dqb + h * stride_dqh + off_m[:, None] * stride_dqs + off_d[None, :] * stride_dqd
    
    q = tl.load(q_ptr, mask=mask_m[:, None], other=0.0)
    do = tl.load(do_ptr, mask=mask_m[:, None], other=0.0)
    o = tl.load(o_ptr, mask=mask_m[:, None], other=0.0).to(tl.float32)
    
    # Delta (D) is computed on the fly per query block to avoid intermediate allocations
    do_f32 = do.to(tl.float32)
    delta = tl.sum(o * do_f32, axis=1)
    
    l_ptr = L + b * stride_lb + h * stride_lh + off_m * stride_ls
    l = tl.load(l_ptr, mask=mask_m, other=0.0)
    
    dq = tl.zeros([BLOCK_M, D_HEAD], dtype=tl.float32)
    
    # Causal sequence limit handling
    max_n = tl.minimum(S, (pid_m + 1) * BLOCK_M)
    num_n_blocks = tl.cdiv(max_n, BLOCK_N)
    
    k_ptr_base = K + b * stride_kb + h * stride_kh + off_d[None, :] * stride_kd
    v_ptr_base = V + b * stride_vb + h * stride_vh + off_d[None, :] * stride_vd
    
    for n_idx in range(num_n_blocks):
        off_n = n_idx * BLOCK_N + off_n_base
        mask_n = off_n < S
        
        k_ptr = k_ptr_base + off_n[:, None] * stride_ks
        v_ptr = v_ptr_base + off_n[:, None] * stride_vs
        
        k = tl.load(k_ptr, mask=mask_n[:, None], other=0.0)
        v = tl.load(v_ptr, mask=mask_n[:, None], other=0.0)
        
        qk = tl.dot(q, k.T, out_dtype=tl.float32) * scale
        
        valid_mask = (off_m[:, None] >= off_n[None, :]) & mask_m[:, None] & mask_n[None, :]
        qk = tl.where(valid_mask, qk, float('-inf'))
        
        p = tl.math.exp(qk - l[:, None])
        p = tl.where(valid_mask, p, 0.0)
        
        dp = tl.dot(do, v.T, out_dtype=tl.float32)
        
        ds = p * (dp - delta[:, None]) * scale
        
        dq += tl.dot(ds.to(q.dtype), k, out_dtype=tl.float32)
        
    tl.store(dq_ptr, dq.to(DQ.dtype.element_ty), mask=mask_m[:, None])


@triton.autotune(configs=configs_dk, key=['S'])
@triton.jit
def _bwd_kernel_dk_dv(
    Q, K, V, O, DO, DK, DV, L,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    stride_lb, stride_lh, stride_ls,
    H, S, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D_HEAD: tl.constexpr
):
    pid_n = tl.program_id(0) 
    pid_bh = tl.program_id(1)
    
    b = pid_bh // H
    h = pid_bh % H
    
    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = off_n < S
    
    off_m_base = tl.arange(0, BLOCK_M)
    off_d = tl.arange(0, D_HEAD)
    
    k_ptr = K + b * stride_kb + h * stride_kh + off_n[:, None] * stride_ks + off_d[None, :] * stride_kd
    v_ptr = V + b * stride_vb + h * stride_vh + off_n[:, None] * stride_vs + off_d[None, :] * stride_vd
    
    dk_ptr = DK + b * stride_dkb + h * stride_dkh + off_n[:, None] * stride_dks + off_d[None, :] * stride_dkd
    dv_ptr = DV + b * stride_dvb + h * stride_dvh + off_n[:, None] * stride_dvs + off_d[None, :] * stride_dvd
    
    k = tl.load(k_ptr, mask=mask_n[:, None], other=0.0)
    v = tl.load(v_ptr, mask=mask_n[:, None], other=0.0)
    
    dk = tl.zeros([BLOCK_N, D_HEAD], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, D_HEAD], dtype=tl.float32)
    
    start_m_idx = (pid_n * BLOCK_N) // BLOCK_M
    num_m_blocks = tl.cdiv(S, BLOCK_M)
    
    q_ptr_base = Q + b * stride_qb + h * stride_qh + off_d[None, :] * stride_qd
    do_ptr_base = DO + b * stride_dob + h * stride_doh + off_d[None, :] * stride_dod
    o_ptr_base = O + b * stride_ob + h * stride_oh + off_d[None, :] * stride_od
    
    for m_idx in range(start_m_idx, num_m_blocks):
        off_m = m_idx * BLOCK_M + off_m_base
        mask_m = off_m < S
        
        q_ptr = q_ptr_base + off_m[:, None] * stride_qs
        do_ptr = do_ptr_base + off_m[:, None] * stride_dos
        o_ptr = o_ptr_base + off_m[:, None] * stride_os
        
        q = tl.load(q_ptr, mask=mask_m[:, None], other=0.0)
        do = tl.load(do_ptr, mask=mask_m[:, None], other=0.0)
        o = tl.load(o_ptr, mask=mask_m[:, None], other=0.0).to(tl.float32)
        
        delta = tl.sum(o * do.to(tl.float32), axis=1)
        
        l_ptr = L + b * stride_lb + h * stride_lh + off_m * stride_ls
        l = tl.load(l_ptr, mask=mask_m, other=0.0)
        
        qk = tl.dot(q, k.T, out_dtype=tl.float32) * scale
        
        valid_mask = (off_m[:, None] >= off_n[None, :]) & mask_m[:, None] & mask_n[None, :]
        qk = tl.where(valid_mask, qk, float('-inf'))
        
        p = tl.math.exp(qk - l[:, None])
        p = tl.where(valid_mask, p, 0.0)
        
        dv += tl.dot(p.to(q.dtype).T, do, out_dtype=tl.float32)
        
        dp = tl.dot(do, v.T, out_dtype=tl.float32)
        
        ds = p * (dp - delta[:, None]) * scale
        
        dk += tl.dot(ds.to(q.dtype).T, q, out_dtype=tl.float32)
        
    tl.store(dk_ptr, dk.to(DK.dtype.element_ty), mask=mask_n[:, None])
    tl.store(dv_ptr, dv.to(DV.dtype.element_ty), mask=mask_n[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes backward pass for Causal Multi-Head Attention without intermediate allocations.
    """
    B, H, S, D = Q.shape
    scale = 1.0 / math.sqrt(D)
    torch.cuda.set_device(Q.device)
    
    grid_dq = lambda META: (
        triton.cdiv(S, META['BLOCK_M']),
        B * H
    )
    
    grid_dk = lambda META: (
        triton.cdiv(S, META['BLOCK_N']),
        B * H
    )

    _bwd_kernel_dq[grid_dq](
        Q, K, V, O, dO, dQ, L,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        H, S, scale,
        D_HEAD=D,
    )

    _bwd_kernel_dk_dv[grid_dk](
        Q, K, V, O, dO, dK, dV, L,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        H, S, scale,
        D_HEAD=D,
    )