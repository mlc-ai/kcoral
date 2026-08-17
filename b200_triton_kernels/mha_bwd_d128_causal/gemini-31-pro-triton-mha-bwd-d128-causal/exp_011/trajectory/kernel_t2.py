import math
import torch
import triton
import triton.language as tl


@triton.jit
def bwd_dq_kernel(
    Q, K, V, O, dO, dQ, L,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    stride_lb, stride_lh, stride_ls,
    S, sm_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D_HEAD: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)
    
    start_m = pid_m * BLOCK_M
    off_m = start_m + tl.arange(0, BLOCK_M)
    mask_m = off_m < S
    
    q_offset = pid_b * stride_qb + pid_h * stride_qh
    k_offset = pid_b * stride_kb + pid_h * stride_kh
    v_offset = pid_b * stride_vb + pid_h * stride_vh
    o_offset = pid_b * stride_ob + pid_h * stride_oh
    do_offset = pid_b * stride_dob + pid_h * stride_doh
    dq_offset = pid_b * stride_dqb + pid_h * stride_dqh
    l_offset = pid_b * stride_lb + pid_h * stride_lh
    
    q_ptrs = Q + q_offset + off_m[:, None] * stride_qs + tl.arange(0, D_HEAD)[None, :] * stride_qd
    do_ptrs = dO + do_offset + off_m[:, None] * stride_dos + tl.arange(0, D_HEAD)[None, :] * stride_dod
    out_ptrs = O + o_offset + off_m[:, None] * stride_os + tl.arange(0, D_HEAD)[None, :] * stride_od
    l_ptrs = L + l_offset + off_m * stride_ls
    
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
    out = tl.load(out_ptrs, mask=mask_m[:, None], other=0.0)
    l = tl.load(l_ptrs, mask=mask_m, other=0.0)
    
    # D_i = sum(O_i * dO_i) cached in registers globally for the M scope
    d = tl.sum(out.to(tl.float32) * do.to(tl.float32), axis=1)
    del out
    
    dq = tl.zeros([BLOCK_M, D_HEAD], dtype=tl.float32)
    
    # Calculate bounding logic directly to limit zero-padding overhead
    n_blocks = (start_m + BLOCK_M + BLOCK_N - 1) // BLOCK_N
    max_start_n = (S + BLOCK_N - 1) // BLOCK_N
    if n_blocks > max_start_n:
        n_blocks = max_start_n
        
    off_n = tl.arange(0, BLOCK_N)
    k_ptrs = K + k_offset + off_n[:, None] * stride_ks + tl.arange(0, D_HEAD)[None, :] * stride_kd
    v_ptrs = V + v_offset + off_n[:, None] * stride_vs + tl.arange(0, D_HEAD)[None, :] * stride_vd
    
    for start_n in range(0, n_blocks * BLOCK_N, BLOCK_N):
        off_n_curr = start_n + off_n
        mask_n = off_n_curr < S
        
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        
        qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * sm_scale
        qk_l = qk - l[:, None]
        
        is_causal_boundary = start_n + BLOCK_N > start_m
        if is_causal_boundary:
            qk_l = tl.where(off_m[:, None] >= off_n_curr[None, :], qk_l, float("-inf"))
            
        if start_m + BLOCK_M > S or start_n + BLOCK_N > S:
            qk_l = tl.where(mask_m[:, None] & mask_n[None, :], qk_l, float("-inf"))
            
        p = tl.exp(qk_l)
        
        dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32)
        ds = p * (dp - d[:, None]) * sm_scale
        
        dq += tl.dot(ds.to(q.dtype), k, out_dtype=tl.float32)
        
        k_ptrs += BLOCK_N * stride_ks
        v_ptrs += BLOCK_N * stride_vs
        
    dq_ptrs = dQ + dq_offset + off_m[:, None] * stride_dqs + tl.arange(0, D_HEAD)[None, :] * stride_dqd
    tl.store(dq_ptrs, dq.to(dQ.dtype.element_ty), mask=mask_m[:, None])


@triton.jit
def bwd_dkdv_kernel(
    Q, K, V, O, dO, dK, dV, L,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    stride_lb, stride_lh, stride_ls,
    S, sm_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D_HEAD: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)
    
    start_n = pid_n * BLOCK_N
    off_n = start_n + tl.arange(0, BLOCK_N)
    mask_n = off_n < S
    
    q_offset = pid_b * stride_qb + pid_h * stride_qh
    k_offset = pid_b * stride_kb + pid_h * stride_kh
    v_offset = pid_b * stride_vb + pid_h * stride_vh
    o_offset = pid_b * stride_ob + pid_h * stride_oh
    do_offset = pid_b * stride_dob + pid_h * stride_doh
    dk_offset = pid_b * stride_dkb + pid_h * stride_dkh
    dv_offset = pid_b * stride_dvb + pid_h * stride_dvh
    l_offset = pid_b * stride_lb + pid_h * stride_lh
    
    k_ptrs = K + k_offset + off_n[:, None] * stride_ks + tl.arange(0, D_HEAD)[None, :] * stride_kd
    v_ptrs = V + v_offset + off_n[:, None] * stride_vs + tl.arange(0, D_HEAD)[None, :] * stride_vd
    
    k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
    
    dk = tl.zeros([BLOCK_N, D_HEAD], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, D_HEAD], dtype=tl.float32)
    
    # We bypass strict upper triangle evaluations using starting offsets
    start_m_initial = (start_n // BLOCK_M) * BLOCK_M
    
    off_m_base = tl.arange(0, BLOCK_M)
    q_ptrs = Q + q_offset + (start_m_initial + off_m_base)[:, None] * stride_qs + tl.arange(0, D_HEAD)[None, :] * stride_qd
    do_ptrs = dO + do_offset + (start_m_initial + off_m_base)[:, None] * stride_dos + tl.arange(0, D_HEAD)[None, :] * stride_dod
    out_ptrs = O + o_offset + (start_m_initial + off_m_base)[:, None] * stride_os + tl.arange(0, D_HEAD)[None, :] * stride_od
    l_ptrs = L + l_offset + (start_m_initial + off_m_base) * stride_ls
    
    for start_m in range(start_m_initial, S, BLOCK_M):
        off_m_curr = start_m + off_m_base
        mask_m = off_m_curr < S
        
        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
        do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
        out = tl.load(out_ptrs, mask=mask_m[:, None], other=0.0)
        l = tl.load(l_ptrs, mask=mask_m, other=0.0)
        
        d = tl.sum(out.to(tl.float32) * do.to(tl.float32), axis=1)
        
        qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * sm_scale
        qk_l = qk - l[:, None]
        
        is_causal_boundary = start_m < start_n + BLOCK_N
        if is_causal_boundary:
            qk_l = tl.where(off_m_curr[:, None] >= off_n[None, :], qk_l, float("-inf"))
            
        if start_m + BLOCK_M > S or start_n + BLOCK_N > S:
            qk_l = tl.where(mask_m[:, None] & mask_n[None, :], qk_l, float("-inf"))
            
        p = tl.exp(qk_l)
        
        dv += tl.dot(tl.trans(p.to(q.dtype)), do, out_dtype=tl.float32)
        dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32)
        
        ds = p * (dp - d[:, None]) * sm_scale
        dk += tl.dot(tl.trans(ds.to(q.dtype)), q, out_dtype=tl.float32)
        
        q_ptrs += BLOCK_M * stride_qs
        do_ptrs += BLOCK_M * stride_dos
        out_ptrs += BLOCK_M * stride_os
        l_ptrs += BLOCK_M * stride_ls
        
    dk_ptrs = dK + dk_offset + off_n[:, None] * stride_dks + tl.arange(0, D_HEAD)[None, :] * stride_dkd
    dv_ptrs = dV + dv_offset + off_n[:, None] * stride_dvs + tl.arange(0, D_HEAD)[None, :] * stride_dvd
    
    tl.store(dk_ptrs, dk.to(dK.dtype.element_ty), mask=mask_n[:, None])
    tl.store(dv_ptrs, dv.to(dV.dtype.element_ty), mask=mask_n[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes standard destination-passing causal multi-head attention backward.
    """
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    sm_scale = 1.0 / math.sqrt(d)

    # Assure shared memory loads fall comfortably below 228 KB limit via tuned block dimensions
    BLOCK_M_DQ = 128
    BLOCK_N_DQ = 64

    BLOCK_M_DK = 64
    BLOCK_N_DK = 128

    # 1. Launch dQ evaluation kernel
    grid_dq = (triton.cdiv(S, BLOCK_M_DQ), B, H)
    bwd_dq_kernel[grid_dq](
        Q, K, V, O, dO, dQ, L,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        S, sm_scale,
        BLOCK_M=BLOCK_M_DQ, BLOCK_N=BLOCK_N_DQ, D_HEAD=d,
        num_warps=4, num_stages=3
    )

    # 2. Launch dK & dV evaluation kernel
    grid_dkdv = (triton.cdiv(S, BLOCK_N_DK), B, H)
    bwd_dkdv_kernel[grid_dkdv](
        Q, K, V, O, dO, dK, dV, L,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        S, sm_scale,
        BLOCK_M=BLOCK_M_DK, BLOCK_N=BLOCK_N_DK, D_HEAD=d,
        num_warps=4, num_stages=3
    )