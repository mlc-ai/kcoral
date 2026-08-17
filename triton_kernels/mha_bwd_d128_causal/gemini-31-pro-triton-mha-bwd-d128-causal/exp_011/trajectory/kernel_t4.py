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
    
    # Int64 casting to strictly prevent out-of-bounds 32-bit overflows on view striding
    off_b = pid_b.to(tl.int64)
    off_h = pid_h.to(tl.int64)
    
    q_offset = off_b * stride_qb + off_h * stride_qh
    k_offset = off_b * stride_kb + off_h * stride_kh
    v_offset = off_b * stride_vb + off_h * stride_vh
    o_offset = off_b * stride_ob + off_h * stride_oh
    do_offset = off_b * stride_dob + off_h * stride_doh
    dq_offset = off_b * stride_dqb + off_h * stride_dqh
    l_offset = off_b * stride_lb + off_h * stride_lh
    
    start_m = pid_m * BLOCK_M
    off_m = start_m + tl.arange(0, BLOCK_M)
    mask_m = off_m < S
    
    # Safely clamp sequence coordinates to strictly prevent out-of-bounds pointer construction
    off_m_safe = tl.where(mask_m, off_m, S - 1).to(tl.int64)
    head_dim = tl.arange(0, D_HEAD)
    
    q_ptrs = Q + q_offset + off_m_safe[:, None] * stride_qs + head_dim[None, :] * stride_qd
    do_ptrs = dO + do_offset + off_m_safe[:, None] * stride_dos + head_dim[None, :] * stride_dod
    out_ptrs = O + o_offset + off_m_safe[:, None] * stride_os + head_dim[None, :] * stride_od
    l_ptrs = L + l_offset + off_m_safe * stride_ls
    
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
    out = tl.load(out_ptrs, mask=mask_m[:, None], other=0.0)
    l = tl.load(l_ptrs, mask=mask_m, other=0.0)
    
    # D_i = sum(O_i * dO_i). Locally resolved without D cache mapping overhead.
    d = tl.sum(out.to(tl.float32) * do.to(tl.float32), axis=1)
    del out  # free registers dynamically
    
    dq = tl.zeros([BLOCK_M, D_HEAD], dtype=tl.float32)
    
    n_blocks = (start_m + BLOCK_M + BLOCK_N - 1) // BLOCK_N
    max_start_n = (S + BLOCK_N - 1) // BLOCK_N
    if n_blocks > max_start_n:
        n_blocks = max_start_n
        
    for start_n in range(0, n_blocks * BLOCK_N, BLOCK_N):
        off_n_curr = start_n + tl.arange(0, BLOCK_N)
        mask_n = off_n_curr < S
        off_n_safe = tl.where(mask_n, off_n_curr, S - 1).to(tl.int64)
        
        k_ptrs = K + k_offset + off_n_safe[:, None] * stride_ks + head_dim[None, :] * stride_kd
        v_ptrs = V + v_offset + off_n_safe[:, None] * stride_vs + head_dim[None, :] * stride_vd
        
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
        
    dq_ptrs = dQ + dq_offset + off_m_safe[:, None] * stride_dqs + head_dim[None, :] * stride_dqd
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
    
    off_b = pid_b.to(tl.int64)
    off_h = pid_h.to(tl.int64)
    
    q_offset = off_b * stride_qb + off_h * stride_qh
    k_offset = off_b * stride_kb + off_h * stride_kh
    v_offset = off_b * stride_vb + off_h * stride_vh
    o_offset = off_b * stride_ob + off_h * stride_oh
    do_offset = off_b * stride_dob + off_h * stride_doh
    dk_offset = off_b * stride_dkb + off_h * stride_dkh
    dv_offset = off_b * stride_dvb + off_h * stride_dvh
    l_offset = off_b * stride_lb + off_h * stride_lh
    
    start_n = pid_n * BLOCK_N
    off_n = start_n + tl.arange(0, BLOCK_N)
    mask_n = off_n < S
    
    # Safely clamp sequence coordinates dynamically avoiding math bounds overflow exceptions
    off_n_safe = tl.where(mask_n, off_n, S - 1).to(tl.int64)
    head_dim = tl.arange(0, D_HEAD)
    
    k_ptrs = K + k_offset + off_n_safe[:, None] * stride_ks + head_dim[None, :] * stride_kd
    v_ptrs = V + v_offset + off_n_safe[:, None] * stride_vs + head_dim[None, :] * stride_vd
    
    k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
    
    dk = tl.zeros([BLOCK_N, D_HEAD], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, D_HEAD], dtype=tl.float32)
    
    # Initializing from the upper loop offset bounds effectively skips triangle zeroes
    start_m_initial = (start_n // BLOCK_M) * BLOCK_M
    
    for start_m in range(start_m_initial, S, BLOCK_M):
        off_m_curr = start_m + tl.arange(0, BLOCK_M)
        mask_m = off_m_curr < S
        off_m_safe = tl.where(mask_m, off_m_curr, S - 1).to(tl.int64)
        
        q_ptrs = Q + q_offset + off_m_safe[:, None] * stride_qs + head_dim[None, :] * stride_qd
        do_ptrs = dO + do_offset + off_m_safe[:, None] * stride_dos + head_dim[None, :] * stride_dod
        out_ptrs = O + o_offset + off_m_safe[:, None] * stride_os + head_dim[None, :] * stride_od
        l_ptrs = L + l_offset + off_m_safe * stride_ls
        
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
        
    dk_ptrs = dK + dk_offset + off_n_safe[:, None] * stride_dks + head_dim[None, :] * stride_dkd
    dv_ptrs = dV + dv_offset + off_n_safe[:, None] * stride_dvs + head_dim[None, :] * stride_dvd
    
    tl.store(dk_ptrs, dk.to(dK.dtype.element_ty), mask=mask_n[:, None])
    tl.store(dv_ptrs, dv.to(dV.dtype.element_ty), mask=mask_n[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes standard destination-passing causal multi-head attention backward.
    """
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    sm_scale = 1.0 / math.sqrt(d)

    # WGMMA tuned 2x layout balances register pressure cleanly to 255 slots over 8 warps
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
        num_warps=8, num_stages=3
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
        num_warps=8, num_stages=3
    )