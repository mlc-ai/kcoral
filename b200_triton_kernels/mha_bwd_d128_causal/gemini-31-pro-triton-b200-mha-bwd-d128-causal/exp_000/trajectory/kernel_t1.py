import torch
import triton
import triton.language as tl

@triton.jit
def _bwd_kernel_dq(
    Q, K, V, O, sm_scale,
    DO, DQ, L,
    S, H,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    stride_lb, stride_lh, stride_ls,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D_HEAD: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < S
    # Clamp the memory offsets to guarantee in-bounds pointer arithmetic during async prefetching
    offs_m_safe = tl.where(mask_m, offs_m, S - 1)
    
    offs_d = tl.arange(0, D_HEAD)
    
    q_ptrs = Q + b * stride_qb + h * stride_qh + offs_m_safe[:, None] * stride_qs + offs_d[None, :] * stride_qd
    do_ptrs = DO + b * stride_dob + h * stride_doh + offs_m_safe[:, None] * stride_dos + offs_d[None, :] * stride_dod
    o_ptrs = O + b * stride_ob + h * stride_oh + offs_m_safe[:, None] * stride_os + offs_d[None, :] * stride_od
    
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
    o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
    
    l_ptrs = L + b * stride_lb + h * stride_lh + offs_m_safe * stride_ls
    lse = tl.load(l_ptrs, mask=mask_m, other=0.0)
    
    # Precompute Delta securely directly inside the kernel (avoids redundant workspace HBM write/reads)
    delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
    
    dq = tl.zeros([BLOCK_M, D_HEAD], dtype=tl.float32)
    
    n_max = tl.minimum(S, (pid_m + 1) * BLOCK_M)
    n_steps = tl.cdiv(n_max, BLOCK_N)
    
    for n_idx in range(0, n_steps):
        offs_n = n_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S
        offs_n_safe = tl.where(mask_n, offs_n, S - 1)
        
        k_ptrs = K + b * stride_kb + h * stride_kh + offs_n_safe[:, None] * stride_ks + offs_d[None, :] * stride_kd
        v_ptrs = V + b * stride_vb + h * stride_vh + offs_n_safe[:, None] * stride_vs + offs_d[None, :] * stride_vd
        
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        
        qk = tl.dot(q, tl.trans(k)) * sm_scale
        p = tl.exp(qk - lse[:, None])
        
        # Use strictly logically correct (un-clamped) indices for masking the operation values
        mask_mn = (offs_m[:, None] < S) & (offs_n[None, :] < S)
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        p = tl.where(mask_mn & causal_mask, p, 0.0)
        
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])
        ds_scaled = ds * sm_scale
        
        dq += tl.dot(ds_scaled.to(Q.dtype.element_ty), k)
        
    dq_ptrs = DQ + b * stride_dqb + h * stride_dqh + offs_m_safe[:, None] * stride_dqs + offs_d[None, :] * stride_dqd
    tl.store(dq_ptrs, dq.to(DQ.dtype.element_ty), mask=mask_m[:, None])

@triton.jit
def _bwd_kernel_dk_dv(
    Q, K, V, O, sm_scale,
    DO, DK, DV, L,
    S, H,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    stride_lb, stride_lh, stride_ls,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D_HEAD: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < S
    offs_n_safe = tl.where(mask_n, offs_n, S - 1)
    
    offs_d = tl.arange(0, D_HEAD)
    
    k_ptrs = K + b * stride_kb + h * stride_kh + offs_n_safe[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = V + b * stride_vb + h * stride_vh + offs_n_safe[:, None] * stride_vs + offs_d[None, :] * stride_vd
    
    k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
    
    dk = tl.zeros([BLOCK_N, D_HEAD], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, D_HEAD], dtype=tl.float32)
    
    m_start = (pid_n * BLOCK_N) // BLOCK_M
    m_steps = tl.cdiv(S, BLOCK_M)
    
    for m_idx in range(m_start, m_steps):
        offs_m = m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S
        offs_m_safe = tl.where(mask_m, offs_m, S - 1)
        
        q_ptrs = Q + b * stride_qb + h * stride_qh + offs_m_safe[:, None] * stride_qs + offs_d[None, :] * stride_qd
        do_ptrs = DO + b * stride_dob + h * stride_doh + offs_m_safe[:, None] * stride_dos + offs_d[None, :] * stride_dod
        o_ptrs = O + b * stride_ob + h * stride_oh + offs_m_safe[:, None] * stride_os + offs_d[None, :] * stride_od
        
        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
        do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
        o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
        
        l_ptrs = L + b * stride_lb + h * stride_lh + offs_m_safe * stride_ls
        lse = tl.load(l_ptrs, mask=mask_m, other=0.0)
        
        delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
        
        qk = tl.dot(q, tl.trans(k)) * sm_scale
        p = tl.exp(qk - lse[:, None])
        
        mask_mn = (offs_m[:, None] < S) & (offs_n[None, :] < S)
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        p = tl.where(mask_mn & causal_mask, p, 0.0)
        
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])
        ds_scaled = ds * sm_scale
        
        dv += tl.dot(tl.trans(p).to(Q.dtype.element_ty), do)
        dk += tl.dot(tl.trans(ds_scaled).to(Q.dtype.element_ty), q)
        
    dk_ptrs = DK + b * stride_dkb + h * stride_dkh + offs_n_safe[:, None] * stride_dks + offs_d[None, :] * stride_dkd
    dv_ptrs = DV + b * stride_dvb + h * stride_dvh + offs_n_safe[:, None] * stride_dvs + offs_d[None, :] * stride_dvd
    
    tl.store(dk_ptrs, dk.to(DK.dtype.element_ty), mask=mask_n[:, None])
    tl.store(dv_ptrs, dv.to(DV.dtype.element_ty), mask=mask_n[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Evaluates causal multi-head attention backward gradients.
    Destination passing cleanly intercepts computational targets for outputs explicitly.
    """
    with torch.cuda.device(Q.device):
        B, H, S, d = Q.shape
        sm_scale = 1.0 / (d ** 0.5)

        stride_lb = L.stride(0)
        stride_lh = L.stride(1)
        stride_ls = L.stride(2) if L.dim() >= 3 else 1

        BLOCK_M = 128
        BLOCK_N = 64
        
        # dQ kernel execution configuration
        grid_dq = (triton.cdiv(S, BLOCK_M), B * H)
        _bwd_kernel_dq[grid_dq](
            Q, K, V, O, sm_scale,
            dO, dQ, L,
            S, H,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
            dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
            stride_lb, stride_lh, stride_ls,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, D_HEAD=d,
            num_warps=8, num_stages=3
        )

        # dK and dV kernels execution configuration 
        grid_dkdv = (triton.cdiv(S, BLOCK_N), B * H)
        _bwd_kernel_dk_dv[grid_dkdv](
            Q, K, V, O, sm_scale,
            dO, dK, dV, L,
            S, H,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
            dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
            dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
            stride_lb, stride_lh, stride_ls,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, D_HEAD=d,
            num_warps=8, num_stages=3
        )