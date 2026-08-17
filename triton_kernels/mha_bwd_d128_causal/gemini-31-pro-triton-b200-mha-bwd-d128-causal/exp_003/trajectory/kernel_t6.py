import math
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
    ],
    key=['S']
)
@triton.jit
def bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, dQ_ptr, L_ptr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    stride_lb, stride_lh, stride_ls,
    S, H, inv_sqrt_d,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, d: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    start_m = pid_m * BLOCK_M
    if start_m >= S:
        return
        
    pid_b = pid_bh // H
    pid_h = pid_bh % H
    
    m_offs = start_m + tl.arange(0, BLOCK_M)
    # Hardware bounds fault safety: strictly clamp pointers independent of the execution mask
    safe_m_offs = tl.minimum(m_offs, S - 1)
    d_offs = tl.arange(0, d)
    
    q_ptrs = Q_ptr + pid_b * stride_qb + pid_h * stride_qh + safe_m_offs[:, None] * stride_qs + d_offs[None, :] * stride_qd
    do_ptrs = dO_ptr + pid_b * stride_dob + pid_h * stride_doh + safe_m_offs[:, None] * stride_dos + d_offs[None, :] * stride_dod
    o_ptrs = O_ptr + pid_b * stride_ob + pid_h * stride_oh + safe_m_offs[:, None] * stride_os + d_offs[None, :] * stride_od
    
    mask_m = m_offs < S
    mask_m_2d = mask_m[:, None]
    
    q = tl.load(q_ptrs, mask=mask_m_2d, other=0.0)
    do = tl.load(do_ptrs, mask=mask_m_2d, other=0.0)
    o = tl.load(o_ptrs, mask=mask_m_2d, other=0.0)
    
    do_o = do.to(tl.float32) * o.to(tl.float32)
    D_M = tl.sum(do_o, axis=1)
    
    l_ptrs = L_ptr + pid_b * stride_lb + pid_h * stride_lh + safe_m_offs * stride_ls
    l = tl.load(l_ptrs, mask=mask_m, other=float('inf'))
    
    dq = tl.zeros([BLOCK_M, d], dtype=tl.float32)
    
    N_unmasked = (start_m // BLOCK_N) * BLOCK_N
    
    k_ptrs = K_ptr + pid_b * stride_kb + pid_h * stride_kh + d_offs[None, :] * stride_kd
    v_ptrs = V_ptr + pid_b * stride_vb + pid_h * stride_vh + d_offs[None, :] * stride_vd
    
    # Primary Loop (No causal masking logic required)
    if N_unmasked > 0:
        for start_n in range(0, N_unmasked, BLOCK_N):
            n_offs = start_n + tl.arange(0, BLOCK_N)
            
            curr_k_ptrs = k_ptrs + n_offs[:, None] * stride_ks
            curr_v_ptrs = v_ptrs + n_offs[:, None] * stride_vs
            
            k = tl.load(curr_k_ptrs)
            v = tl.load(curr_v_ptrs)
            
            p = tl.dot(q, k.T, out_dtype=tl.float32) * inv_sqrt_d
            s = tl.exp(p - l[:, None])
            
            dp_unscaled = tl.dot(do, v.T, out_dtype=tl.float32)
            ds = (dp_unscaled - D_M[:, None]) * s * inv_sqrt_d
            
            dq += tl.dot(ds.to(q.dtype), k, out_dtype=tl.float32)
            
    # Trailing Loop (Precise overlapping causal diagonal masking bounds)
    start_n_masked = N_unmasked
    while start_n_masked <= start_m and start_n_masked < S:
        n_offs = start_n_masked + tl.arange(0, BLOCK_N)
        safe_n_offs = tl.minimum(n_offs, S - 1)
        mask_n_2d = (n_offs < S)[:, None]
        
        curr_k_ptrs = k_ptrs + safe_n_offs[:, None] * stride_ks
        curr_v_ptrs = v_ptrs + safe_n_offs[:, None] * stride_vs
        
        k = tl.load(curr_k_ptrs, mask=mask_n_2d, other=0.0)
        v = tl.load(curr_v_ptrs, mask=mask_n_2d, other=0.0)
        
        p = tl.dot(q, k.T, out_dtype=tl.float32) * inv_sqrt_d
        
        mask = m_offs[:, None] >= n_offs[None, :]
        p = tl.where(mask, p, -float('inf'))
        
        s = tl.exp(p - l[:, None])
        
        dp_unscaled = tl.dot(do, v.T, out_dtype=tl.float32)
        ds = (dp_unscaled - D_M[:, None]) * s * inv_sqrt_d
        
        dq += tl.dot(ds.to(q.dtype), k, out_dtype=tl.float32)
        start_n_masked += BLOCK_N
        
    dq_ptrs = dQ_ptr + pid_b * stride_dqb + pid_h * stride_dqh + safe_m_offs[:, None] * stride_dqs + d_offs[None, :] * stride_dqd
    tl.store(dq_ptrs, dq.to(q.dtype), mask=mask_m_2d)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
    ],
    key=['S']
)
@triton.jit
def bwd_dk_dv_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, dK_ptr, dV_ptr, L_ptr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    stride_lb, stride_lh, stride_ls,
    S, H, inv_sqrt_d,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, d: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    start_n = pid_n * BLOCK_N
    if start_n >= S:
        return
        
    pid_b = pid_bh // H
    pid_h = pid_bh % H
    
    n_offs = start_n + tl.arange(0, BLOCK_N)
    safe_n_offs = tl.minimum(n_offs, S - 1)
    d_offs = tl.arange(0, d)
    mask_n = n_offs < S
    mask_n_2d = mask_n[:, None]
    
    k_ptrs = K_ptr + pid_b * stride_kb + pid_h * stride_kh + safe_n_offs[:, None] * stride_ks + d_offs[None, :] * stride_kd
    v_ptrs = V_ptr + pid_b * stride_vb + pid_h * stride_vh + safe_n_offs[:, None] * stride_vs + d_offs[None, :] * stride_vd
    
    k = tl.load(k_ptrs, mask=mask_n_2d, other=0.0)
    v = tl.load(v_ptrs, mask=mask_n_2d, other=0.0)
    
    dk = tl.zeros([BLOCK_N, d], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, d], dtype=tl.float32)
    
    start_m_aligned = (start_n // BLOCK_M) * BLOCK_M
    S_aligned = ((S + BLOCK_M - 1) // BLOCK_M) * BLOCK_M
    
    M_unmasked = ((start_n + BLOCK_N + BLOCK_M - 1) // BLOCK_M) * BLOCK_M
    if M_unmasked > S_aligned:
        M_unmasked = S_aligned
        
    q_ptrs = Q_ptr + pid_b * stride_qb + pid_h * stride_qh + d_offs[None, :] * stride_qd
    do_ptrs = dO_ptr + pid_b * stride_dob + pid_h * stride_doh + d_offs[None, :] * stride_dod
    o_ptrs = O_ptr + pid_b * stride_ob + pid_h * stride_oh + d_offs[None, :] * stride_od
    
    start_m_masked = start_m_aligned
    while start_m_masked < M_unmasked and start_m_masked < S:
        m_offs = start_m_masked + tl.arange(0, BLOCK_M)
        safe_m_offs = tl.minimum(m_offs, S - 1)
        mask_m = m_offs < S
        mask_m_2d = mask_m[:, None]
        
        curr_q_ptrs = q_ptrs + safe_m_offs[:, None] * stride_qs
        curr_do_ptrs = do_ptrs + safe_m_offs[:, None] * stride_dos
        curr_o_ptrs = o_ptrs + safe_m_offs[:, None] * stride_os
        
        q = tl.load(curr_q_ptrs, mask=mask_m_2d, other=0.0)
        do = tl.load(curr_do_ptrs, mask=mask_m_2d, other=0.0)
        o = tl.load(curr_o_ptrs, mask=mask_m_2d, other=0.0)
        
        do_o = do.to(tl.float32) * o.to(tl.float32)
        D_M_masked = tl.sum(do_o, axis=1)
        
        l_ptrs = L_ptr + pid_b * stride_lb + pid_h * stride_lh + safe_m_offs * stride_ls
        l = tl.load(l_ptrs, mask=mask_m, other=float('inf'))
        
        p = tl.dot(q, k.T, out_dtype=tl.float32) * inv_sqrt_d
        
        mask = m_offs[:, None] >= n_offs[None, :]
        p = tl.where(mask, p, -float('inf'))
        
        s = tl.exp(p - l[:, None])
        
        dv += tl.dot(s.T.to(do.dtype), do, out_dtype=tl.float32)
        
        dp_unscaled = tl.dot(do, v.T, out_dtype=tl.float32)
        ds = (dp_unscaled - D_M_masked[:, None]) * s * inv_sqrt_d
        
        dk += tl.dot(ds.T.to(q.dtype), q, out_dtype=tl.float32)
        start_m_masked += BLOCK_M

    if M_unmasked < S_aligned:
        for start_m in range(M_unmasked, S_aligned, BLOCK_M):
            m_offs = start_m + tl.arange(0, BLOCK_M)
            safe_m_offs = tl.minimum(m_offs, S - 1)
            mask_m = m_offs < S
            mask_m_2d = mask_m[:, None]
            
            curr_q_ptrs = q_ptrs + safe_m_offs[:, None] * stride_qs
            curr_do_ptrs = do_ptrs + safe_m_offs[:, None] * stride_dos
            curr_o_ptrs = o_ptrs + safe_m_offs[:, None] * stride_os
            
            q = tl.load(curr_q_ptrs, mask=mask_m_2d, other=0.0)
            do = tl.load(curr_do_ptrs, mask=mask_m_2d, other=0.0)
            o = tl.load(curr_o_ptrs, mask=mask_m_2d, other=0.0)
            
            do_o = do.to(tl.float32) * o.to(tl.float32)
            D_M_unmasked = tl.sum(do_o, axis=1)
            
            l_ptrs = L_ptr + pid_b * stride_lb + pid_h * stride_lh + safe_m_offs * stride_ls
            l = tl.load(l_ptrs, mask=mask_m, other=float('inf'))
            
            p = tl.dot(q, k.T, out_dtype=tl.float32) * inv_sqrt_d
            s = tl.exp(p - l[:, None])
            
            dv += tl.dot(s.T.to(do.dtype), do, out_dtype=tl.float32)
            
            dp_unscaled = tl.dot(do, v.T, out_dtype=tl.float32)
            ds = (dp_unscaled - D_M_unmasked[:, None]) * s * inv_sqrt_d
            
            dk += tl.dot(ds.T.to(q.dtype), q, out_dtype=tl.float32)
            
    dk_ptrs = dK_ptr + pid_b * stride_dkb + pid_h * stride_dkh + safe_n_offs[:, None] * stride_dks + d_offs[None, :] * stride_dkd
    dv_ptrs = dV_ptr + pid_b * stride_dvb + pid_h * stride_dvh + safe_n_offs[:, None] * stride_dvs + d_offs[None, :] * stride_dvd
    
    tl.store(dk_ptrs, dk.to(k.dtype), mask=mask_n_2d)
    tl.store(dv_ptrs, dv.to(v.dtype), mask=mask_n_2d)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Perform causal multi-head attention backward correctly in destination-passing style.
    Calculates inline gradient operations completely correctly independent of tensor contiguous strides.
    """
    with torch.cuda.device(Q.device):
        B, H, S, d = Q.shape
        inv_sqrt_d = 1.0 / math.sqrt(d)
        
        # Dispatch 1: Compute Query Gradients independently
        grid_dq = lambda META: (triton.cdiv(S, META['BLOCK_M']), B * H)
        bwd_dq_kernel[grid_dq](
            Q, K, V, O, dO, dQ, L,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
            dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
            L.stride(0), L.stride(1), L.stride(2) if L.dim() >= 3 else 1,
            S, H, inv_sqrt_d,
            d=d
        )
        
        # Dispatch 2: Compute Key and Value Gradients independently
        grid_dkv = lambda META: (triton.cdiv(S, META['BLOCK_N']), B * H)
        bwd_dk_dv_kernel[grid_dkv](
            Q, K, V, O, dO, dK, dV, L,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
            dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
            dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
            L.stride(0), L.stride(1), L.stride(2) if L.dim() >= 3 else 1,
            S, H, inv_sqrt_d,
            d=d
        )