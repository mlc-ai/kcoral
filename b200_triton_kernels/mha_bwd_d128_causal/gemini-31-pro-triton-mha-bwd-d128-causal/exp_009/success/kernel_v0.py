import math
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=4, num_warps=8),
    ],
    key=['seq_len']
)
@triton.jit
def bwd_kernel_dq(
    Q, K, V, O, sm_scale, dO, dQ, L,
    seq_len,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    stride_lb, stride_lh, stride_ls,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D_HEAD: tl.constexpr
):
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_m = tl.program_id(0)

    start_i = pid_m * BLOCK_M
    
    dq = tl.zeros([BLOCK_M, D_HEAD], dtype=tl.float32)
    
    q_ptr = Q + pid_b * stride_qb + pid_h * stride_qh
    do_ptr = dO + pid_b * stride_dob + pid_h * stride_doh
    o_ptr = O + pid_b * stride_ob + pid_h * stride_oh
    l_ptr = L + pid_b * stride_lb + pid_h * stride_lh
    
    offs_m = start_i + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D_HEAD)
    
    q_ptrs = q_ptr + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    do_ptrs = do_ptr + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod
    o_ptrs = o_ptr + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    
    mask_m = offs_m < seq_len
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
    o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
    
    l_ptrs = l_ptr + offs_m * stride_ls
    l = tl.load(l_ptrs, mask=mask_m, other=0.0)
    
    # Precompute local row sum of (O * dO) for the current Q block
    d_val = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
    
    k_ptr = K + pid_b * stride_kb + pid_h * stride_kh
    v_ptr = V + pid_b * stride_vb + pid_h * stride_vh
    
    offs_n_base = tl.arange(0, BLOCK_N)
    
    # Causal masking logic for dQ: we only need to attend to K_j up to i
    max_j = tl.minimum(start_i + BLOCK_M, seq_len)
    num_steps = tl.cdiv(max_j, BLOCK_N)
    
    for step in range(num_steps):
        start_n = step * BLOCK_N
        offs_n = start_n + offs_n_base
        mask_n = offs_n < seq_len
        
        k_ptrs = k_ptr + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        v_ptrs = v_ptr + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
        
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        
        acc_qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, tl.trans(k), acc=acc_qk)
        qk = qk * sm_scale
        
        mask_causal = offs_m[:, None] >= offs_n[None, :]
        mask_valid = mask_m[:, None] & mask_n[None, :]
        mask_all = mask_causal & mask_valid
        
        qk = tl.where(mask_all, qk, float("-inf"))
        
        p = tl.exp(qk - l[:, None])
        p = tl.where(mask_all, p, 0.0)
        
        acc_dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        dp = tl.dot(do, tl.trans(v), acc=acc_dp)
        
        ds = p * (dp - d_val[:, None]) * sm_scale
        ds_cast = tl.cast(ds, q.dtype)
        
        dq = tl.dot(ds_cast, k, acc=dq)
        
    dq_ptr = dQ + pid_b * stride_dqb + pid_h * stride_dqh + offs_m[:, None] * stride_dqs + offs_d[None, :] * stride_dqd
    tl.store(dq_ptr, tl.cast(dq, dQ.dtype.element_ty), mask=mask_m[:, None])


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 64, 'BLOCK_M': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_N': 64, 'BLOCK_M': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_N': 64, 'BLOCK_M': 128}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_N': 64, 'BLOCK_M': 128}, num_stages=4, num_warps=8),
    ],
    key=['seq_len']
)
@triton.jit
def bwd_kernel_dk_dv(
    Q, K, V, O, sm_scale, dO, dK, dV, L,
    seq_len,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    stride_lb, stride_lh, stride_ls,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D_HEAD: tl.constexpr
):
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_n = tl.program_id(0)

    start_j = pid_n * BLOCK_N
    
    dk = tl.zeros([BLOCK_N, D_HEAD], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, D_HEAD], dtype=tl.float32)
    
    k_ptr = K + pid_b * stride_kb + pid_h * stride_kh
    v_ptr = V + pid_b * stride_vb + pid_h * stride_vh
    
    offs_n = start_j + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D_HEAD)
    
    k_ptrs = k_ptr + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = v_ptr + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
    
    mask_n = offs_n < seq_len
    k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
    
    q_ptr = Q + pid_b * stride_qb + pid_h * stride_qh
    do_ptr = dO + pid_b * stride_dob + pid_h * stride_doh
    o_ptr = O + pid_b * stride_ob + pid_h * stride_oh
    l_ptr = L + pid_b * stride_lb + pid_h * stride_lh
    
    # Causal masking logic for dKdV: we only need to attend from Q_i down to j
    start_i = (start_j // BLOCK_M) * BLOCK_M
    offs_m_base = tl.arange(0, BLOCK_M)
    
    num_steps = tl.cdiv(seq_len - start_i, BLOCK_M)
    
    for step in range(num_steps):
        start_m = start_i + step * BLOCK_M
        offs_m = start_m + offs_m_base
        mask_m = offs_m < seq_len
        
        q_ptrs = q_ptr + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
        do_ptrs = do_ptr + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod
        o_ptrs = o_ptr + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
        
        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
        do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
        o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
        
        l_ptrs = l_ptr + offs_m * stride_ls
        l = tl.load(l_ptrs, mask=mask_m, other=0.0)
        
        d_val = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
        
        acc_qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, tl.trans(k), acc=acc_qk)
        qk = qk * sm_scale
        
        mask_causal = offs_m[:, None] >= offs_n[None, :]
        mask_valid = mask_m[:, None] & mask_n[None, :]
        mask_all = mask_causal & mask_valid
        
        qk = tl.where(mask_all, qk, float("-inf"))
        
        p = tl.exp(qk - l[:, None])
        p = tl.where(mask_all, p, 0.0)
        
        acc_dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        dp = tl.dot(do, tl.trans(v), acc=acc_dp)
        
        ds = p * (dp - d_val[:, None]) * sm_scale
        
        ds_cast = tl.cast(ds, q.dtype)
        p_cast = tl.cast(p, q.dtype)
        
        dv = tl.dot(tl.trans(p_cast), do, acc=dv)
        dk = tl.dot(tl.trans(ds_cast), q, acc=dk)
        
    dk_ptr = dK + pid_b * stride_dkb + pid_h * stride_dkh + offs_n[:, None] * stride_dks + offs_d[None, :] * stride_dkd
    dv_ptr = dV + pid_b * stride_dvb + pid_h * stride_dvh + offs_n[:, None] * stride_dvs + offs_d[None, :] * stride_dvd
    
    tl.store(dk_ptr, tl.cast(dk, dK.dtype.element_ty), mask=mask_n[:, None])
    tl.store(dv_ptr, tl.cast(dv, dV.dtype.element_ty), mask=mask_n[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes causal Multi-Head Attention backward pass using standard Triton semantics.
    Results are written inplace into preallocated destination tensors: `dQ`, `dK`, `dV`.
    """
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    sm_scale = 1.0 / math.sqrt(d)
    
    grid_dq = lambda META: (
        triton.cdiv(S, META['BLOCK_M']),
        B,
        H
    )
    
    bwd_kernel_dq[grid_dq](
        Q, K, V, O, sm_scale, dO, dQ, L,
        S,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        D_HEAD=d
    )

    grid_dkdv = lambda META: (
        triton.cdiv(S, META['BLOCK_N']),
        B,
        H
    )
    
    bwd_kernel_dk_dv[grid_dkdv](
        Q, K, V, O, sm_scale, dO, dK, dV, L,
        S,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        D_HEAD=d
    )