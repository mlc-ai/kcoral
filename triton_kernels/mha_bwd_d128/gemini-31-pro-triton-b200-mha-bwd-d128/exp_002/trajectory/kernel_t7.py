import math
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
    ],
    key=['S']
)
@triton.jit
def bwd_dq_kernel(
    Q, K, V, O, dO, L, dQ,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    S, alpha,
    H: tl.constexpr, d: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    b = pid_bh // H
    h = pid_bh % H
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, d)
    
    mask_m = offs_m < S
    
    Q += b * stride_qb + h * stride_qh
    K += b * stride_kb + h * stride_kh
    V += b * stride_vb + h * stride_vh
    O += b * stride_ob + h * stride_oh
    dO += b * stride_dob + h * stride_doh
    L += b * stride_lb + h * stride_lh
    dQ += b * stride_dqb + h * stride_dqh
    
    q_ptrs = Q + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    o_ptrs = O + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    do_ptrs = dO + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod
    l_ptrs = L + offs_m * stride_ls
    
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
    do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
    l = tl.load(l_ptrs, mask=mask_m, other=0.0)
    
    q_scaled = (q.to(tl.float32) * alpha).to(tl.bfloat16)
    d_val = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
    
    dq = tl.zeros([BLOCK_M, d], dtype=tl.float32)
    
    k_ptrs = K + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = V + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
    
    for start_n in range(0, S, BLOCK_N):
        curr_n = start_n + offs_n
        mask_n = curr_n < S
        
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        
        s_attn = tl.dot(q_scaled, k.T, out_dtype=tl.float32)
        p = tl.exp(s_attn - l[:, None])
        p = tl.where(mask_m[:, None] & mask_n[None, :], p, 0.0)
        
        dp = tl.dot(do, v.T, out_dtype=tl.float32)
        da = p * (dp - d_val[:, None])
        
        dq = tl.dot(da.to(tl.bfloat16), k, acc=dq)
        
        k_ptrs += BLOCK_N * stride_ks
        v_ptrs += BLOCK_N * stride_vs
        
    dq = dq * alpha
    
    dq_ptrs = dQ + offs_m[:, None] * stride_dqs + offs_d[None, :] * stride_dqd
    tl.store(dq_ptrs, dq.to(tl.bfloat16), mask=mask_m[:, None])


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 128, 'BLOCK_M': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_M': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 64, 'BLOCK_M': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 64, 'BLOCK_M': 64}, num_warps=4, num_stages=4),
    ],
    key=['S']
)
@triton.jit
def bwd_dkv_kernel(
    Q, K, V, O, dO, L, dK, dV,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    S, alpha, 
    H: tl.constexpr, d: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    b = pid_bh // H
    h = pid_bh % H
    
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, d)
    
    mask_n = offs_n < S
    
    Q += b * stride_qb + h * stride_qh
    K += b * stride_kb + h * stride_kh
    V += b * stride_vb + h * stride_vh
    O += b * stride_ob + h * stride_oh
    dO += b * stride_dob + h * stride_doh
    L += b * stride_lb + h * stride_lh
    dK += b * stride_dkb + h * stride_dkh
    dV += b * stride_dvb + h * stride_dvh
    
    k_ptrs = K + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = V + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
    
    k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
    
    k_scaled = (k.to(tl.float32) * alpha).to(tl.bfloat16)
    
    dk = tl.zeros([BLOCK_N, d], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, d], dtype=tl.float32)
    
    q_ptrs = Q + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    o_ptrs = O + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    do_ptrs = dO + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod
    l_ptrs = L + offs_m * stride_ls
    
    for start_m in range(0, S, BLOCK_M):
        curr_m = start_m + offs_m
        mask_m = curr_m < S
        
        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
        o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
        do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
        l = tl.load(l_ptrs, mask=mask_m, other=0.0)
        
        d_val = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
        
        # Reformulated K @ Q^T yields [BLOCK_N, BLOCK_M] without intermediate shared memory transpose overheads
        s_attn_t = tl.dot(k_scaled, q.T, out_dtype=tl.float32)
        p_t = tl.exp(s_attn_t - l[None, :])
        p_t = tl.where(mask_n[:, None] & mask_m[None, :], p_t, 0.0)
        
        dp_t = tl.dot(v, do.T, out_dtype=tl.float32)
        da_t = p_t * (dp_t - d_val[None, :])
        
        dv = tl.dot(p_t.to(tl.bfloat16), do, acc=dv)
        dk = tl.dot(da_t.to(tl.bfloat16), q, acc=dk)
        
        q_ptrs += BLOCK_M * stride_qs
        o_ptrs += BLOCK_M * stride_os
        do_ptrs += BLOCK_M * stride_dos
        l_ptrs += BLOCK_M * stride_ls
        
    dk = dk * alpha
    
    dk_ptrs = dK + offs_n[:, None] * stride_dks + offs_d[None, :] * stride_dkd
    dv_ptrs = dV + offs_n[:, None] * stride_dvs + offs_d[None, :] * stride_dvd
    
    tl.store(dk_ptrs, dk.to(tl.bfloat16), mask=mask_n[:, None])
    tl.store(dv_ptrs, dv.to(tl.bfloat16), mask=mask_n[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes FlashAttention backward pass for sequence blocks without causal mask natively.
    Robust against arbitrary physical striding without relying on 4D descriptor TMA paths.
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, d = Q.shape
    alpha = 1.0 / math.sqrt(d)
    
    grid_dq = lambda META: (triton.cdiv(S, META['BLOCK_M']), B * H)
    bwd_dq_kernel[grid_dq](
        Q, K, V, O, dO, L, dQ,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        S, alpha,
        H=H, d=d
    )
    
    grid_dkv = lambda META: (triton.cdiv(S, META['BLOCK_N']), B * H)
    bwd_dkv_kernel[grid_dkv](
        Q, K, V, O, dO, L, dK, dV,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        S, alpha,
        H=H, d=d
    )
    
    return dQ, dK, dV