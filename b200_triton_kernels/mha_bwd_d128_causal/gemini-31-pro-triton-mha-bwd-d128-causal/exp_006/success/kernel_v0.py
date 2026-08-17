import math

import torch
import triton
import triton.language as tl


@triton.jit
def bwd_kernel_dq(
    Q, K, V, O, DO, dQ, L,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    stride_lb, stride_lh, stride_ls,
    scale, S,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D_HEAD: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D_HEAD)
    
    # Offsets for Q, O, DO, L
    q_ptrs = Q + pid_b * stride_qb + pid_h * stride_qh + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    o_ptrs = O + pid_b * stride_ob + pid_h * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    do_ptrs = DO + pid_b * stride_dob + pid_h * stride_doh + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod
    l_ptrs = L + pid_b * stride_lb + pid_h * stride_lh + offs_m * stride_ls
    
    mask_m = offs_m < S
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
    do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
    l = tl.load(l_ptrs, mask=mask_m, other=0.0)
    
    # Recompute D directly from O and dO
    d = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
    
    dq = tl.zeros([BLOCK_M, D_HEAD], dtype=tl.float32)
    
    # Causal sequence bound for the current Q block
    start_n = 0
    end_n = tl.minimum((pid_m + 1) * BLOCK_M, S)
    
    k_ptrs = K + pid_b * stride_kb + pid_h * stride_kh + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = V + pid_b * stride_vb + pid_h * stride_vh + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
    
    for start_n_val in range(start_n, end_n, BLOCK_N):
        curr_n = start_n_val + offs_n
        mask_n = curr_n < S
        
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        
        s = tl.dot(q, tl.trans(k)) * scale
        
        causal_mask = offs_m[:, None] >= curr_n[None, :]
        valid_mask = causal_mask & mask_m[:, None] & mask_n[None, :]
        s = tl.where(valid_mask, s, float("-inf"))
        
        p = tl.exp(s - l[:, None])
        p = tl.where(valid_mask, p, 0.0)
        
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - d[:, None]) * scale
        
        dq += tl.dot(ds.to(q.dtype), k)
        
        k_ptrs += BLOCK_N * stride_ks
        v_ptrs += BLOCK_N * stride_vs
        
    dq_ptrs = dQ + pid_b * stride_dqb + pid_h * stride_dqh + offs_m[:, None] * stride_dqs + offs_d[None, :] * stride_dqd
    tl.store(dq_ptrs, dq.to(q.dtype), mask=mask_m[:, None])


@triton.jit
def bwd_kernel_dk_dv(
    Q, K, V, O, DO, dK, dV, L,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    stride_lb, stride_lh, stride_ls,
    scale, S,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D_HEAD: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)
    
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D_HEAD)
    
    k_ptrs = K + pid_b * stride_kb + pid_h * stride_kh + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = V + pid_b * stride_vb + pid_h * stride_vh + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
    
    mask_n = offs_n < S
    k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
    
    dk = tl.zeros([BLOCK_N, D_HEAD], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, D_HEAD], dtype=tl.float32)
    
    # For causal mask, start_m maps exactly to the block containing pid_n * BLOCK_N
    start_m = (pid_n * BLOCK_N // BLOCK_M) * BLOCK_M
    
    q_ptrs = Q + pid_b * stride_qb + pid_h * stride_qh + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    o_ptrs = O + pid_b * stride_ob + pid_h * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    do_ptrs = DO + pid_b * stride_dob + pid_h * stride_doh + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod
    l_ptrs = L + pid_b * stride_lb + pid_h * stride_lh + offs_m * stride_ls
    
    # Advance pointers to causal beginning
    q_ptrs += start_m * stride_qs
    o_ptrs += start_m * stride_os
    do_ptrs += start_m * stride_dos
    l_ptrs += start_m * stride_ls
    
    for start_m_val in range(start_m, S, BLOCK_M):
        curr_m = start_m_val + offs_m
        mask_m = curr_m < S
        
        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
        o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
        do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
        l = tl.load(l_ptrs, mask=mask_m, other=0.0)
        
        d = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
        
        s = tl.dot(q, tl.trans(k)) * scale
        
        causal_mask = curr_m[:, None] >= offs_n[None, :]
        valid_mask = causal_mask & mask_m[:, None] & mask_n[None, :]
        s = tl.where(valid_mask, s, float("-inf"))
        
        p = tl.exp(s - l[:, None])
        p = tl.where(valid_mask, p, 0.0)
        
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - d[:, None]) * scale
        
        dk += tl.dot(tl.trans(ds).to(k.dtype), q)
        dv += tl.dot(tl.trans(p).to(v.dtype), do)
        
        q_ptrs += BLOCK_M * stride_qs
        o_ptrs += BLOCK_M * stride_os
        do_ptrs += BLOCK_M * stride_dos
        l_ptrs += BLOCK_M * stride_ls
        
    dk_ptrs = dK + pid_b * stride_dkb + pid_h * stride_dkh + offs_n[:, None] * stride_dks + offs_d[None, :] * stride_dkd
    dv_ptrs = dV + pid_b * stride_dvb + pid_h * stride_dvh + offs_n[:, None] * stride_dvs + offs_d[None, :] * stride_dvd
    
    tl.store(dk_ptrs, dk.to(k.dtype), mask=mask_n[:, None])
    tl.store(dv_ptrs, dv.to(v.dtype), mask=mask_n[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    with torch.cuda.device(Q.device):
        B, H, S, D_HEAD = Q.shape
        scale = 1.0 / math.sqrt(D_HEAD)

        grid_dq = (triton.cdiv(S, 64), H, B)
        bwd_kernel_dq[grid_dq](
            Q, K, V, O, dO, dQ, L,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
            dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
            L.stride(0), L.stride(1), L.stride(2),
            scale, S,
            BLOCK_M=64, BLOCK_N=64, D_HEAD=128,
            num_warps=4, num_stages=3
        )

        grid_dk_dv = (triton.cdiv(S, 64), H, B)
        bwd_kernel_dk_dv[grid_dk_dv](
            Q, K, V, O, dO, dK, dV, L,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
            dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
            dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
            L.stride(0), L.stride(1), L.stride(2),
            scale, S,
            BLOCK_M=64, BLOCK_N=64, D_HEAD=128,
            num_warps=4, num_stages=3
        )