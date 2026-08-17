import math
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=3),
    ],
    key=["S"],
)
@triton.jit
def bwd_kernel_dq(
    Q, K, V, O, dO, L, dQ,
    sm_scale,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    B, H, S, d: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    b = pid_bh // H
    h = pid_bh % H
    
    q_offset = b * stride_qb + h * stride_qh
    k_offset = b * stride_kb + h * stride_kh
    v_offset = b * stride_vb + h * stride_vh
    o_offset = b * stride_ob + h * stride_oh
    do_offset = b * stride_dob + h * stride_doh
    l_offset = b * stride_lb + h * stride_lh
    dq_offset = b * stride_dqb + h * stride_dqh
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, d)
    
    mask_m = offs_m < S
    mask_md = mask_m[:, None]
    
    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    o_ptrs = O + o_offset + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    do_ptrs = dO + do_offset + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod
    
    q = tl.load(q_ptrs, mask=mask_md, other=0.0)
    o = tl.load(o_ptrs, mask=mask_md, other=0.0)
    do = tl.load(do_ptrs, mask=mask_md, other=0.0)
    
    l_ptrs = L + l_offset + offs_m * stride_ls
    l = tl.load(l_ptrs, mask=mask_m, other=0.0)
    
    do_f32 = do.to(tl.float32)
    o_f32 = o.to(tl.float32)
    D = tl.sum(do_f32 * o_f32, axis=1)
    
    acc_dq = tl.zeros([BLOCK_M, d], dtype=tl.float32)
    
    for j in range(0, tl.cdiv(S, BLOCK_N)):
        offs_n = j * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S
        mask_nd = mask_n[:, None]
        
        k_ptrs = K + k_offset + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        v_ptrs = V + v_offset + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
        
        k = tl.load(k_ptrs, mask=mask_nd, other=0.0)
        v = tl.load(v_ptrs, mask=mask_nd, other=0.0)
        
        s = tl.dot(q, tl.trans(k)) * sm_scale
        
        p = tl.exp(s - l[:, None])
        mask_mn = mask_m[:, None] & mask_n[None, :]
        p = tl.where(mask_mn, p, 0.0)
        
        dp = tl.dot(do, tl.trans(v))
        
        ds = p * (dp - D[:, None])
        ds = ds * sm_scale
        
        acc_dq = tl.dot(ds.to(tl.bfloat16), k, acc=acc_dq)
        
    dq_ptrs = dQ + dq_offset + offs_m[:, None] * stride_dqs + offs_d[None, :] * stride_dqd
    tl.store(dq_ptrs, acc_dq.to(tl.bfloat16), mask=mask_md)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 64, "BLOCK_M": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_N": 64, "BLOCK_M": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_N": 128, "BLOCK_M": 64}, num_warps=8, num_stages=3),
    ],
    key=["S"],
)
@triton.jit
def bwd_kernel_dk_dv(
    Q, K, V, O, dO, L, dK, dV,
    sm_scale,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    B, H, S, d: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    b = pid_bh // H
    h = pid_bh % H
    
    q_offset = b * stride_qb + h * stride_qh
    k_offset = b * stride_kb + h * stride_kh
    v_offset = b * stride_vb + h * stride_vh
    o_offset = b * stride_ob + h * stride_oh
    do_offset = b * stride_dob + h * stride_doh
    l_offset = b * stride_lb + h * stride_lh
    dk_offset = b * stride_dkb + h * stride_dkh
    dv_offset = b * stride_dvb + h * stride_dvh
    
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, d)
    mask_n = offs_n < S
    mask_nd = mask_n[:, None]
    
    k_ptrs = K + k_offset + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = V + v_offset + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
    
    k = tl.load(k_ptrs, mask=mask_nd, other=0.0)
    v = tl.load(v_ptrs, mask=mask_nd, other=0.0)
    
    acc_dk = tl.zeros([BLOCK_N, d], dtype=tl.float32)
    acc_dv = tl.zeros([BLOCK_N, d], dtype=tl.float32)
    
    for i in range(0, tl.cdiv(S, BLOCK_M)):
        offs_m = i * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S
        mask_md = mask_m[:, None]
        
        q_ptrs = Q + q_offset + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
        o_ptrs = O + o_offset + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
        do_ptrs = dO + do_offset + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod
        
        q = tl.load(q_ptrs, mask=mask_md, other=0.0)
        o = tl.load(o_ptrs, mask=mask_md, other=0.0)
        do = tl.load(do_ptrs, mask=mask_md, other=0.0)
        
        l_ptrs = L + l_offset + offs_m * stride_ls
        l = tl.load(l_ptrs, mask=mask_m, other=0.0)
        
        do_f32 = do.to(tl.float32)
        o_f32 = o.to(tl.float32)
        D = tl.sum(do_f32 * o_f32, axis=1)
        
        s = tl.dot(q, tl.trans(k)) * sm_scale
        
        p = tl.exp(s - l[:, None])
        mask_mn = mask_m[:, None] & mask_n[None, :]
        p = tl.where(mask_mn, p, 0.0)
        
        dp = tl.dot(do, tl.trans(v))
        
        ds = p * (dp - D[:, None])
        ds = ds * sm_scale
        
        acc_dk = tl.dot(tl.trans(ds.to(tl.bfloat16)), q, acc=acc_dk)
        acc_dv = tl.dot(tl.trans(p.to(tl.bfloat16)), do, acc=acc_dv)
        
    dk_ptrs = dK + dk_offset + offs_n[:, None] * stride_dks + offs_d[None, :] * stride_dkd
    dv_ptrs = dV + dv_offset + offs_n[:, None] * stride_dvs + offs_d[None, :] * stride_dvd
    
    tl.store(dk_ptrs, acc_dk.to(tl.bfloat16), mask=mask_nd)
    tl.store(dv_ptrs, acc_dv.to(tl.bfloat16), mask=mask_nd)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    with torch.cuda.device(Q.device):
        B, H, S, d = Q.shape
        sm_scale = 1.0 / math.sqrt(d)
        
        # In case the sequence logsumexp retains the final unary dimension
        if L.dim() == 4:
            L = L.squeeze(-1)
            
        grid_dq = lambda META: (triton.cdiv(S, META['BLOCK_M']), B * H, 1)
        bwd_kernel_dq[grid_dq](
            Q, K, V, O, dO, L, dQ,
            sm_scale,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
            L.stride(0), L.stride(1), L.stride(2),
            dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
            B, H, S, d
        )
        
        grid_dk_dv = lambda META: (triton.cdiv(S, META['BLOCK_N']), B * H, 1)
        bwd_kernel_dk_dv[grid_dk_dv](
            Q, K, V, O, dO, L, dK, dV,
            sm_scale,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
            L.stride(0), L.stride(1), L.stride(2),
            dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
            dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
            B, H, S, d
        )