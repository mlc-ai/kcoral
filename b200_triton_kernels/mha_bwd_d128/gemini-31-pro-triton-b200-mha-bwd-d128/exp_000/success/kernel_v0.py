import math
import torch
import triton
import triton.language as tl


def get_autotune_configs():
    return [
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
    ]


@triton.autotune(configs=get_autotune_configs(), key=['S'])
@triton.jit
def bwd_kernel_dq(
    Q, K, V, O, dO, L, dQ,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    B, H, S, sm_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    b = pid_bh // H
    h = pid_bh % H
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    
    q_base = Q + b * stride_qb + h * stride_qh
    do_base = dO + b * stride_dob + h * stride_doh
    o_base = O + b * stride_ob + h * stride_oh
    
    q_ptrs = q_base + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    do_ptrs = do_base + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod
    o_ptrs = o_base + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    l_ptrs = L + b * stride_lb + h * stride_lh + offs_m * stride_ls
    
    mask_m = offs_m < S
    
    q_i = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    do_i = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
    o_i = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
    l_i = tl.load(l_ptrs, mask=mask_m, other=0.0)
    
    # Compute row-wise dot product of dO_i and O_i
    D_i = tl.sum(tl.cast(do_i, tl.float32) * tl.cast(o_i, tl.float32), axis=1)
    
    dq_i = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    k_base = K + b * stride_kb + h * stride_kh
    v_base = V + b * stride_vb + h * stride_vh
    
    for j in range(0, tl.cdiv(S, BLOCK_N)):
        offs_n = j * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S
        
        k_ptrs = k_base + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        v_ptrs = v_base + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
        
        k_j = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        v_j = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        
        s_ij = tl.dot(q_i, tl.trans(k_j))
        s_ij = s_ij * sm_scale
        
        p_ij = tl.where(mask_m[:, None] & mask_n[None, :], tl.exp(s_ij - l_i[:, None]), 0.0)
        
        dp_ij = tl.dot(do_i, tl.trans(v_j))
        
        ds_ij = p_ij * (dp_ij - D_i[:, None]) * sm_scale
        
        ds_ij_bf16 = tl.cast(ds_ij, tl.bfloat16)
        dq_i = tl.dot(ds_ij_bf16, k_j, acc=dq_i)
        
    dq_base = dQ + b * stride_dqb + h * stride_dqh
    dq_ptrs = dq_base + offs_m[:, None] * stride_dqs + offs_d[None, :] * stride_dqd
    tl.store(dq_ptrs, tl.cast(dq_i, tl.bfloat16), mask=mask_m[:, None])


@triton.autotune(configs=get_autotune_configs(), key=['S'])
@triton.jit
def bwd_kernel_dk_dv(
    Q, K, V, O, dO, L, dK, dV,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    B, H, S, sm_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    b = pid_bh // H
    h = pid_bh % H
    
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    
    k_base = K + b * stride_kb + h * stride_kh
    v_base = V + b * stride_vb + h * stride_vh
    
    k_ptrs = k_base + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = v_base + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
    
    mask_n = offs_n < S
    
    k_j = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
    v_j = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
    
    dk_j = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    dv_j = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    
    q_base = Q + b * stride_qb + h * stride_qh
    do_base = dO + b * stride_dob + h * stride_doh
    o_base = O + b * stride_ob + h * stride_oh
    
    for i in range(0, tl.cdiv(S, BLOCK_M)):
        offs_m = i * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S
        
        q_ptrs = q_base + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
        do_ptrs = do_base + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod
        o_ptrs = o_base + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
        l_ptrs = L + b * stride_lb + h * stride_lh + offs_m * stride_ls
        
        q_i = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
        do_i = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
        o_i = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
        l_i = tl.load(l_ptrs, mask=mask_m, other=0.0)
        
        D_i = tl.sum(tl.cast(do_i, tl.float32) * tl.cast(o_i, tl.float32), axis=1)
        
        s_ij = tl.dot(q_i, tl.trans(k_j))
        s_ij = s_ij * sm_scale
        
        p_ij = tl.where(mask_m[:, None] & mask_n[None, :], tl.exp(s_ij - l_i[:, None]), 0.0)
        
        p_ij_bf16 = tl.cast(p_ij, tl.bfloat16)
        dv_j = tl.dot(tl.trans(p_ij_bf16), do_i, acc=dv_j)
        
        dp_ij = tl.dot(do_i, tl.trans(v_j))
        
        ds_ij = p_ij * (dp_ij - D_i[:, None]) * sm_scale
        
        ds_ij_bf16 = tl.cast(ds_ij, tl.bfloat16)
        dk_j = tl.dot(tl.trans(ds_ij_bf16), q_i, acc=dk_j)
        
    dk_base = dK + b * stride_dkb + h * stride_dkh
    dv_base = dV + b * stride_dvb + h * stride_dvh
    
    dk_ptrs = dk_base + offs_n[:, None] * stride_dks + offs_d[None, :] * stride_dkd
    dv_ptrs = dv_base + offs_n[:, None] * stride_dvs + offs_d[None, :] * stride_dvd
    
    tl.store(dk_ptrs, tl.cast(dk_j, tl.bfloat16), mask=mask_n[:, None])
    tl.store(dv_ptrs, tl.cast(dv_j, tl.bfloat16), mask=mask_n[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    sm_scale = 1.0 / math.sqrt(d)
    
    # Kernel for dQ computation
    grid_dq = lambda META: (triton.cdiv(S, META['BLOCK_M']), B * H)
    bwd_kernel_dq[grid_dq](
        Q, K, V, O, dO, L, dQ,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        B, H, S, sm_scale,
        BLOCK_D=d
    )
    
    # Kernel for dK and dV computation
    grid_dk_dv = lambda META: (triton.cdiv(S, META['BLOCK_N']), B * H)
    bwd_kernel_dk_dv[grid_dk_dv](
        Q, K, V, O, dO, L, dK, dV,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        B, H, S, sm_scale,
        BLOCK_D=d
    )