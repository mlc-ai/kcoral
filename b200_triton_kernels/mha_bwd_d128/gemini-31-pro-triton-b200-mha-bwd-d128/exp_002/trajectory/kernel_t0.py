import math
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
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
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    stride_lb, stride_lh, stride_ls,
    S, alpha, H: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    b = pid_bh // H
    h = pid_bh % H
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < S
    offs_d = tl.arange(0, 128)
    
    Q_ptr = Q + b * stride_qb + h * stride_qh + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    O_ptr = O + b * stride_ob + h * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    dO_ptr = dO + b * stride_dob + h * stride_doh + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod
    L_ptr = L + b * stride_lb + h * stride_lh + offs_m * stride_ls
    
    q = tl.load(Q_ptr, mask=mask_m[:, None], other=0.0)
    o = tl.load(O_ptr, mask=mask_m[:, None], other=0.0)
    do = tl.load(dO_ptr, mask=mask_m[:, None], other=0.0)
    l = tl.load(L_ptr, mask=mask_m, other=0.0)
    
    o_fp32 = o.to(tl.float32)
    do_fp32 = do.to(tl.float32)
    d_val = tl.sum(o_fp32 * do_fp32, axis=1)
    
    dq = tl.zeros([BLOCK_M, 128], dtype=tl.float32)
    
    K_base = K + b * stride_kb + h * stride_kh + offs_d[None, :] * stride_kd
    V_base = V + b * stride_vb + h * stride_vh + offs_d[None, :] * stride_vd
    
    for start_n in range(0, S, BLOCK_N):
        curr_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = curr_n < S
        
        k = tl.load(K_base + curr_n[:, None] * stride_ks, mask=mask_n[:, None], other=0.0)
        v = tl.load(V_base + curr_n[:, None] * stride_vs, mask=mask_n[:, None], other=0.0)
        
        s_attn = tl.dot(q, k.trans())
        s_attn = s_attn * alpha
        p = tl.exp(s_attn - l[:, None])
        p = tl.where(mask_m[:, None] & mask_n[None, :], p, 0.0)
        
        dp = tl.dot(do, v.trans())
        
        da = p * (dp - d_val[:, None])
        da = da.to(tl.bfloat16)
        
        dq += tl.dot(da, k)
        
    dq = dq * alpha
    
    dQ_ptr = dQ + b * stride_dqb + h * stride_dqh + offs_m[:, None] * stride_dqs + offs_d[None, :] * stride_dqd
    tl.store(dQ_ptr, dq.to(tl.bfloat16), mask=mask_m[:, None])

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 64, 'BLOCK_M': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 64, 'BLOCK_M': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_M': 128}, num_warps=8, num_stages=3),
    ],
    key=['S']
)
@triton.jit
def bwd_dk_dv_kernel(
    Q, K, V, O, dO, L, dK, dV,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    stride_lb, stride_lh, stride_ls,
    S, alpha, H: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    b = pid_bh // H
    h = pid_bh % H
    
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < S
    offs_d = tl.arange(0, 128)
    
    K_ptr = K + b * stride_kb + h * stride_kh + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    V_ptr = V + b * stride_vb + h * stride_vh + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
    
    k = tl.load(K_ptr, mask=mask_n[:, None], other=0.0)
    v = tl.load(V_ptr, mask=mask_n[:, None], other=0.0)
    
    dk = tl.zeros([BLOCK_N, 128], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, 128], dtype=tl.float32)
    
    Q_base = Q + b * stride_qb + h * stride_qh + offs_d[None, :] * stride_qd
    O_base = O + b * stride_ob + h * stride_oh + offs_d[None, :] * stride_od
    dO_base = dO + b * stride_dob + h * stride_doh + offs_d[None, :] * stride_dod
    L_base = L + b * stride_lb + h * stride_lh
    
    for start_m in range(0, S, BLOCK_M):
        curr_m = start_m + tl.arange(0, BLOCK_M)
        mask_m = curr_m < S
        
        q = tl.load(Q_base + curr_m[:, None] * stride_qs, mask=mask_m[:, None], other=0.0)
        o = tl.load(O_base + curr_m[:, None] * stride_os, mask=mask_m[:, None], other=0.0)
        do = tl.load(dO_base + curr_m[:, None] * stride_dos, mask=mask_m[:, None], other=0.0)
        l = tl.load(L_base + curr_m * stride_ls, mask=mask_m, other=0.0)
        
        o_fp32 = o.to(tl.float32)
        do_fp32 = do.to(tl.float32)
        d_val = tl.sum(o_fp32 * do_fp32, axis=1)
        
        s_attn = tl.dot(q, k.trans())
        s_attn = s_attn * alpha
        p = tl.exp(s_attn - l[:, None])
        p = tl.where(mask_m[:, None] & mask_n[None, :], p, 0.0)
        
        p_bf16 = p.to(tl.bfloat16)
        dv += tl.dot(p_bf16.trans(), do)
        
        dp = tl.dot(do, v.trans())
        
        da = p * (dp - d_val[:, None])
        da = da.to(tl.bfloat16)
        
        dk += tl.dot(da.trans(), q)
        
    dk = dk * alpha
    
    dK_ptr = dK + b * stride_dkb + h * stride_dkh + offs_n[:, None] * stride_dks + offs_d[None, :] * stride_dkd
    dV_ptr = dV + b * stride_dvb + h * stride_dvh + offs_n[:, None] * stride_dvs + offs_d[None, :] * stride_dvd
    
    tl.store(dK_ptr, dk.to(tl.bfloat16), mask=mask_n[:, None])
    tl.store(dV_ptr, dv.to(tl.bfloat16), mask=mask_n[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes FlashAttention backward pass for sequence blocks without causal mask.
    All inputs and outputs are in Bfloat16.
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, d = Q.shape
    alpha = 1.0 / math.sqrt(d)
    
    # Kernel 1: computes dQ using Q blocks mapping and K/V blocks inner loops
    grid_dq = lambda META: (triton.cdiv(S, META['BLOCK_M']), B * H)
    bwd_dq_kernel[grid_dq](
        Q, K, V, O, dO, L, dQ,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        S, alpha, H
    )
    
    # Kernel 2: computes dK and dV using K/V blocks mapping and Q blocks inner loops
    grid_dkv = lambda META: (triton.cdiv(S, META['BLOCK_N']), B * H)
    bwd_dk_dv_kernel[grid_dkv](
        Q, K, V, O, dO, L, dK, dV,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        S, alpha, H
    )