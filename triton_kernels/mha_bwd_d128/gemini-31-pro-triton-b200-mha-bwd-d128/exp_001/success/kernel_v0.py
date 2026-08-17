import math
import torch
import triton
import triton.language as tl

def get_configs():
    return [
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
    ]

@triton.autotune(configs=get_configs(), key=['S'])
@triton.jit
def bwd_dq_kernel(
    Q, K, V, O, dO, L, dQ,
    stride_q_b, stride_q_h, stride_q_s, stride_q_d,
    stride_k_b, stride_k_h, stride_k_s, stride_k_d,
    stride_v_b, stride_v_h, stride_v_s, stride_v_d,
    stride_o_b, stride_o_h, stride_o_s, stride_o_d,
    stride_do_b, stride_do_h, stride_do_s, stride_do_d,
    stride_l_b, stride_l_h, stride_l_s,
    stride_dq_b, stride_dq_h, stride_dq_s, stride_dq_d,
    S, sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    b = pid_bh // 48
    h = pid_bh % 48
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    
    q_ptrs = Q + b * stride_q_b + h * stride_q_h + offs_m[:, None] * stride_q_s + offs_d[None, :] * stride_q_d
    o_ptrs = O + b * stride_o_b + h * stride_o_h + offs_m[:, None] * stride_o_s + offs_d[None, :] * stride_o_d
    do_ptrs = dO + b * stride_do_b + h * stride_do_h + offs_m[:, None] * stride_do_s + offs_d[None, :] * stride_do_d
    l_ptrs = L + b * stride_l_b + h * stride_l_h + offs_m * stride_l_s
    
    mask_m = offs_m < S
    
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
    do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
    l = tl.load(l_ptrs, mask=mask_m, other=0.0)
    
    delta = tl.sum(do.to(tl.float32) * o.to(tl.float32), axis=1)
    
    dq = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    
    k_ptrs = K + b * stride_k_b + h * stride_k_h + offs_d[None, :] * stride_k_d
    v_ptrs = V + b * stride_v_b + h * stride_v_h + offs_d[None, :] * stride_v_d
    
    for start_n in tl.range(0, tl.cdiv(S, BLOCK_N)):
        offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S
        
        k_curr = tl.load(k_ptrs + offs_n[:, None] * stride_k_s, mask=mask_n[:, None], other=0.0)
        v_curr = tl.load(v_ptrs + offs_n[:, None] * stride_v_s, mask=mask_n[:, None], other=0.0)
        
        qk = tl.dot(q, k_curr.T) * sm_scale
        p = tl.exp(qk - l[:, None])
        p = tl.where(mask_m[:, None] & mask_n[None, :], p, 0.0)
        
        dp = tl.dot(do, v_curr.T) - delta[:, None]
        dp = p * dp
        
        dq += tl.dot(dp.to(tl.bfloat16), k_curr)
        
    dq *= sm_scale
    
    dq_ptrs = dQ + b * stride_dq_b + h * stride_dq_h + offs_m[:, None] * stride_dq_s + offs_d[None, :] * stride_dq_d
    tl.store(dq_ptrs, dq.to(tl.bfloat16), mask=mask_m[:, None])

@triton.autotune(configs=get_configs(), key=['S'])
@triton.jit
def bwd_dk_dv_kernel(
    Q, K, V, O, dO, L, dK, dV,
    stride_q_b, stride_q_h, stride_q_s, stride_q_d,
    stride_k_b, stride_k_h, stride_k_s, stride_k_d,
    stride_v_b, stride_v_h, stride_v_s, stride_v_d,
    stride_o_b, stride_o_h, stride_o_s, stride_o_d,
    stride_do_b, stride_do_h, stride_do_s, stride_do_d,
    stride_l_b, stride_l_h, stride_l_s,
    stride_dk_b, stride_dk_h, stride_dk_s, stride_dk_d,
    stride_dv_b, stride_dv_h, stride_dv_s, stride_dv_d,
    S, sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    b = pid_bh // 48
    h = pid_bh % 48
    
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    
    mask_n = offs_n < S
    
    k_ptrs = K + b * stride_k_b + h * stride_k_h + offs_n[:, None] * stride_k_s + offs_d[None, :] * stride_k_d
    v_ptrs = V + b * stride_v_b + h * stride_v_h + offs_n[:, None] * stride_v_s + offs_d[None, :] * stride_v_d
    
    k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
    
    dk = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    dv = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    
    q_ptrs_base = Q + b * stride_q_b + h * stride_q_h + offs_d[None, :] * stride_q_d
    o_ptrs_base = O + b * stride_o_b + h * stride_o_h + offs_d[None, :] * stride_o_d
    do_ptrs_base = dO + b * stride_do_b + h * stride_do_h + offs_d[None, :] * stride_do_d
    l_ptrs_base = L + b * stride_l_b + h * stride_l_h
    
    for start_m in tl.range(0, tl.cdiv(S, BLOCK_M)):
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S
        
        q_curr = tl.load(q_ptrs_base + offs_m[:, None] * stride_q_s, mask=mask_m[:, None], other=0.0)
        o_curr = tl.load(o_ptrs_base + offs_m[:, None] * stride_o_s, mask=mask_m[:, None], other=0.0)
        do_curr = tl.load(do_ptrs_base + offs_m[:, None] * stride_do_s, mask=mask_m[:, None], other=0.0)
        l_curr = tl.load(l_ptrs_base + offs_m * stride_l_s, mask=mask_m, other=0.0)
        
        delta = tl.sum(do_curr.to(tl.float32) * o_curr.to(tl.float32), axis=1)
        
        kq = tl.dot(k, q_curr.T) * sm_scale
        p = tl.exp(kq - l_curr[None, :])
        p = tl.where(mask_n[:, None] & mask_m[None, :], p, 0.0)
        
        dv += tl.dot(p.to(tl.bfloat16), do_curr)
        
        dp = tl.dot(v, do_curr.T) - delta[None, :]
        dp = p * dp
        
        dk += tl.dot(dp.to(tl.bfloat16), q_curr)
        
    dk *= sm_scale
    
    dk_ptrs = dK + b * stride_dk_b + h * stride_dk_h + offs_n[:, None] * stride_dk_s + offs_d[None, :] * stride_dk_d
    dv_ptrs = dV + b * stride_dv_b + h * stride_dv_h + offs_n[:, None] * stride_dv_s + offs_d[None, :] * stride_dv_d
    
    tl.store(dk_ptrs, dk.to(tl.bfloat16), mask=mask_n[:, None])
    tl.store(dv_ptrs, dv.to(tl.bfloat16), mask=mask_n[:, None])

def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes standard FlashAttention backward pass for unmasked multi-head attention.
    Expects Q, K, V, O, dO of shape [B, H, S, d] in bfloat16.
    Expects L of shape [B, H, S] in float32.
    Outputs are written to properly allocated dQ, dK, dV of shape [B, H, S, d].
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, d = Q.shape
    sm_scale = 1.0 / math.sqrt(d)
    
    grid_m = lambda META: (triton.cdiv(S, META['BLOCK_M']), B * H)
    bwd_dq_kernel[grid_m](
        Q, K, V, O, dO, L, dQ,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        S, sm_scale,
        BLOCK_D=128,
    )
    
    grid_n = lambda META: (triton.cdiv(S, META['BLOCK_N']), B * H)
    bwd_dk_dv_kernel[grid_n](
        Q, K, V, O, dO, L, dK, dV,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        S, sm_scale,
        BLOCK_D=128,
    )