import torch
import triton
import triton.language as tl
import math

@triton.jit
def _zero_kv_kernel(
    dK, dV,
    stride_k_b, stride_k_h, stride_k_s, stride_k_d,
    stride_v_b, stride_v_h, stride_v_s, stride_v_d,
    H, S,
    BLOCK_S: tl.constexpr, d: tl.constexpr
):
    pid_s = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    b = pid_bh // H
    h = pid_bh % H
    
    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_d = tl.arange(0, d)
    
    mask_s = offs_s < S
    valid_s_2d = mask_s[:, None] & (offs_d[None, :] < d)
    
    dk_ptrs = dK + b * stride_k_b + h * stride_k_h + offs_s[:, None] * stride_k_s + offs_d[None, :] * stride_k_d
    dv_ptrs = dV + b * stride_v_b + h * stride_v_h + offs_s[:, None] * stride_v_s + offs_d[None, :] * stride_v_d
    
    tl.store(dk_ptrs, 0.0, mask=valid_s_2d)
    tl.store(dv_ptrs, 0.0, mask=valid_s_2d)


@triton.jit
def _bwd_kernel(
    Q, K, V, O, dO, L,
    dQ, dK, dV,
    sm_scale,
    stride_q_b, stride_q_h, stride_q_s, stride_q_d,
    stride_k_b, stride_k_h, stride_k_s, stride_k_d,
    stride_v_b, stride_v_h, stride_v_s, stride_v_d,
    stride_o_b, stride_o_h, stride_o_s, stride_o_d,
    stride_do_b, stride_do_h, stride_do_s, stride_do_d,
    stride_l_b, stride_l_h, stride_l_s,
    stride_dq_b, stride_dq_h, stride_dq_s, stride_dq_d,
    stride_dk_b, stride_dk_h, stride_dk_s, stride_dk_d,
    stride_dv_b, stride_dv_h, stride_dv_s, stride_dv_d,
    H, S,
    d: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    b = pid_bh // H
    h = pid_bh % H
    
    q_base = Q + b * stride_q_b + h * stride_q_h
    k_base = K + b * stride_k_b + h * stride_k_h
    v_base = V + b * stride_v_b + h * stride_v_h
    o_base = O + b * stride_o_b + h * stride_o_h
    do_base = dO + b * stride_do_b + h * stride_do_h
    l_base = L + b * stride_l_b + h * stride_l_h
    dq_base = dQ + b * stride_dq_b + h * stride_dq_h
    dk_base = dK + b * stride_dk_b + h * stride_dk_h
    dv_base = dV + b * stride_dv_b + h * stride_dv_h
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, d)
    
    mask_m = offs_m < S
    valid_m_2d = mask_m[:, None] & (offs_d[None, :] < d)
    
    q_ptrs = q_base + offs_m[:, None] * stride_q_s + offs_d[None, :] * stride_q_d
    o_ptrs = o_base + offs_m[:, None] * stride_o_s + offs_d[None, :] * stride_o_d
    do_ptrs = do_base + offs_m[:, None] * stride_do_s + offs_d[None, :] * stride_do_d
    l_ptrs = l_base + offs_m * stride_l_s
    
    q = tl.load(q_ptrs, mask=valid_m_2d, other=0.0)
    o = tl.load(o_ptrs, mask=valid_m_2d, other=0.0)
    do = tl.load(do_ptrs, mask=valid_m_2d, other=0.0)
    lse = tl.load(l_ptrs, mask=mask_m, other=0.0)
    
    delta = tl.sum(do.to(tl.float32) * o.to(tl.float32), axis=1)
    
    dq = tl.zeros((BLOCK_M, d), dtype=tl.float32)
    
    max_q_idx = tl.minimum((pid_m + 1) * BLOCK_M - 1, S - 1)
    num_k_chunks = tl.cdiv(max_q_idx + 1, BLOCK_N)
    
    for k_idx in range(0, num_k_chunks):
        offs_n = k_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S
        valid_n_2d = mask_n[:, None] & (offs_d[None, :] < d)
        
        k_ptrs = k_base + offs_n[:, None] * stride_k_s + offs_d[None, :] * stride_k_d
        v_ptrs = v_base + offs_n[:, None] * stride_v_s + offs_d[None, :] * stride_v_d
        
        k = tl.load(k_ptrs, mask=valid_n_2d, other=0.0)
        v = tl.load(v_ptrs, mask=valid_n_2d, other=0.0)
        
        scores = tl.dot(q, tl.trans(k)) * sm_scale
        
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        valid_score = causal_mask & mask_m[:, None] & mask_n[None, :]
        
        scores = tl.where(valid_score, scores, -float("inf"))
        
        p = tl.math.exp(scores - lse[:, None])
        p = tl.where(valid_score, p, 0.0)
        
        dp = tl.dot(do, tl.trans(v))
        
        ds = p * (dp - delta[:, None]) * sm_scale
        ds = tl.where(valid_score, ds, 0.0)
        
        ds_dtype = ds.to(q.dtype)
        p_dtype = p.to(q.dtype)
        
        dq += tl.dot(ds_dtype, k)
        
        dk_partial = tl.dot(tl.trans(ds_dtype), q)
        dv_partial = tl.dot(tl.trans(p_dtype), do)
        
        dk_ptrs = dk_base + offs_n[:, None] * stride_dk_s + offs_d[None, :] * stride_dk_d
        dv_ptrs = dv_base + offs_n[:, None] * stride_dv_s + offs_d[None, :] * stride_dv_d
        
        tl.atomic_add(dk_ptrs, dk_partial.to(q.dtype), mask=valid_n_2d)
        tl.atomic_add(dv_ptrs, dv_partial.to(q.dtype), mask=valid_n_2d)
        
    dq_ptrs = dq_base + offs_m[:, None] * stride_dq_s + offs_d[None, :] * stride_dq_d
    tl.store(dq_ptrs, dq.to(q.dtype), mask=valid_m_2d)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes causal Multi-Head Attention backward efficiently. 
    Destination-passing pattern requires pre-allocated buffers dQ, dK, dV.
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, d = Q.shape
    sm_scale = 1.0 / math.sqrt(d)
    
    if S == 0:
        return
        
    grid_zero = (triton.cdiv(S, 64), B * H, 1)
    _zero_kv_kernel[grid_zero](
        dK, dV,
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        H, S,
        BLOCK_S=64, d=128
    )
    
    grid = (triton.cdiv(S, 64), B * H, 1)
    _bwd_kernel[grid](
        Q, K, V, O, dO, L,
        dQ, dK, dV,
        sm_scale,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        H=H, S=S,
        d=128, BLOCK_M=64, BLOCK_N=64,
        num_warps=4, num_stages=2
    )