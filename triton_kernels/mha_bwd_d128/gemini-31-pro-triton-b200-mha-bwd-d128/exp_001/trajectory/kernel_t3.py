import math
import torch
import triton
import triton.language as tl

# Triton allocator for device-side tensor-descriptor storage creation.
def _alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(_alloc_fn)

def get_configs_dq():
    return [
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64},  num_warps=4, num_stages=4),
    ]

def get_configs_dkv():
    return [
        triton.Config({'BLOCK_N': 128, 'BLOCK_M': 64},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_M': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 64,  'BLOCK_M': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 64,  'BLOCK_M': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 64,  'BLOCK_M': 64},  num_warps=4, num_stages=4),
    ]

@triton.autotune(configs=get_configs_dq(), key=['S'])
@triton.jit
def bwd_dq_kernel_tma(
    Q, K, V, O, dO, dQ, L,
    stride_q_b, stride_q_h, stride_q_s, stride_q_d,
    stride_k_b, stride_k_h, stride_k_s, stride_k_d,
    stride_v_b, stride_v_h, stride_v_s, stride_v_d,
    stride_o_b, stride_o_h, stride_o_s, stride_o_d,
    stride_do_b, stride_do_h, stride_do_s, stride_do_d,
    stride_dq_b, stride_dq_h, stride_dq_s, stride_dq_d,
    stride_l_b, stride_l_h, stride_l_s,
    S, sm_scale, H,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    
    start_m = pid_m * BLOCK_M
    
    # Precompute structural bases to configure device-side TMA 2D tensors effectively
    q_base = Q + b * stride_q_b + h * stride_q_h
    o_base = O + b * stride_o_b + h * stride_o_h
    do_base = dO + b * stride_do_b + h * stride_do_h
    dq_base = dQ + b * stride_dq_b + h * stride_dq_h
    k_base = K + b * stride_k_b + h * stride_k_h
    v_base = V + b * stride_v_b + h * stride_v_h
    
    Q_desc = tl.make_tensor_descriptor(
        q_base, shape=[S, BLOCK_D], strides=[stride_q_s, stride_q_d],
        block_shape=[BLOCK_M, BLOCK_D], padding_option="zero"
    )
    O_desc = tl.make_tensor_descriptor(
        o_base, shape=[S, BLOCK_D], strides=[stride_o_s, stride_o_d],
        block_shape=[BLOCK_M, BLOCK_D], padding_option="zero"
    )
    dO_desc = tl.make_tensor_descriptor(
        do_base, shape=[S, BLOCK_D], strides=[stride_do_s, stride_do_d],
        block_shape=[BLOCK_M, BLOCK_D], padding_option="zero"
    )
    dQ_desc = tl.make_tensor_descriptor(
        dq_base, shape=[S, BLOCK_D], strides=[stride_dq_s, stride_dq_d],
        block_shape=[BLOCK_M, BLOCK_D], padding_option="zero" 
    )
    K_desc = tl.make_tensor_descriptor(
        k_base, shape=[S, BLOCK_D], strides=[stride_k_s, stride_k_d],
        block_shape=[BLOCK_N, BLOCK_D], padding_option="zero"
    )
    V_desc = tl.make_tensor_descriptor(
        v_base, shape=[S, BLOCK_D], strides=[stride_v_s, stride_v_d],
        block_shape=[BLOCK_N, BLOCK_D], padding_option="zero"
    )
    
    q = Q_desc.load([start_m, 0])
    o = O_desc.load([start_m, 0])
    do = dO_desc.load([start_m, 0])
    
    # Computes once analytically per M outer block avoiding inner loop overhead
    delta = tl.sum(do.to(tl.float32) * o.to(tl.float32), axis=1)
    dq = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    
    offs_m = start_m + tl.arange(0, BLOCK_M)
    mask_m = offs_m < S
    
    l_ptrs = L + b * stride_l_b + h * stride_l_h + offs_m * stride_l_s
    l = tl.load(l_ptrs, mask=mask_m, other=0.0)
    
    for n_idx in tl.range(0, tl.cdiv(S, BLOCK_N)):
        start_n = n_idx * BLOCK_N
        k = K_desc.load([start_n, 0])
        v = V_desc.load([start_n, 0])
        
        qk = tl.dot(q, k.T) * sm_scale
        p = tl.exp(qk - l[:, None])
        
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S
        p = tl.where(mask_m[:, None] & mask_n[None, :], p, 0.0)
        
        dp = tl.dot(do, v.T) - delta[:, None]
        dp = p * dp
        
        dq = tl.dot(dp.to(q.dtype), k, acc=dq)
        
    dq *= sm_scale
    dQ_desc.store([start_m, 0], dq.to(q.dtype))


@triton.autotune(configs=get_configs_dkv(), key=['S'])
@triton.jit
def bwd_dk_dv_kernel_tma(
    Q, K, V, O, dO, dK, dV, L,
    stride_q_b, stride_q_h, stride_q_s, stride_q_d,
    stride_k_b, stride_k_h, stride_k_s, stride_k_d,
    stride_v_b, stride_v_h, stride_v_s, stride_v_d,
    stride_o_b, stride_o_h, stride_o_s, stride_o_d,
    stride_do_b, stride_do_h, stride_do_s, stride_do_d,
    stride_dk_b, stride_dk_h, stride_dk_s, stride_dk_d,
    stride_dv_b, stride_dv_h, stride_dv_s, stride_dv_d,
    stride_l_b, stride_l_h, stride_l_s,
    S, sm_scale, H,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    
    start_n = pid_n * BLOCK_N
    
    q_base = Q + b * stride_q_b + h * stride_q_h
    k_base = K + b * stride_k_b + h * stride_k_h
    v_base = V + b * stride_v_b + h * stride_v_h
    o_base = O + b * stride_o_b + h * stride_o_h
    do_base = dO + b * stride_do_b + h * stride_do_h
    dk_base = dK + b * stride_dk_b + h * stride_dk_h
    dv_base = dV + b * stride_dv_b + h * stride_dv_h
    
    K_desc = tl.make_tensor_descriptor(
        k_base, shape=[S, BLOCK_D], strides=[stride_k_s, stride_k_d],
        block_shape=[BLOCK_N, BLOCK_D], padding_option="zero"
    )
    V_desc = tl.make_tensor_descriptor(
        v_base, shape=[S, BLOCK_D], strides=[stride_v_s, stride_v_d],
        block_shape=[BLOCK_N, BLOCK_D], padding_option="zero"
    )
    dK_desc = tl.make_tensor_descriptor(
        dk_base, shape=[S, BLOCK_D], strides=[stride_dk_s, stride_dk_d],
        block_shape=[BLOCK_N, BLOCK_D], padding_option="zero"
    )
    dV_desc = tl.make_tensor_descriptor(
        dv_base, shape=[S, BLOCK_D], strides=[stride_dv_s, stride_dv_d],
        block_shape=[BLOCK_N, BLOCK_D], padding_option="zero"
    )
    
    Q_desc = tl.make_tensor_descriptor(
        q_base, shape=[S, BLOCK_D], strides=[stride_q_s, stride_q_d],
        block_shape=[BLOCK_M, BLOCK_D], padding_option="zero"
    )
    O_desc = tl.make_tensor_descriptor(
        o_base, shape=[S, BLOCK_D], strides=[stride_o_s, stride_o_d],
        block_shape=[BLOCK_M, BLOCK_D], padding_option="zero"
    )
    dO_desc = tl.make_tensor_descriptor(
        do_base, shape=[S, BLOCK_D], strides=[stride_do_s, stride_do_d],
        block_shape=[BLOCK_M, BLOCK_D], padding_option="zero"
    )
    
    k = K_desc.load([start_n, 0])
    v = V_desc.load([start_n, 0])
    
    dk = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    dv = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    
    offs_n = start_n + tl.arange(0, BLOCK_N)
    mask_n = offs_n < S
    
    for m_idx in tl.range(0, tl.cdiv(S, BLOCK_M)):
        start_m = m_idx * BLOCK_M
        
        q = Q_desc.load([start_m, 0])
        o = O_desc.load([start_m, 0])
        do = dO_desc.load([start_m, 0])
        
        offs_m = start_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S
        
        l_ptrs = L + b * stride_l_b + h * stride_l_h + offs_m * stride_l_s
        l = tl.load(l_ptrs, mask=mask_m, other=0.0)
        
        delta = tl.sum(do.to(tl.float32) * o.to(tl.float32), axis=1)
        
        kq = tl.dot(k, q.T) * sm_scale
        p = tl.exp(kq - l[None, :])
        p = tl.where(mask_n[:, None] & mask_m[None, :], p, 0.0)
        
        dv = tl.dot(p.to(q.dtype), do, acc=dv)
        
        dp = tl.dot(v, do.T) - delta[None, :]
        dp = p * dp
        
        dk = tl.dot(dp.to(q.dtype), q, acc=dk)
        
    dk *= sm_scale
    
    # Store strictly locally collected accumulation ensuring atomics completely avoided natively 
    dK_desc.store([start_n, 0], dk.to(k.dtype))
    dV_desc.store([start_n, 0], dv.to(v.dtype))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Evaluates completely inplace analytical Multi-head Attention Backward.
    Strictly coordinates two autonomous grid pathways guaranteeing mathematically coherent precision. 
    """
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    sm_scale = 1.0 / math.sqrt(d)
    
    # Phase 1: Compute dQ resolving sequentially stationary iterations across memory footprints.
    grid_m = lambda META: (triton.cdiv(S, META['BLOCK_M']), B * H)
    bwd_dq_kernel_tma[grid_m](
        Q, K, V, O, dO, dQ, L,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        S, sm_scale, H,
        BLOCK_D=128,
    )
    
    # Phase 2: Compute properly synchronized dK, dV natively clustered mathematically eliminating contention.
    grid_n = lambda META: (triton.cdiv(S, META['BLOCK_N']), B * H)
    bwd_dk_dv_kernel_tma[grid_n](
        Q, K, V, O, dO, dK, dV, L,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        S, sm_scale, H,
        BLOCK_D=128,
    )