import math
import torch
import triton
import triton.language as tl

def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

def get_dq_configs():
    # Strict shared memory bounds checked for 228KiB limit on Blackwell
    return [
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'NUM_STAGES': 2, 'WS': False}, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'NUM_STAGES': 2, 'WS': True}, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'NUM_STAGES': 3, 'WS': False}, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'NUM_STAGES': 3, 'WS': True}, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'NUM_STAGES': 2, 'WS': False}, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'NUM_STAGES': 2, 'WS': True}, num_warps=8),
    ]

@triton.autotune(configs=get_dq_configs(), key=['S'])
@triton.jit
def bwd_kernel_dq(
    Q, K, V, O, dO, L, dQ,
    stride_qb, stride_qh, stride_qs,
    stride_kb, stride_kh, stride_ks,
    stride_vb, stride_vh, stride_vs,
    stride_ob, stride_oh, stride_os,
    stride_dob, stride_doh, stride_dos,
    stride_lb, stride_lh, stride_ls,
    stride_dqb, stride_dqh, stride_dqs,
    B, H, S, sm_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    NUM_STAGES: tl.constexpr, WS: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    # Generate 2D TMA descriptors directly inside the kernel, allowing seamless hardware lowering 
    # without host replacement quirks. Strides default to 1 on the contiguous inner dimension.
    q_base = Q + b * stride_qb + h * stride_qh
    Q_desc = tl.make_tensor_descriptor(
        q_base, shape=[S, BLOCK_D], strides=[stride_qs, 1],
        block_shape=[BLOCK_M, BLOCK_D], padding_option="zero"
    )
    
    do_base = dO + b * stride_dob + h * stride_doh
    dO_desc = tl.make_tensor_descriptor(
        do_base, shape=[S, BLOCK_D], strides=[stride_dos, 1],
        block_shape=[BLOCK_M, BLOCK_D], padding_option="zero"
    )
    
    o_base = O + b * stride_ob + h * stride_oh
    O_desc = tl.make_tensor_descriptor(
        o_base, shape=[S, BLOCK_D], strides=[stride_os, 1],
        block_shape=[BLOCK_M, BLOCK_D], padding_option="zero"
    )
    
    k_base = K + b * stride_kb + h * stride_kh
    K_desc = tl.make_tensor_descriptor(
        k_base, shape=[S, BLOCK_D], strides=[stride_ks, 1],
        block_shape=[BLOCK_N, BLOCK_D], padding_option="zero"
    )
    
    v_base = V + b * stride_vb + h * stride_vh
    V_desc = tl.make_tensor_descriptor(
        v_base, shape=[S, BLOCK_D], strides=[stride_vs, 1],
        block_shape=[BLOCK_N, BLOCK_D], padding_option="zero"
    )
    
    dq_base = dQ + b * stride_dqb + h * stride_dqh
    dQ_desc = tl.make_tensor_descriptor(
        dq_base, shape=[S, BLOCK_D], strides=[stride_dqs, 1],
        block_shape=[BLOCK_M, BLOCK_D], padding_option="zero"
    )

    offset_m = pid_m * BLOCK_M
    
    q = Q_desc.load([offset_m, 0])
    do = dO_desc.load([offset_m, 0])
    o = O_desc.load([offset_m, 0])
    
    # Precompute Delta purely in registers since the inputs are loaded identically just once for M block
    D = tl.sum(tl.cast(do, tl.float32) * tl.cast(o, tl.float32), axis=1)
    
    offs_m = offset_m + tl.arange(0, BLOCK_M)
    l_ptrs = L + b * stride_lb + h * stride_lh + offs_m * stride_ls
    l = tl.load(l_ptrs, mask=offs_m < S, other=0.0)

    dq = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    num_n_steps = tl.cdiv(S, BLOCK_N)
    
    for j in tl.range(0, num_n_steps, num_stages=NUM_STAGES, warp_specialize=WS):
        offset_n = j * BLOCK_N
        k = K_desc.load([offset_n, 0])
        v = V_desc.load([offset_n, 0])
        
        # Executes naturally mapped: LHS (Row-Major) x RHS (Col-Major physically loaded as Row-Major + transposed view)
        s = tl.dot(q, tl.trans(k)) * sm_scale
        
        offs_n = offset_n + tl.arange(0, BLOCK_N)
        mask = (offs_m[:, None] < S) & (offs_n[None, :] < S)
        
        p = tl.where(mask, tl.exp(s - l[:, None]), 0.0)
        p_bf16 = tl.cast(p, tl.bfloat16)
        
        dp = tl.dot(do, tl.trans(v))
        
        ds = p * (dp - D[:, None]) * sm_scale
        ds_bf16 = tl.cast(ds, tl.bfloat16)
        
        dq = tl.dot(ds_bf16, k, acc=dq)
        
    dQ_desc.store([offset_m, 0], tl.cast(dq, tl.bfloat16))


def get_dkdv_configs():
    # Strict 228KiB capacity verified shared memory stages avoiding previous allocation spills
    return [
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'NUM_STAGES': 3, 'WS': False}, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'NUM_STAGES': 3, 'WS': True}, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'NUM_STAGES': 2, 'WS': False}, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'NUM_STAGES': 2, 'WS': True}, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'NUM_STAGES': 3, 'WS': False}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'NUM_STAGES': 3, 'WS': True}, num_warps=4),
    ]

@triton.autotune(configs=get_dkdv_configs(), key=['S'])
@triton.jit
def bwd_kernel_dk_dv(
    Q, K, V, O, dO, L, dK, dV,
    stride_qb, stride_qh, stride_qs,
    stride_kb, stride_kh, stride_ks,
    stride_vb, stride_vh, stride_vs,
    stride_ob, stride_oh, stride_os,
    stride_dob, stride_doh, stride_dos,
    stride_lb, stride_lh, stride_ls,
    stride_dkb, stride_dkh, stride_dks,
    stride_dvb, stride_dvh, stride_dvs,
    B, H, S, sm_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    NUM_STAGES: tl.constexpr, WS: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    q_base = Q + b * stride_qb + h * stride_qh
    Q_desc = tl.make_tensor_descriptor(
        q_base, shape=[S, BLOCK_D], strides=[stride_qs, 1],
        block_shape=[BLOCK_M, BLOCK_D], padding_option="zero"
    )
    
    do_base = dO + b * stride_dob + h * stride_doh
    dO_desc = tl.make_tensor_descriptor(
        do_base, shape=[S, BLOCK_D], strides=[stride_dos, 1],
        block_shape=[BLOCK_M, BLOCK_D], padding_option="zero"
    )
    
    o_base = O + b * stride_ob + h * stride_oh
    O_desc = tl.make_tensor_descriptor(
        o_base, shape=[S, BLOCK_D], strides=[stride_os, 1],
        block_shape=[BLOCK_M, BLOCK_D], padding_option="zero"
    )
    
    k_base = K + b * stride_kb + h * stride_kh
    K_desc = tl.make_tensor_descriptor(
        k_base, shape=[S, BLOCK_D], strides=[stride_ks, 1],
        block_shape=[BLOCK_N, BLOCK_D], padding_option="zero"
    )
    
    v_base = V + b * stride_vb + h * stride_vh
    V_desc = tl.make_tensor_descriptor(
        v_base, shape=[S, BLOCK_D], strides=[stride_vs, 1],
        block_shape=[BLOCK_N, BLOCK_D], padding_option="zero"
    )
    
    dk_base = dK + b * stride_dkb + h * stride_dkh
    dK_desc = tl.make_tensor_descriptor(
        dk_base, shape=[S, BLOCK_D], strides=[stride_dks, 1],
        block_shape=[BLOCK_N, BLOCK_D], padding_option="zero"
    )
    
    dv_base = dV + b * stride_dvb + h * stride_dvh
    dV_desc = tl.make_tensor_descriptor(
        dv_base, shape=[S, BLOCK_D], strides=[stride_dvs, 1],
        block_shape=[BLOCK_N, BLOCK_D], padding_option="zero"
    )

    offset_n = pid_n * BLOCK_N
    
    # N elements are kept static around the M pipeline iteration window
    k = K_desc.load([offset_n, 0])
    v = V_desc.load([offset_n, 0])
    
    dk = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    
    offs_n = offset_n + tl.arange(0, BLOCK_N)
    
    num_m_steps = tl.cdiv(S, BLOCK_M)
    for i in tl.range(0, num_m_steps, num_stages=NUM_STAGES, warp_specialize=WS):
        offset_m = i * BLOCK_M
        q = Q_desc.load([offset_m, 0])
        do = dO_desc.load([offset_m, 0])
        o = O_desc.load([offset_m, 0])
        
        offs_m = offset_m + tl.arange(0, BLOCK_M)
        l_ptrs = L + b * stride_lb + h * stride_lh + offs_m * stride_ls
        l = tl.load(l_ptrs, mask=offs_m < S, other=0.0)
        
        D = tl.sum(tl.cast(do, tl.float32) * tl.cast(o, tl.float32), axis=1)
        
        mask = (offs_n[:, None] < S) & (offs_m[None, :] < S)
        
        # Mathematical derivation applied avoiding heavy register layout conversions
        s_T = tl.dot(k, tl.trans(q)) * sm_scale
        
        p_T = tl.where(mask, tl.exp(s_T - l[None, :]), 0.0)
        p_T_bf16 = tl.cast(p_T, tl.bfloat16)
        
        dv = tl.dot(p_T_bf16, do, acc=dv)
        
        dp_T = tl.dot(v, tl.trans(do))
        ds_T = p_T * (dp_T - D[None, :]) * sm_scale
        ds_T_bf16 = tl.cast(ds_T, tl.bfloat16)
        
        dk = tl.dot(ds_T_bf16, q, acc=dk)
        
    dK_desc.store([offset_n, 0], tl.cast(dk, tl.bfloat16))
    dV_desc.store([offset_n, 0], tl.cast(dv, tl.bfloat16))

def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    sm_scale = 1.0 / math.sqrt(d)
    
    grid_dq = lambda META: (triton.cdiv(S, META['BLOCK_M']), B * H)
    bwd_kernel_dq[grid_dq](
        Q, K, V, O, dO, L, dQ,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        dO.stride(0), dO.stride(1), dO.stride(2),
        L.stride(0), L.stride(1), L.stride(2),
        dQ.stride(0), dQ.stride(1), dQ.stride(2),
        B, H, S, sm_scale,
        BLOCK_D=d
    )
    
    grid_dk_dv = lambda META: (triton.cdiv(S, META['BLOCK_N']), B * H)
    bwd_kernel_dk_dv[grid_dk_dv](
        Q, K, V, O, dO, L, dK, dV,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        dO.stride(0), dO.stride(1), dO.stride(2),
        L.stride(0), L.stride(1), L.stride(2),
        dK.stride(0), dK.stride(1), dK.stride(2),
        dV.stride(0), dV.stride(1), dV.stride(2),
        B, H, S, sm_scale,
        BLOCK_D=d
    )