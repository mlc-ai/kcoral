import math
import torch
import triton
import triton.language as tl

def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

def get_dq_configs():
    return [
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'PIPELINE_STAGES': 2}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'PIPELINE_STAGES': 3}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'PIPELINE_STAGES': 3}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'PIPELINE_STAGES': 3}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'PIPELINE_STAGES': 4}, num_warps=8, num_stages=4),
    ]

@triton.autotune(configs=get_dq_configs(), key=['S'])
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
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    # Create TMA descriptors outside the loop to minimize overhead
    Q_desc = tl.make_tensor_descriptor(
        Q, shape=[B, H, S, BLOCK_D], strides=[stride_qb, stride_qh, stride_qs, stride_qd],
        block_shape=[1, 1, BLOCK_M, BLOCK_D], padding_option="zero"
    )
    dO_desc = tl.make_tensor_descriptor(
        dO, shape=[B, H, S, BLOCK_D], strides=[stride_dob, stride_doh, stride_dos, stride_dod],
        block_shape=[1, 1, BLOCK_M, BLOCK_D], padding_option="zero"
    )
    O_desc = tl.make_tensor_descriptor(
        O, shape=[B, H, S, BLOCK_D], strides=[stride_ob, stride_oh, stride_os, stride_od],
        block_shape=[1, 1, BLOCK_M, BLOCK_D], padding_option="zero"
    )
    dQ_desc = tl.make_tensor_descriptor(
        dQ, shape=[B, H, S, BLOCK_D], strides=[stride_dqb, stride_dqh, stride_dqs, stride_dqd],
        block_shape=[1, 1, BLOCK_M, BLOCK_D], padding_option="zero"
    )
    
    K_desc = tl.make_tensor_descriptor(
        K, shape=[B, H, S, BLOCK_D], strides=[stride_kb, stride_kh, stride_ks, stride_kd],
        block_shape=[1, 1, BLOCK_N, BLOCK_D], padding_option="zero"
    )
    V_desc = tl.make_tensor_descriptor(
        V, shape=[B, H, S, BLOCK_D], strides=[stride_vb, stride_vh, stride_vs, stride_vd],
        block_shape=[1, 1, BLOCK_N, BLOCK_D], padding_option="zero"
    )

    offset_m = pid_m * BLOCK_M
    
    # Load M-blocks once
    q = tl.reshape(Q_desc.load([b, h, offset_m, 0]), [BLOCK_M, BLOCK_D])
    do = tl.reshape(dO_desc.load([b, h, offset_m, 0]), [BLOCK_M, BLOCK_D])
    o = tl.reshape(O_desc.load([b, h, offset_m, 0]), [BLOCK_M, BLOCK_D])
    
    # Compute row-wise dot product D = sum(dO * O, axis=-1) outside loop
    D = tl.sum(tl.cast(do, tl.float32) * tl.cast(o, tl.float32), axis=1)
    
    offs_m = offset_m + tl.arange(0, BLOCK_M)
    l_ptrs = L + b * stride_lb + h * stride_lh + offs_m * stride_ls
    mask_m = offs_m < S
    l = tl.load(l_ptrs, mask=mask_m, other=0.0)

    dq = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    num_n_steps = tl.cdiv(S, BLOCK_N)
    
    # Pipeline the N loop using the explicitly matched parameter
    for j in tl.range(0, num_n_steps, num_stages=PIPELINE_STAGES):
        offset_n = j * BLOCK_N
        k = tl.reshape(K_desc.load([b, h, offset_n, 0]), [BLOCK_N, BLOCK_D])
        v = tl.reshape(V_desc.load([b, h, offset_n, 0]), [BLOCK_N, BLOCK_D])
        
        s = tl.dot(q, tl.trans(k)) * sm_scale
        
        offs_n = offset_n + tl.arange(0, BLOCK_N)
        mask = mask_m[:, None] & (offs_n[None, :] < S)
        
        p = tl.where(mask, tl.exp(s - l[:, None]), 0.0)
        p_bf16 = tl.cast(p, tl.bfloat16)
        
        dp = tl.dot(do, tl.trans(v))
        
        ds = p * (dp - D[:, None]) * sm_scale
        ds_bf16 = tl.cast(ds, tl.bfloat16)
        
        dq = tl.dot(ds_bf16, k, acc=dq)
        
    dQ_desc.store([b, h, offset_m, 0], tl.reshape(tl.cast(dq, tl.bfloat16), [1, 1, BLOCK_M, BLOCK_D]))


def get_dkdv_configs():
    return [
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'PIPELINE_STAGES': 2}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'PIPELINE_STAGES': 3}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'PIPELINE_STAGES': 3}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'PIPELINE_STAGES': 3}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'PIPELINE_STAGES': 4}, num_warps=8, num_stages=4),
    ]

@triton.autotune(configs=get_dkdv_configs(), key=['S'])
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
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    Q_desc = tl.make_tensor_descriptor(
        Q, shape=[B, H, S, BLOCK_D], strides=[stride_qb, stride_qh, stride_qs, stride_qd],
        block_shape=[1, 1, BLOCK_M, BLOCK_D], padding_option="zero"
    )
    dO_desc = tl.make_tensor_descriptor(
        dO, shape=[B, H, S, BLOCK_D], strides=[stride_dob, stride_doh, stride_dos, stride_dod],
        block_shape=[1, 1, BLOCK_M, BLOCK_D], padding_option="zero"
    )
    O_desc = tl.make_tensor_descriptor(
        O, shape=[B, H, S, BLOCK_D], strides=[stride_ob, stride_oh, stride_os, stride_od],
        block_shape=[1, 1, BLOCK_M, BLOCK_D], padding_option="zero"
    )
    
    K_desc = tl.make_tensor_descriptor(
        K, shape=[B, H, S, BLOCK_D], strides=[stride_kb, stride_kh, stride_ks, stride_kd],
        block_shape=[1, 1, BLOCK_N, BLOCK_D], padding_option="zero"
    )
    V_desc = tl.make_tensor_descriptor(
        V, shape=[B, H, S, BLOCK_D], strides=[stride_vb, stride_vh, stride_vs, stride_vd],
        block_shape=[1, 1, BLOCK_N, BLOCK_D], padding_option="zero"
    )
    dK_desc = tl.make_tensor_descriptor(
        dK, shape=[B, H, S, BLOCK_D], strides=[stride_dkb, stride_dkh, stride_dks, stride_dkd],
        block_shape=[1, 1, BLOCK_N, BLOCK_D], padding_option="zero"
    )
    dV_desc = tl.make_tensor_descriptor(
        dV, shape=[B, H, S, BLOCK_D], strides=[stride_dvb, stride_dvh, stride_dvs, stride_dvd],
        block_shape=[1, 1, BLOCK_N, BLOCK_D], padding_option="zero"
    )

    offset_n = pid_n * BLOCK_N
    
    # Load N-blocks once
    k = tl.reshape(K_desc.load([b, h, offset_n, 0]), [BLOCK_N, BLOCK_D])
    v = tl.reshape(V_desc.load([b, h, offset_n, 0]), [BLOCK_N, BLOCK_D])
    
    dk = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    
    offs_n = offset_n + tl.arange(0, BLOCK_N)
    mask_n = offs_n < S
    
    num_m_steps = tl.cdiv(S, BLOCK_M)
    
    # Pipeline the M loop using the explicitly matched parameter
    for i in tl.range(0, num_m_steps, num_stages=PIPELINE_STAGES):
        offset_m = i * BLOCK_M
        q = tl.reshape(Q_desc.load([b, h, offset_m, 0]), [BLOCK_M, BLOCK_D])
        do = tl.reshape(dO_desc.load([b, h, offset_m, 0]), [BLOCK_M, BLOCK_D])
        o = tl.reshape(O_desc.load([b, h, offset_m, 0]), [BLOCK_M, BLOCK_D])
        
        offs_m = offset_m + tl.arange(0, BLOCK_M)
        l_ptrs = L + b * stride_lb + h * stride_lh + offs_m * stride_ls
        l = tl.load(l_ptrs, mask=offs_m < S, other=0.0)
        
        # Computing delta term (D) on the fly
        D = tl.sum(tl.cast(do, tl.float32) * tl.cast(o, tl.float32), axis=1)
        
        mask = mask_n[:, None] & (offs_m[None, :] < S)
        
        # Avoid physical transposes of the mask components
        s_T = tl.dot(k, tl.trans(q)) * sm_scale
        
        p_T = tl.where(mask, tl.exp(s_T - l[None, :]), 0.0)
        p_T_bf16 = tl.cast(p_T, tl.bfloat16)
        
        dv = tl.dot(p_T_bf16, do, acc=dv)
        
        dp_T = tl.dot(v, tl.trans(do))
        ds_T = p_T * (dp_T - D[None, :]) * sm_scale
        ds_T_bf16 = tl.cast(ds_T, tl.bfloat16)
        
        dk = tl.dot(ds_T_bf16, q, acc=dk)
        
    dK_desc.store([b, h, offset_n, 0], tl.reshape(tl.cast(dk, tl.bfloat16), [1, 1, BLOCK_N, BLOCK_D]))
    dV_desc.store([b, h, offset_n, 0], tl.reshape(tl.cast(dv, tl.bfloat16), [1, 1, BLOCK_N, BLOCK_D]))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    sm_scale = 1.0 / math.sqrt(d)
    
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