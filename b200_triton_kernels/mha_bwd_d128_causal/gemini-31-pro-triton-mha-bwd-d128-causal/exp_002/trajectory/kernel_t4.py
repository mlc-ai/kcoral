import math
import torch
import triton
import triton.language as tl

# Provide Triton with a host-side allocation hook for device-created TMA descriptor structures.
# This prevents compilation errors when evaluating `tl.make_tensor_descriptor` inside the JIT.
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

# Configurations heavily optimize for Hopper WGMMA footprint.
# Bounded to 3 stages to prevent exceeding the 228KB H100 shared memory budget.
configs_dq = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
]

configs_dk = [
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
]


@triton.autotune(configs=configs_dq, key=['S'])
@triton.jit
def _bwd_kernel_dq(
    Q_ptr, K_ptr, V_ptr, O_ptr, DO_ptr, DQ_ptr, L_ptr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    stride_lb, stride_lh, stride_ls,
    H, S, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D_HEAD: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    b = pid_bh // H
    h = pid_bh % H
    
    m_start = pid_m * BLOCK_M
    off_m = m_start + tl.arange(0, BLOCK_M)
    
    # Establish per-head memory offsets
    q_base = Q_ptr + b * stride_qb + h * stride_qh
    do_base = DO_ptr + b * stride_dob + h * stride_doh
    o_base = O_ptr + b * stride_ob + h * stride_oh
    dq_base = DQ_ptr + b * stride_dqb + h * stride_dqh
    k_base = K_ptr + b * stride_kb + h * stride_kh
    v_base = V_ptr + b * stride_vb + h * stride_vh
    l_base = L_ptr + b * stride_lb + h * stride_lh

    # Create TMA descriptors bridging unconstrained pointer logic safely.
    q_desc = tl.make_tensor_descriptor(
        q_base, shape=[S, D_HEAD], strides=[stride_qs, stride_qd],
        block_shape=[BLOCK_M, D_HEAD], padding_option="zero"
    )
    do_desc = tl.make_tensor_descriptor(
        do_base, shape=[S, D_HEAD], strides=[stride_dos, stride_dod],
        block_shape=[BLOCK_M, D_HEAD], padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        o_base, shape=[S, D_HEAD], strides=[stride_os, stride_od],
        block_shape=[BLOCK_M, D_HEAD], padding_option="zero"
    )
    dq_desc = tl.make_tensor_descriptor(
        dq_base, shape=[S, D_HEAD], strides=[stride_dqs, stride_dqd],
        block_shape=[BLOCK_M, D_HEAD]
    )

    q = q_desc.load([m_start, 0])
    do = do_desc.load([m_start, 0])
    o = o_desc.load([m_start, 0])
    
    # Conventional loads for 'L' to skip 16-byte alignment bounds
    l_ptr = l_base + off_m * stride_ls
    l = tl.load(l_ptr, mask=off_m < S, other=0.0)

    # Precalculate delta per query block to minimize inner-loop dependencies
    do_f32 = do.to(tl.float32)
    delta = tl.sum(o.to(tl.float32) * do_f32, axis=1)

    dq = tl.zeros([BLOCK_M, D_HEAD], dtype=tl.float32)

    max_n = tl.minimum(S, m_start + BLOCK_M)
    num_n_blocks = (max_n + BLOCK_N - 1) // BLOCK_N

    k_desc = tl.make_tensor_descriptor(
        k_base, shape=[S, D_HEAD], strides=[stride_ks, stride_kd],
        block_shape=[BLOCK_N, D_HEAD], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        v_base, shape=[S, D_HEAD], strides=[stride_vs, stride_vd],
        block_shape=[BLOCK_N, D_HEAD], padding_option="zero"
    )

    off_n_base = tl.arange(0, BLOCK_N)

    for n_idx in range(num_n_blocks):
        n_start = n_idx * BLOCK_N
        off_n = n_start + off_n_base
        
        k = k_desc.load([n_start, 0])
        v = v_desc.load([n_start, 0])
        
        qk = tl.dot(q, k.T, out_dtype=tl.float32) * scale
        
        is_causal_boundary = (n_start + BLOCK_N > m_start)
        is_seq_bound = (n_start + BLOCK_N > S) or (m_start + BLOCK_M > S)
        
        if is_causal_boundary or is_seq_bound:
            mask_m = off_m < S
            mask_n = off_n < S
            valid = (off_m[:, None] >= off_n[None, :]) & mask_m[:, None] & mask_n[None, :]
            qk = tl.where(valid, qk, float('-inf'))
            p = tl.math.exp(qk - l[:, None])
            p = tl.where(valid, p, 0.0)
        else:
            p = tl.math.exp(qk - l[:, None])
            
        dp = tl.dot(do, v.T, out_dtype=tl.float32)
        ds = p * (dp - delta[:, None]) * scale
        
        dq += tl.dot(ds.to(q.dtype), k, out_dtype=tl.float32)
        
    dq_desc.store([m_start, 0], dq.to(q.dtype))


@triton.autotune(configs=configs_dk, key=['S'])
@triton.jit
def _bwd_kernel_dk_dv(
    Q_ptr, K_ptr, V_ptr, O_ptr, DO_ptr, DK_ptr, DV_ptr, L_ptr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    stride_lb, stride_lh, stride_ls,
    H, S, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D_HEAD: tl.constexpr
):
    pid_n = tl.program_id(0) 
    pid_bh = tl.program_id(1)
    
    b = pid_bh // H
    h = pid_bh % H
    
    n_start = pid_n * BLOCK_N
    off_n = n_start + tl.arange(0, BLOCK_N)
    
    q_base = Q_ptr + b * stride_qb + h * stride_qh
    k_base = K_ptr + b * stride_kb + h * stride_kh
    v_base = V_ptr + b * stride_vb + h * stride_vh
    o_base = O_ptr + b * stride_ob + h * stride_oh
    do_base = DO_ptr + b * stride_dob + h * stride_doh
    dk_base = DK_ptr + b * stride_dkb + h * stride_dkh
    dv_base = DV_ptr + b * stride_dvb + h * stride_dvh
    l_base = L_ptr + b * stride_lb + h * stride_lh

    k_desc = tl.make_tensor_descriptor(
        k_base, shape=[S, D_HEAD], strides=[stride_ks, stride_kd],
        block_shape=[BLOCK_N, D_HEAD], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        v_base, shape=[S, D_HEAD], strides=[stride_vs, stride_vd],
        block_shape=[BLOCK_N, D_HEAD], padding_option="zero"
    )
    dk_desc = tl.make_tensor_descriptor(
        dk_base, shape=[S, D_HEAD], strides=[stride_dks, stride_dkd],
        block_shape=[BLOCK_N, D_HEAD]
    )
    dv_desc = tl.make_tensor_descriptor(
        dv_base, shape=[S, D_HEAD], strides=[stride_dvs, stride_dvd],
        block_shape=[BLOCK_N, D_HEAD]
    )

    k = k_desc.load([n_start, 0])
    v = v_desc.load([n_start, 0])
    
    dk = tl.zeros([BLOCK_N, D_HEAD], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, D_HEAD], dtype=tl.float32)
    
    start_m_idx = n_start // BLOCK_M
    num_m_blocks = (S + BLOCK_M - 1) // BLOCK_M
    
    q_desc = tl.make_tensor_descriptor(
        q_base, shape=[S, D_HEAD], strides=[stride_qs, stride_qd],
        block_shape=[BLOCK_M, D_HEAD], padding_option="zero"
    )
    do_desc = tl.make_tensor_descriptor(
        do_base, shape=[S, D_HEAD], strides=[stride_dos, stride_dod],
        block_shape=[BLOCK_M, D_HEAD], padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        o_base, shape=[S, D_HEAD], strides=[stride_os, stride_od],
        block_shape=[BLOCK_M, D_HEAD], padding_option="zero"
    )
    
    off_m_base = tl.arange(0, BLOCK_M)
    
    for m_idx in range(start_m_idx, num_m_blocks):
        m_start = m_idx * BLOCK_M
        off_m = m_start + off_m_base
        
        q = q_desc.load([m_start, 0])
        do = do_desc.load([m_start, 0])
        o = o_desc.load([m_start, 0])
        
        l_ptr = l_base + off_m * stride_ls
        l = tl.load(l_ptr, mask=off_m < S, other=0.0)
        
        do_f32 = do.to(tl.float32)
        delta = tl.sum(o.to(tl.float32) * do_f32, axis=1)
        
        qk = tl.dot(q, k.T, out_dtype=tl.float32) * scale
        
        is_causal_boundary = (m_start < n_start + BLOCK_N)
        is_seq_bound = (m_start + BLOCK_M > S) or (n_start + BLOCK_N > S)
        
        if is_causal_boundary or is_seq_bound:
            mask_m = off_m < S
            mask_n = off_n < S
            valid = (off_m[:, None] >= off_n[None, :]) & mask_m[:, None] & mask_n[None, :]
            qk = tl.where(valid, qk, float('-inf'))
            p = tl.math.exp(qk - l[:, None])
            p = tl.where(valid, p, 0.0)
        else:
            p = tl.math.exp(qk - l[:, None])
            
        p_bf16 = p.to(q.dtype)
        dv += tl.dot(p_bf16.T, do, out_dtype=tl.float32)
        
        dp = tl.dot(do, v.T, out_dtype=tl.float32)
        ds = p * (dp - delta[:, None]) * scale
        
        ds_bf16 = ds.to(q.dtype)
        dk += tl.dot(ds_bf16.T, q, out_dtype=tl.float32)
        
    dk_desc.store([n_start, 0], dk.to(k.dtype))
    dv_desc.store([n_start, 0], dv.to(v.dtype))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes backward pass for Causal Multi-Head Attention, targeting Hopper TMA optimizations.
    Fully uncoupled parallel grids compute dQ independently of dK/dV omitting atomics.
    """
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    scale = 1.0 / math.sqrt(D)
    
    stride_lb, stride_lh, stride_ls = L.stride(0), L.stride(1), L.stride(2)
        
    grid_dq = lambda META: (
        triton.cdiv(S, META['BLOCK_M']),
        B * H
    )
    
    grid_dk = lambda META: (
        triton.cdiv(S, META['BLOCK_N']),
        B * H
    )

    _bwd_kernel_dq[grid_dq](
        Q, K, V, O, dO, dQ, L,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        stride_lb, stride_lh, stride_ls,
        H, S, scale,
        D_HEAD=D,
    )

    _bwd_kernel_dk_dv[grid_dk](
        Q, K, V, O, dO, dK, dV, L,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        stride_lb, stride_lh, stride_ls,
        H, S, scale,
        D_HEAD=D,
    )