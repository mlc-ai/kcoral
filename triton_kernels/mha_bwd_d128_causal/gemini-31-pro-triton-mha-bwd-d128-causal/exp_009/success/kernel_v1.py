import math
import torch
import triton
import triton.language as tl

# Install Triton's descriptor allocator for device-side TMA descriptors
def _alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(_alloc_fn)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=3, num_warps=4),
    ],
    key=['seq_len']
)
@triton.jit
def bwd_kernel_dq(
    Q, K, V, O, sm_scale, dO, dQ, L,
    seq_len,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    stride_lb, stride_lh, stride_ls,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D_HEAD: tl.constexpr
):
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_m = tl.program_id(0)

    start_m = pid_m * BLOCK_M
    
    # TMA descriptors for Q, dO, O
    q_ptr = Q + pid_b * stride_qb + pid_h * stride_qh
    q_desc = tl.make_tensor_descriptor(
        q_ptr, shape=[seq_len, D_HEAD], strides=[stride_qs, stride_qd],
        block_shape=[BLOCK_M, D_HEAD], padding_option="zero"
    )
    q = q_desc.load([start_m, 0])
    
    do_ptr = dO + pid_b * stride_dob + pid_h * stride_doh
    do_desc = tl.make_tensor_descriptor(
        do_ptr, shape=[seq_len, D_HEAD], strides=[stride_dos, stride_dod],
        block_shape=[BLOCK_M, D_HEAD], padding_option="zero"
    )
    do = do_desc.load([start_m, 0])
    
    o_ptr = O + pid_b * stride_ob + pid_h * stride_oh
    o_desc = tl.make_tensor_descriptor(
        o_ptr, shape=[seq_len, D_HEAD], strides=[stride_os, stride_od],
        block_shape=[BLOCK_M, D_HEAD], padding_option="zero"
    )
    o = o_desc.load([start_m, 0])
    
    # Standard pointer load for L
    offs_m = start_m + tl.arange(0, BLOCK_M)
    mask_m = offs_m < seq_len
    l_ptrs = L + pid_b * stride_lb + pid_h * stride_lh + offs_m * stride_ls
    l = tl.load(l_ptrs, mask=mask_m, other=0.0)
    
    # Precompute row sum of (O * dO) for the current Q block
    d_val = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
    
    # TMA descriptors for K and V
    k_ptr = K + pid_b * stride_kb + pid_h * stride_kh
    k_desc = tl.make_tensor_descriptor(
        k_ptr, shape=[seq_len, D_HEAD], strides=[stride_ks, stride_kd],
        block_shape=[BLOCK_N, D_HEAD], padding_option="zero"
    )
    
    v_ptr = V + pid_b * stride_vb + pid_h * stride_vh
    v_desc = tl.make_tensor_descriptor(
        v_ptr, shape=[seq_len, D_HEAD], strides=[stride_vs, stride_vd],
        block_shape=[BLOCK_N, D_HEAD], padding_option="zero"
    )
    
    dq = tl.zeros([BLOCK_M, D_HEAD], dtype=tl.float32)
    
    max_n = tl.minimum(start_m + BLOCK_M, seq_len)
    num_steps = tl.cdiv(max_n, BLOCK_N)
    
    offs_n_base = tl.arange(0, BLOCK_N)
    
    for step in range(num_steps):
        start_n = step * BLOCK_N
        offs_n = start_n + offs_n_base
        
        k = k_desc.load([start_n, 0])
        v = v_desc.load([start_n, 0])
        
        qk = tl.dot(q, tl.trans(k))
        qk = qk * sm_scale
        
        is_causal_step = (start_n + BLOCK_N > start_m)
        is_boundary_step = (start_n + BLOCK_N > seq_len) or (start_m + BLOCK_M > seq_len)
        
        if is_causal_step or is_boundary_step:
            mask_n = offs_n < seq_len
            mask = mask_m[:, None] & mask_n[None, :]
            if is_causal_step:
                mask = mask & (offs_m[:, None] >= offs_n[None, :])
            qk = tl.where(mask, qk, float("-inf"))
            p = tl.exp(qk - l[:, None])
            p = tl.where(mask, p, 0.0)
        else:
            p = tl.exp(qk - l[:, None])
            
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - d_val[:, None]) * sm_scale
        ds_cast = tl.cast(ds, q.dtype)
        
        dq = tl.dot(ds_cast, k, acc=dq)
        
    dq_ptr = dQ + pid_b * stride_dqb + pid_h * stride_dqh
    dq_desc = tl.make_tensor_descriptor(
        dq_ptr, shape=[seq_len, D_HEAD], strides=[stride_dqs, stride_dqd],
        block_shape=[BLOCK_M, D_HEAD]
    )
    dq_desc.store([start_m, 0], tl.cast(dq, dQ.dtype.element_ty))


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=3, num_warps=4),
    ],
    key=['seq_len']
)
@triton.jit
def bwd_kernel_dk_dv(
    Q, K, V, O, sm_scale, dO, dK, dV, L,
    seq_len,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    stride_lb, stride_lh, stride_ls,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D_HEAD: tl.constexpr
):
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_n = tl.program_id(0)

    start_n = pid_n * BLOCK_N
    
    # TMA descriptors for K and V
    k_ptr = K + pid_b * stride_kb + pid_h * stride_kh
    k_desc = tl.make_tensor_descriptor(
        k_ptr, shape=[seq_len, D_HEAD], strides=[stride_ks, stride_kd],
        block_shape=[BLOCK_N, D_HEAD], padding_option="zero"
    )
    k = k_desc.load([start_n, 0])
    
    v_ptr = V + pid_b * stride_vb + pid_h * stride_vh
    v_desc = tl.make_tensor_descriptor(
        v_ptr, shape=[seq_len, D_HEAD], strides=[stride_vs, stride_vd],
        block_shape=[BLOCK_N, D_HEAD], padding_option="zero"
    )
    v = v_desc.load([start_n, 0])
    
    # TMA descriptors for Q, dO, O
    q_ptr = Q + pid_b * stride_qb + pid_h * stride_qh
    q_desc = tl.make_tensor_descriptor(
        q_ptr, shape=[seq_len, D_HEAD], strides=[stride_qs, stride_qd],
        block_shape=[BLOCK_M, D_HEAD], padding_option="zero"
    )
    
    do_ptr = dO + pid_b * stride_dob + pid_h * stride_doh
    do_desc = tl.make_tensor_descriptor(
        do_ptr, shape=[seq_len, D_HEAD], strides=[stride_dos, stride_dod],
        block_shape=[BLOCK_M, D_HEAD], padding_option="zero"
    )
    
    o_ptr = O + pid_b * stride_ob + pid_h * stride_oh
    o_desc = tl.make_tensor_descriptor(
        o_ptr, shape=[seq_len, D_HEAD], strides=[stride_os, stride_od],
        block_shape=[BLOCK_M, D_HEAD], padding_option="zero"
    )
    
    dk = tl.zeros([BLOCK_N, D_HEAD], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, D_HEAD], dtype=tl.float32)
    
    # Identify the first Q block that needs to attend to this K block (causal logic)
    start_m_initial = (start_n // BLOCK_M) * BLOCK_M
    num_steps = tl.cdiv(seq_len - start_m_initial, BLOCK_M)
    
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_m_base = tl.arange(0, BLOCK_M)
    
    l_ptr_base = L + pid_b * stride_lb + pid_h * stride_lh
    
    for step in range(num_steps):
        start_m = start_m_initial + step * BLOCK_M
        offs_m = start_m + offs_m_base
        mask_m = offs_m < seq_len
        
        q = q_desc.load([start_m, 0])
        do = do_desc.load([start_m, 0])
        o = o_desc.load([start_m, 0])
        
        l_ptrs = l_ptr_base + offs_m * stride_ls
        l = tl.load(l_ptrs, mask=mask_m, other=0.0)
        
        d_val = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
        
        qk = tl.dot(q, tl.trans(k))
        qk = qk * sm_scale
        
        is_causal_step = (start_m < start_n + BLOCK_N)
        is_boundary_step = (start_m + BLOCK_M > seq_len) or (start_n + BLOCK_N > seq_len)
        
        if is_causal_step or is_boundary_step:
            mask_n = offs_n < seq_len
            mask = mask_m[:, None] & mask_n[None, :]
            if is_causal_step:
                mask = mask & (offs_m[:, None] >= offs_n[None, :])
            qk = tl.where(mask, qk, float("-inf"))
            p = tl.exp(qk - l[:, None])
            p = tl.where(mask, p, 0.0)
        else:
            p = tl.exp(qk - l[:, None])
            
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - d_val[:, None]) * sm_scale
        
        ds_cast = tl.cast(ds, q.dtype)
        p_cast = tl.cast(p, q.dtype)
        
        dv = tl.dot(tl.trans(p_cast), do, acc=dv)
        dk = tl.dot(tl.trans(ds_cast), q, acc=dk)
        
    dk_ptr = dK + pid_b * stride_dkb + pid_h * stride_dkh
    dk_desc = tl.make_tensor_descriptor(
        dk_ptr, shape=[seq_len, D_HEAD], strides=[stride_dks, stride_dkd],
        block_shape=[BLOCK_N, D_HEAD]
    )
    dk_desc.store([start_n, 0], tl.cast(dk, dK.dtype.element_ty))
    
    dv_ptr = dV + pid_b * stride_dvb + pid_h * stride_dvh
    dv_desc = tl.make_tensor_descriptor(
        dv_ptr, shape=[seq_len, D_HEAD], strides=[stride_dvs, stride_dvd],
        block_shape=[BLOCK_N, D_HEAD]
    )
    dv_desc.store([start_n, 0], tl.cast(dv, dV.dtype.element_ty))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes causal Multi-Head Attention backward pass using standard Triton semantics.
    Results are written inplace into preallocated destination tensors: `dQ`, `dK`, `dV`.
    Optimized for NVIDIA Hopper TMA / WGMMA paths.
    """
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    sm_scale = 1.0 / math.sqrt(d)
    
    grid_dq = lambda META: (
        triton.cdiv(S, META['BLOCK_M']),
        B,
        H
    )
    
    bwd_kernel_dq[grid_dq](
        Q, K, V, O, sm_scale, dO, dQ, L,
        S,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        D_HEAD=d
    )

    grid_dkdv = lambda META: (
        triton.cdiv(S, META['BLOCK_N']),
        B,
        H
    )
    
    bwd_kernel_dk_dv[grid_dkdv](
        Q, K, V, O, sm_scale, dO, dK, dV, L,
        S,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        D_HEAD=d
    )