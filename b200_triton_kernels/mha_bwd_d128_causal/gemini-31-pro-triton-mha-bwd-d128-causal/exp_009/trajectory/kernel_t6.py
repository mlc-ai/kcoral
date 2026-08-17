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
        triton.Config({'BLOCK': 128}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK': 64}, num_stages=3, num_warps=4),
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
    BLOCK: tl.constexpr, D_HEAD: tl.constexpr
):
    pid_m = tl.program_id(0)
    start_m = pid_m * BLOCK
    
    if start_m >= seq_len:
        return
        
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    # TMA descriptions inherently pad out-of-bounds sequences with zeros
    q_ptr = Q + pid_b * stride_qb + pid_h * stride_qh
    q_desc = tl.make_tensor_descriptor(
        q_ptr, shape=[seq_len, D_HEAD], strides=[stride_qs, stride_qd],
        block_shape=[BLOCK, D_HEAD], padding_option="zero"
    )
    q = q_desc.load([start_m, 0])
    
    do_ptr = dO + pid_b * stride_dob + pid_h * stride_doh
    do_desc = tl.make_tensor_descriptor(
        do_ptr, shape=[seq_len, D_HEAD], strides=[stride_dos, stride_dod],
        block_shape=[BLOCK, D_HEAD], padding_option="zero"
    )
    do = do_desc.load([start_m, 0])
    
    o_ptr = O + pid_b * stride_ob + pid_h * stride_oh
    o_desc = tl.make_tensor_descriptor(
        o_ptr, shape=[seq_len, D_HEAD], strides=[stride_os, stride_od],
        block_shape=[BLOCK, D_HEAD], padding_option="zero"
    )
    o = o_desc.load([start_m, 0])
    
    offs_m = start_m + tl.arange(0, BLOCK)
    mask_m = offs_m < seq_len
    l_ptrs = L + pid_b * stride_lb + pid_h * stride_lh + offs_m * stride_ls
    l = tl.load(l_ptrs, mask=mask_m, other=0.0)
    
    # Precompute local row sum of (O * dO) for the current Q block
    d_val = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
    
    k_ptr = K + pid_b * stride_kb + pid_h * stride_kh
    k_desc = tl.make_tensor_descriptor(
        k_ptr, shape=[seq_len, D_HEAD], strides=[stride_ks, stride_kd],
        block_shape=[BLOCK, D_HEAD], padding_option="zero"
    )
    
    v_ptr = V + pid_b * stride_vb + pid_h * stride_vh
    v_desc = tl.make_tensor_descriptor(
        v_ptr, shape=[seq_len, D_HEAD], strides=[stride_vs, stride_vd],
        block_shape=[BLOCK, D_HEAD], padding_option="zero"
    )
    
    dq = tl.zeros([BLOCK, D_HEAD], dtype=tl.float32)
    
    is_m_fully_inside = start_m + BLOCK <= seq_len
    
    # 1. Unmasked iterations safely outside causal footprint
    if is_m_fully_inside:
        for step in range(pid_m):
            start_n = step * BLOCK
            k = k_desc.load([start_n, 0])
            v = v_desc.load([start_n, 0])
            
            acc_qk = tl.zeros([BLOCK, BLOCK], dtype=tl.float32)
            qk = tl.dot(q, k.T, acc=acc_qk)
            
            acc_dp = tl.zeros([BLOCK, BLOCK], dtype=tl.float32)
            dp = tl.dot(do, v.T, acc=acc_dp)
            
            qk = qk * sm_scale
            p = tl.exp(qk - l[:, None])
            
            ds = p * (dp - d_val[:, None]) * sm_scale
            ds_cast = tl.cast(ds, q.dtype)
            
            dq = tl.dot(ds_cast, k, acc=dq)
    else:
        for step in range(pid_m):
            start_n = step * BLOCK
            k = k_desc.load([start_n, 0])
            v = v_desc.load([start_n, 0])
            
            acc_qk = tl.zeros([BLOCK, BLOCK], dtype=tl.float32)
            qk = tl.dot(q, k.T, acc=acc_qk)
            
            acc_dp = tl.zeros([BLOCK, BLOCK], dtype=tl.float32)
            dp = tl.dot(do, v.T, acc=acc_dp)
            
            qk = qk * sm_scale
            qk = tl.where(mask_m[:, None], qk, float("-inf"))
            p = tl.exp(qk - l[:, None])
            p = tl.where(mask_m[:, None], p, 0.0)
            
            ds = p * (dp - d_val[:, None]) * sm_scale
            ds_cast = tl.cast(ds, q.dtype)
            
            dq = tl.dot(ds_cast, k, acc=dq)
            
    # 2. Causal masked iteration strictly bounded to diagonal step
    start_n = pid_m * BLOCK
    if start_n < seq_len:
        k = k_desc.load([start_n, 0])
        v = v_desc.load([start_n, 0])
        
        acc_qk = tl.zeros([BLOCK, BLOCK], dtype=tl.float32)
        qk = tl.dot(q, k.T, acc=acc_qk)
        
        acc_dp = tl.zeros([BLOCK, BLOCK], dtype=tl.float32)
        dp = tl.dot(do, v.T, acc=acc_dp)
        
        qk = qk * sm_scale
        
        offs_n = start_n + tl.arange(0, BLOCK)
        mask_causal = (offs_m[:, None] >= offs_n[None, :])
        if not is_m_fully_inside:
            mask_causal = mask_causal & mask_m[:, None]
            
        qk = tl.where(mask_causal, qk, float("-inf"))
        p = tl.exp(qk - l[:, None])
        p = tl.where(mask_causal, p, 0.0)
        
        ds = p * (dp - d_val[:, None]) * sm_scale
        ds_cast = tl.cast(ds, q.dtype)
        
        dq = tl.dot(ds_cast, k, acc=dq)
        
    # Using standard stores avoids out-of-bounds issues robustly for boundary values
    offs_d = tl.arange(0, D_HEAD)
    dq_ptrs = dQ + pid_b * stride_dqb + pid_h * stride_dqh + offs_m[:, None] * stride_dqs + offs_d[None, :] * stride_dqd
    tl.store(dq_ptrs, tl.cast(dq, dQ.dtype.element_ty), mask=mask_m[:, None])


@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 128}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK': 64}, num_stages=3, num_warps=4),
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
    BLOCK: tl.constexpr, D_HEAD: tl.constexpr
):
    pid_n = tl.program_id(0)
    start_n = pid_n * BLOCK
    
    if start_n >= seq_len:
        return
        
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    k_ptr = K + pid_b * stride_kb + pid_h * stride_kh
    k_desc = tl.make_tensor_descriptor(
        k_ptr, shape=[seq_len, D_HEAD], strides=[stride_ks, stride_kd],
        block_shape=[BLOCK, D_HEAD], padding_option="zero"
    )
    k = k_desc.load([start_n, 0])
    
    v_ptr = V + pid_b * stride_vb + pid_h * stride_vh
    v_desc = tl.make_tensor_descriptor(
        v_ptr, shape=[seq_len, D_HEAD], strides=[stride_vs, stride_vd],
        block_shape=[BLOCK, D_HEAD], padding_option="zero"
    )
    v = v_desc.load([start_n, 0])
    
    q_ptr = Q + pid_b * stride_qb + pid_h * stride_qh
    q_desc = tl.make_tensor_descriptor(
        q_ptr, shape=[seq_len, D_HEAD], strides=[stride_qs, stride_qd],
        block_shape=[BLOCK, D_HEAD], padding_option="zero"
    )
    
    do_ptr = dO + pid_b * stride_dob + pid_h * stride_doh
    do_desc = tl.make_tensor_descriptor(
        do_ptr, shape=[seq_len, D_HEAD], strides=[stride_dos, stride_dod],
        block_shape=[BLOCK, D_HEAD], padding_option="zero"
    )
    
    o_ptr = O + pid_b * stride_ob + pid_h * stride_oh
    o_desc = tl.make_tensor_descriptor(
        o_ptr, shape=[seq_len, D_HEAD], strides=[stride_os, stride_od],
        block_shape=[BLOCK, D_HEAD], padding_option="zero"
    )
    
    l_ptr_base = L + pid_b * stride_lb + pid_h * stride_lh
    
    dk = tl.zeros([BLOCK, D_HEAD], dtype=tl.float32)
    dv = tl.zeros([BLOCK, D_HEAD], dtype=tl.float32)
    
    offs_n = start_n + tl.arange(0, BLOCK)
    
    # 1. Causal localized interactions strictly intersecting diagonal sequence bounds
    start_m = pid_n * BLOCK
    
    q = q_desc.load([start_m, 0])
    do = do_desc.load([start_m, 0])
    o = o_desc.load([start_m, 0])
    
    offs_m = start_m + tl.arange(0, BLOCK)
    mask_m = offs_m < seq_len
    l = tl.load(l_ptr_base + offs_m * stride_ls, mask=mask_m, other=0.0)
    
    d_val = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
    
    acc_qk = tl.zeros([BLOCK, BLOCK], dtype=tl.float32)
    qk = tl.dot(q, k.T, acc=acc_qk)
    
    acc_dp = tl.zeros([BLOCK, BLOCK], dtype=tl.float32)
    dp = tl.dot(do, v.T, acc=acc_dp)
    
    qk = qk * sm_scale
    
    mask_causal = (offs_m[:, None] >= offs_n[None, :]) & mask_m[:, None]
        
    qk = tl.where(mask_causal, qk, float("-inf"))
    p = tl.exp(qk - l[:, None])
    p = tl.where(mask_causal, p, 0.0)
    
    ds = p * (dp - d_val[:, None]) * sm_scale
    ds_cast = tl.cast(ds, q.dtype)
    p_cast = tl.cast(p, q.dtype)
    
    # Standard Register-Shared layout paths execute fully transparently with symmetric BLOCK shapes
    dv = tl.dot(p_cast.T, do, acc=dv)
    dk = tl.dot(ds_cast.T, q, acc=dk)
        
    # 2. Fully optimized unmasked core segment 
    total_q_blocks = tl.cdiv(seq_len, BLOCK)
    
    for step in range(pid_n + 1, total_q_blocks - 1):
        start_m = step * BLOCK
        
        q = q_desc.load([start_m, 0])
        do = do_desc.load([start_m, 0])
        o = o_desc.load([start_m, 0])
        
        l = tl.load(l_ptr_base + (start_m + tl.arange(0, BLOCK)) * stride_ls)
        d_val = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
        
        acc_qk = tl.zeros([BLOCK, BLOCK], dtype=tl.float32)
        qk = tl.dot(q, k.T, acc=acc_qk)
        
        acc_dp = tl.zeros([BLOCK, BLOCK], dtype=tl.float32)
        dp = tl.dot(do, v.T, acc=acc_dp)
        
        qk = qk * sm_scale
        p = tl.exp(qk - l[:, None])
        
        ds = p * (dp - d_val[:, None]) * sm_scale
        ds_cast = tl.cast(ds, q.dtype)
        p_cast = tl.cast(p, q.dtype)
        
        dv = tl.dot(p_cast.T, do, acc=dv)
        dk = tl.dot(ds_cast.T, q, acc=dk)
        
    # Handling final chunk cleanly mapping boundaries strictly when misaligned
    if pid_n + 1 < total_q_blocks:
        step = total_q_blocks - 1
        start_m = step * BLOCK
        is_m_fully_inside = start_m + BLOCK <= seq_len
        
        q = q_desc.load([start_m, 0])
        do = do_desc.load([start_m, 0])
        o = o_desc.load([start_m, 0])
        
        if is_m_fully_inside:
            l = tl.load(l_ptr_base + (start_m + tl.arange(0, BLOCK)) * stride_ls)
            d_val = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
            
            acc_qk = tl.zeros([BLOCK, BLOCK], dtype=tl.float32)
            qk = tl.dot(q, k.T, acc=acc_qk)
            
            acc_dp = tl.zeros([BLOCK, BLOCK], dtype=tl.float32)
            dp = tl.dot(do, v.T, acc=acc_dp)
            
            qk = qk * sm_scale
            p = tl.exp(qk - l[:, None])
            
            ds = p * (dp - d_val[:, None]) * sm_scale
            ds_cast = tl.cast(ds, q.dtype)
            p_cast = tl.cast(p, q.dtype)
            
            dv = tl.dot(p_cast.T, do, acc=dv)
            dk = tl.dot(ds_cast.T, q, acc=dk)
        else:
            offs_m_final = start_m + tl.arange(0, BLOCK)
            mask_m_final = offs_m_final < seq_len
            l = tl.load(l_ptr_base + offs_m_final * stride_ls, mask=mask_m_final, other=0.0)
            d_val = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
            
            acc_qk = tl.zeros([BLOCK, BLOCK], dtype=tl.float32)
            qk = tl.dot(q, k.T, acc=acc_qk)
            
            acc_dp = tl.zeros([BLOCK, BLOCK], dtype=tl.float32)
            dp = tl.dot(do, v.T, acc=acc_dp)
            
            qk = qk * sm_scale
            qk = tl.where(mask_m_final[:, None], qk, float("-inf"))
            p = tl.exp(qk - l[:, None])
            p = tl.where(mask_m_final[:, None], p, 0.0)
            
            ds = p * (dp - d_val[:, None]) * sm_scale
            ds_cast = tl.cast(ds, q.dtype)
            p_cast = tl.cast(p, q.dtype)
            
            dv = tl.dot(p_cast.T, do, acc=dv)
            dk = tl.dot(ds_cast.T, q, acc=dk)
        
    offs_d = tl.arange(0, D_HEAD)
    mask_n = offs_n < seq_len
    
    dk_ptrs = dK + pid_b * stride_dkb + pid_h * stride_dkh + offs_n[:, None] * stride_dks + offs_d[None, :] * stride_dkd
    tl.store(dk_ptrs, tl.cast(dk, dK.dtype.element_ty), mask=mask_n[:, None])
    
    dv_ptrs = dV + pid_b * stride_dvb + pid_h * stride_dvh + offs_n[:, None] * stride_dvs + offs_d[None, :] * stride_dvd
    tl.store(dv_ptrs, tl.cast(dv, dV.dtype.element_ty), mask=mask_n[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes causal Multi-Head Attention backward pass maximizing NVIDIA Hopper capabilities securely.
    Results are seamlessly injected directly inplace into standard destination tensors: `dQ`, `dK`, `dV`.
    """
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    sm_scale = 1.0 / math.sqrt(d)
    
    L_sq = L.squeeze(-1) if L.dim() == 4 else L
    
    grid_dq = lambda META: (
        triton.cdiv(S, META['BLOCK']),
        B,
        H
    )
    
    bwd_kernel_dq[grid_dq](
        Q, K, V, O, sm_scale, dO, dQ, L_sq,
        S,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        L_sq.stride(0), L_sq.stride(1), L_sq.stride(2),
        D_HEAD=d
    )

    grid_dkdv = lambda META: (
        triton.cdiv(S, META['BLOCK']),
        B,
        H
    )
    
    bwd_kernel_dk_dv[grid_dkdv](
        Q, K, V, O, sm_scale, dO, dK, dV, L_sq,
        S,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        L_sq.stride(0), L_sq.stride(1), L_sq.stride(2),
        D_HEAD=d
    )