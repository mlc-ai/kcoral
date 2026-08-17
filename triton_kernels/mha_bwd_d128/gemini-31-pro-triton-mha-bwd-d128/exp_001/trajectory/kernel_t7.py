import math
import torch
import triton
import triton.language as tl

# Set global infrastructure allocator required by `tl.make_tensor_descriptor`
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


@triton.jit
def _precompute_d(
    O, dO, dQ_as_D,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    H, S, d: tl.constexpr, BLOCK_M: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    batch = pid_bh // H
    head = pid_bh % H
    
    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = off_m < S
    off_d = tl.arange(0, d)
    
    o_ptrs = O + batch * stride_ob + head * stride_oh + off_m[:, None] * stride_os + off_d[None, :] * stride_od
    do_ptrs = dO + batch * stride_dob + head * stride_doh + off_m[:, None] * stride_dos + off_d[None, :] * stride_dod
    
    o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
    do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
    
    # Precompute running statistic efficiently
    d_val = tl.sum(o * do, axis=1)
    
    # Intentionally circumvent memory allocation bounds by safely bitcasting D_val straight into the footprint of dQ
    d_val_i32 = d_val.to(tl.int32, bitcast=True)
    bf16_0 = d_val_i32.to(tl.int16).to(tl.bfloat16, bitcast=True)
    bf16_1 = (d_val_i32 >> 16).to(tl.int16).to(tl.bfloat16, bitcast=True)
    
    dq_base = dQ_as_D + batch * stride_dqb + head * stride_dqh + off_m * stride_dqs
    tl.store(dq_base + 0 * stride_dqd, bf16_0, mask=mask_m)
    tl.store(dq_base + 1 * stride_dqd, bf16_1, mask=mask_m)


@triton.jit
def _bwd_kernel_dk_dv(
    Q, K, V, dO, L, dK, dV, dQ_as_D,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    stride_lb, stride_lh, stride_ls,
    H, S, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, d: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    batch = pid_bh // H
    head = pid_bh % H
    
    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    off_d = tl.arange(0, d)
    mask_n = off_n < S
    
    # 1. Evaluate stationary K and V safely tracking strictly inside registers
    k_ptrs = K + batch * stride_kb + head * stride_kh + off_n[:, None] * stride_ks + off_d[None, :] * stride_kd
    v_ptrs = V + batch * stride_vb + head * stride_vh + off_n[:, None] * stride_vs + off_d[None, :] * stride_vd
    
    k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
    
    dk = tl.zeros([BLOCK_N, d], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, d], dtype=tl.float32)
    
    # 2. Setup Asynchronous Memcheck-Safe TMA loads for Inner loop bounds 
    desc_q = tl.make_tensor_descriptor(
        Q + batch * stride_qb + head * stride_qh,
        shape=[S, d], strides=[stride_qs, 1],
        block_shape=[BLOCK_M, d], padding_option="zero"
    )
    desc_do = tl.make_tensor_descriptor(
        dO + batch * stride_dob + head * stride_doh,
        shape=[S, d], strides=[stride_dos, 1],
        block_shape=[BLOCK_M, d], padding_option="zero"
    )
    
    dq_base = dQ_as_D + batch * stride_dqb + head * stride_dqh
    l_base = L + batch * stride_lb + head * stride_lh
    
    for start_m in tl.range(0, S, BLOCK_M, num_stages=3):
        q = tl.load(desc_q, [start_m, 0])
        do = tl.load(desc_do, [start_m, 0])
        
        off_m = start_m + tl.arange(0, BLOCK_M)
        mask_m = off_m < S
        
        l = tl.load(l_base + off_m * stride_ls, mask=mask_m, other=0.0)
        
        # Reliably reconstruct stashed precalculated running-sum statistic WITHOUT reloading `O`
        bf16_0 = tl.load(dq_base + off_m * stride_dqs + 0 * stride_dqd, mask=mask_m, other=0.0)
        bf16_1 = tl.load(dq_base + off_m * stride_dqs + 1 * stride_dqd, mask=mask_m, other=0.0)
        
        i16_0 = bf16_0.to(tl.int16, bitcast=True)
        i16_1 = bf16_1.to(tl.int16, bitcast=True)
        i32 = (i16_0.to(tl.int32) & 0xFFFF) | ((i16_1.to(tl.int32) & 0xFFFF) << 16)
        d_val = i32.to(tl.float32, bitcast=True)
        
        # Explicit algorithmic Layout Transpose enforces Hopper WGMMA natively unpacking implicitly transposed `B` operands seamlessly out of shared memory
        s_T = tl.dot(k, q.T) * scale
        
        p_T = tl.exp(s_T - l[None, :])
        mask_combined = mask_n[:, None] & mask_m[None, :]
        p_T = tl.where(mask_combined, p_T, 0.0)
        
        dv = tl.dot(p_T.to(do.dtype), do, acc=dv)
        
        dp_T = tl.dot(v, do.T)
        ds_T = p_T * (dp_T - d_val[None, :]) * scale
        
        dk = tl.dot(ds_T.to(q.dtype), q, acc=dk)
        
    dk_ptrs = dK + batch * stride_dkb + head * stride_dkh + off_n[:, None] * stride_dks + off_d[None, :] * stride_dkd
    dv_ptrs = dV + batch * stride_dvb + head * stride_dvh + off_n[:, None] * stride_dvs + off_d[None, :] * stride_dvd
    
    tl.store(dk_ptrs, dk.to(dK.dtype.element_ty), mask=mask_n[:, None])
    tl.store(dv_ptrs, dv.to(dV.dtype.element_ty), mask=mask_n[:, None])


@triton.jit
def _bwd_kernel_dq(
    Q, K, V, dO, L, dQ,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    stride_lb, stride_lh, stride_ls,
    H, S, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, d: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    batch = pid_bh // H
    head = pid_bh % H
    
    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_d = tl.arange(0, d)
    mask_m = off_m < S
    
    # 1. Execute Queries securely within registers
    q_ptrs = Q + batch * stride_qb + head * stride_qh + off_m[:, None] * stride_qs + off_d[None, :] * stride_qd
    do_ptrs = dO + batch * stride_dob + head * stride_doh + off_m[:, None] * stride_dos + off_d[None, :] * stride_dod
    l_ptrs = L + batch * stride_lb + head * stride_lh + off_m * stride_ls
    
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
    l = tl.load(l_ptrs, mask=mask_m, other=0.0)
    
    dq_base = dQ + batch * stride_dqb + head * stride_dqh
    bf16_0 = tl.load(dq_base + off_m * stride_dqs + 0 * stride_dqd, mask=mask_m, other=0.0)
    bf16_1 = tl.load(dq_base + off_m * stride_dqs + 1 * stride_dqd, mask=mask_m, other=0.0)
    
    i16_0 = bf16_0.to(tl.int16, bitcast=True)
    i16_1 = bf16_1.to(tl.int16, bitcast=True)
    i32 = (i16_0.to(tl.int32) & 0xFFFF) | ((i16_1.to(tl.int32) & 0xFFFF) << 16)
    d_val = i32.to(tl.float32, bitcast=True)
    
    dq = tl.zeros([BLOCK_M, d], dtype=tl.float32)
    
    # 2. Setup Asynchronous Memcheck-Safe TMA loads for Inner loop boundaries
    desc_k = tl.make_tensor_descriptor(
        K + batch * stride_kb + head * stride_kh,
        shape=[S, d], strides=[stride_ks, 1],
        block_shape=[BLOCK_N, d], padding_option="zero"
    )
    desc_v = tl.make_tensor_descriptor(
        V + batch * stride_vb + head * stride_vh,
        shape=[S, d], strides=[stride_vs, 1],
        block_shape=[BLOCK_N, d], padding_option="zero"
    )
    
    for start_n in tl.range(0, S, BLOCK_N, num_stages=3):
        k = tl.load(desc_k, [start_n, 0])
        v = tl.load(desc_v, [start_n, 0])
        
        off_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = off_n < S
        
        s = tl.dot(q, k.T) * scale
        
        p = tl.exp(s - l[:, None])
        mask_combined = mask_m[:, None] & mask_n[None, :]
        p = tl.where(mask_combined, p, 0.0)
        
        dp = tl.dot(do, v.T)
        ds = p * (dp - d_val[:, None]) * scale
        
        dq = tl.dot(ds.to(q.dtype), k, acc=dq)
        
    # Implicitly erases securely the staging ground placed by `_precompute_d` preserving functional purity perfectly
    dq_ptrs = dQ + batch * stride_dqb + head * stride_dqh + off_m[:, None] * stride_dqs + off_d[None, :] * stride_dqd
    tl.store(dq_ptrs, dq.to(dQ.dtype.element_ty), mask=mask_m[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    with torch.cuda.device(Q.device):
        triton.set_allocator(alloc_fn)
        
        B, H, S, d = Q.shape
        scale = 1.0 / math.sqrt(d)
        
        grid_pre = (triton.cdiv(S, 128), B * H)
        _precompute_d[grid_pre](
            O, dO, dQ,
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
            dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
            H, S, d=128, BLOCK_M=128,
            num_warps=4, num_stages=2
        )
        
        grid_dk_dv = (triton.cdiv(S, 64), B * H)
        _bwd_kernel_dk_dv[grid_dk_dv](
            Q, K, V, dO, L, dK, dV, dQ,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
            dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
            dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
            dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
            L.stride(0), L.stride(1), L.stride(2),
            H, S, scale,
            BLOCK_M=128, BLOCK_N=64, d=128,
            num_warps=8, num_stages=3
        )
        
        grid_dq = (triton.cdiv(S, 128), B * H)
        _bwd_kernel_dq[grid_dq](
            Q, K, V, dO, L, dQ,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
            dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
            L.stride(0), L.stride(1), L.stride(2),
            H, S, scale,
            BLOCK_M=128, BLOCK_N=64, d=128,
            num_warps=8, num_stages=3
        )