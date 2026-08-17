import math
import torch
import triton
import triton.language as tl

@triton.jit
def _precompute_d(
    O, dO, dQ_as_D,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    S, d: tl.constexpr, BLOCK_M: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = off_m < S
    
    off_d = tl.arange(0, d)
    
    o_ptrs = O + pid_bh * stride_oh + off_m[:, None] * stride_os + off_d[None, :] * stride_od
    do_ptrs = dO + pid_bh * stride_doh + off_m[:, None] * stride_dos + off_d[None, :] * stride_dod
    
    o = tl.load(o_ptrs, mask=mask[:, None], other=0.0).to(tl.float32)
    do = tl.load(do_ptrs, mask=mask[:, None], other=0.0).to(tl.float32)
    
    # Compute the D statistic per row
    d_val = tl.sum(o * do, axis=1)
    
    # Safely bitcast FP32 -> two INT16 values -> two BFLOAT16 values (avoids external allocations)
    d_val_i32 = d_val.to(tl.int32, bitcast=True)
    d_val_i16_0 = (d_val_i32 & 0xFFFF).to(tl.int16)
    d_val_i16_1 = ((d_val_i32 >> 16) & 0xFFFF).to(tl.int16)
    
    dq_d0_ptrs = dQ_as_D + pid_bh * stride_dqh + off_m * stride_dqs + 0 * stride_dqd
    dq_d1_ptrs = dQ_as_D + pid_bh * stride_dqh + off_m * stride_dqs + 1 * stride_dqd
    
    tl.store(dq_d0_ptrs, d_val_i16_0.to(tl.bfloat16, bitcast=True), mask=mask)
    tl.store(dq_d1_ptrs, d_val_i16_1.to(tl.bfloat16, bitcast=True), mask=mask)


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
    off_m = tl.arange(0, BLOCK_M)
    off_d = tl.arange(0, d)
    
    k_ptrs = K + batch * stride_kb + head * stride_kh + off_n[:, None] * stride_ks + off_d[None, :] * stride_kd
    v_ptrs = V + batch * stride_vb + head * stride_vh + off_n[:, None] * stride_vs + off_d[None, :] * stride_vd
    
    mask_n = off_n < S
    
    k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
    
    dk = tl.zeros([BLOCK_N, d], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, d], dtype=tl.float32)
    
    q_ptrs = Q + batch * stride_qb + head * stride_qh + off_m[:, None] * stride_qs + off_d[None, :] * stride_qd
    do_ptrs = dO + batch * stride_dob + head * stride_doh + off_m[:, None] * stride_dos + off_d[None, :] * stride_dod
    l_ptrs = L + batch * stride_lb + head * stride_lh + off_m * stride_ls
    
    dq_d0_ptrs = dQ_as_D + batch * stride_dqb + head * stride_dqh + off_m * stride_dqs + 0 * stride_dqd
    dq_d1_ptrs = dQ_as_D + batch * stride_dqb + head * stride_dqh + off_m * stride_dqs + 1 * stride_dqd
    
    for start_m in range(0, S, BLOCK_M):
        mask_m = (start_m + off_m) < S
        
        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
        do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
        l = tl.load(l_ptrs, mask=mask_m, other=0.0)
        
        # Load and unpack safely stored D metric back to pure FP32
        bf16_0 = tl.load(dq_d0_ptrs, mask=mask_m, other=0.0)
        bf16_1 = tl.load(dq_d1_ptrs, mask=mask_m, other=0.0)
        i16_0 = bf16_0.to(tl.int16, bitcast=True)
        i16_1 = bf16_1.to(tl.int16, bitcast=True)
        i32_0 = i16_0.to(tl.int32) & 0xFFFF
        i32_1 = i16_1.to(tl.int32) & 0xFFFF
        i32 = i32_0 | (i32_1 << 16)
        d_val = i32.to(tl.float32, bitcast=True)
        
        s = tl.dot(q, k.T) * scale
        p = tl.exp(s - l[:, None])
        p = tl.where(mask_m[:, None], p, 0.0)
        
        p_bf16 = p.to(do.dtype)
        dv = tl.dot(p_bf16.T, do, acc=dv)
        
        dp = tl.dot(do, v.T)
        ds = p * (dp - d_val[:, None]) * scale
        
        ds_bf16 = ds.to(q.dtype)
        dk = tl.dot(ds_bf16.T, q, acc=dk)
        
        q_ptrs += BLOCK_M * stride_qs
        do_ptrs += BLOCK_M * stride_dos
        l_ptrs += BLOCK_M * stride_ls
        dq_d0_ptrs += BLOCK_M * stride_dqs
        dq_d1_ptrs += BLOCK_M * stride_dqs
        
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
    off_n = tl.arange(0, BLOCK_N)
    off_d = tl.arange(0, d)
    
    mask_m = off_m < S
    
    dq_d0_ptrs = dQ + batch * stride_dqb + head * stride_dqh + off_m * stride_dqs + 0 * stride_dqd
    dq_d1_ptrs = dQ + batch * stride_dqb + head * stride_dqh + off_m * stride_dqs + 1 * stride_dqd
    
    bf16_0 = tl.load(dq_d0_ptrs, mask=mask_m, other=0.0)
    bf16_1 = tl.load(dq_d1_ptrs, mask=mask_m, other=0.0)
    
    i16_0 = bf16_0.to(tl.int16, bitcast=True)
    i16_1 = bf16_1.to(tl.int16, bitcast=True)
    i32_0 = i16_0.to(tl.int32) & 0xFFFF
    i32_1 = i16_1.to(tl.int32) & 0xFFFF
    i32 = i32_0 | (i32_1 << 16)
    d_val = i32.to(tl.float32, bitcast=True)
    
    q_ptrs = Q + batch * stride_qb + head * stride_qh + off_m[:, None] * stride_qs + off_d[None, :] * stride_qd
    do_ptrs = dO + batch * stride_dob + head * stride_doh + off_m[:, None] * stride_dos + off_d[None, :] * stride_dod
    l_ptrs = L + batch * stride_lb + head * stride_lh + off_m * stride_ls
    
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
    l = tl.load(l_ptrs, mask=mask_m, other=0.0)
    
    dq = tl.zeros([BLOCK_M, d], dtype=tl.float32)
    
    k_ptrs = K + batch * stride_kb + head * stride_kh + off_n[:, None] * stride_ks + off_d[None, :] * stride_kd
    v_ptrs = V + batch * stride_vb + head * stride_vh + off_n[:, None] * stride_vs + off_d[None, :] * stride_vd
    
    for start_n in range(0, S, BLOCK_N):
        mask_n = (start_n + off_n) < S
        
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        
        s = tl.dot(q, k.T) * scale
        p = tl.exp(s - l[:, None])
        p = tl.where(mask_n[None, :], p, 0.0)
        
        dp = tl.dot(do, v.T)
        ds = p * (dp - d_val[:, None]) * scale
        
        ds_bf16 = ds.to(q.dtype)
        dq = tl.dot(ds_bf16, k, acc=dq)
        
        k_ptrs += BLOCK_N * stride_ks
        v_ptrs += BLOCK_N * stride_vs
        
    dq_ptrs = dQ + batch * stride_dqb + head * stride_dqh + off_m[:, None] * stride_dqs + off_d[None, :] * stride_dqd
    tl.store(dq_ptrs, dq.to(dQ.dtype.element_ty), mask=mask_m[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute backward multi-head attention gradients cleanly."""
    with torch.cuda.device(Q.device):
        B, H, S, d = Q.shape
        scale = 1.0 / math.sqrt(d)
        
        # 1. Stash Pre-computed 'D' statistic dynamically in `dQ` tensor buffers 
        grid_pre = (triton.cdiv(S, 128), B * H)
        _precompute_d[grid_pre](
            O, dO, dQ,
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
            dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
            S, d=128, BLOCK_M=128,
            num_warps=4, num_stages=2
        )
        
        # 2. Key/Value Pass 
        grid_dk_dv = (triton.cdiv(S, 128), B * H)
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
            BLOCK_M=128, BLOCK_N=128, d=128,
            num_warps=8, num_stages=2
        )
        
        # 3. Query pass implicitly clears scratchpad stored at front columns of `dQ` via standard overwrite
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
            BLOCK_M=128, BLOCK_N=128, d=128,
            num_warps=8, num_stages=2
        )