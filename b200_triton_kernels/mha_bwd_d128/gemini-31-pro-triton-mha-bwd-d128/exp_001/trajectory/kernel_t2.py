import math
import torch
import triton
import triton.language as tl


@triton.jit
def _bwd_kernel_dq(
    Q, K, V, O, dO, L, dQ,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
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
    
    # Outer loop processes Queries in BLOCK_M
    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, BLOCK_N)
    off_d = tl.arange(0, d)
    
    q_ptrs = Q + batch * stride_qb + head * stride_qh + off_m[:, None] * stride_qs + off_d[None, :] * stride_qd
    o_ptrs = O + batch * stride_ob + head * stride_oh + off_m[:, None] * stride_os + off_d[None, :] * stride_od
    do_ptrs = dO + batch * stride_dob + head * stride_doh + off_m[:, None] * stride_dos + off_d[None, :] * stride_dod
    l_ptrs = L + batch * stride_lb + head * stride_lh + off_m * stride_ls
    
    mask_m = off_m < S
    
    # Load stationary Query and Output block variables
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
    do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
    l = tl.load(l_ptrs, mask=mask_m, other=0.0)
    
    # D_m = sum(O * dO, axis=1) computed on the fly
    d_val = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
    
    dq = tl.zeros([BLOCK_M, d], dtype=tl.float32)
    
    k_ptrs = K + batch * stride_kb + head * stride_kh + off_n[:, None] * stride_ks + off_d[None, :] * stride_kd
    v_ptrs = V + batch * stride_vb + head * stride_vh + off_n[:, None] * stride_vs + off_d[None, :] * stride_vd
    
    # Inner loop over Keys/Values
    for start_n in range(0, S, BLOCK_N):
        mask_n = (start_n + off_n) < S
        
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        
        # WGMMA natively reads k.T and v.T from shared memory cleanly
        s = tl.dot(q, k.T) * scale
        
        P = tl.exp(s - l[:, None])
        mask_combined = mask_m[:, None] & mask_n[None, :]
        P = tl.where(mask_combined, P, 0.0)
        
        dP = tl.dot(do, v.T)
        ds = P * (dP - d_val[:, None]) * scale
        
        dq = tl.dot(ds.to(q.dtype), k, acc=dq)
        
        k_ptrs += BLOCK_N * stride_ks
        v_ptrs += BLOCK_N * stride_vs
        
    dq_ptrs = dQ + batch * stride_dqb + head * stride_dqh + off_m[:, None] * stride_dqs + off_d[None, :] * stride_dqd
    tl.store(dq_ptrs, dq.to(dQ.dtype.element_ty), mask=mask_m[:, None])


@triton.jit
def _bwd_kernel_dk_dv(
    Q, K, V, O, dO, L, dK, dV,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    stride_lb, stride_lh, stride_ls,
    H, S, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, d: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    batch = pid_bh // H
    head = pid_bh % H
    
    # Outer loop processes Keys/Values in BLOCK_N
    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    off_m = tl.arange(0, BLOCK_M)
    off_d = tl.arange(0, d)
    
    k_ptrs = K + batch * stride_kb + head * stride_kh + off_n[:, None] * stride_ks + off_d[None, :] * stride_kd
    v_ptrs = V + batch * stride_vb + head * stride_vh + off_n[:, None] * stride_vs + off_d[None, :] * stride_vd
    
    mask_n = off_n < S
    
    # Load stationary Key and Value block variables
    k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
    
    dk = tl.zeros([BLOCK_N, d], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, d], dtype=tl.float32)
    
    q_ptrs = Q + batch * stride_qb + head * stride_qh + off_m[:, None] * stride_qs + off_d[None, :] * stride_qd
    o_ptrs = O + batch * stride_ob + head * stride_oh + off_m[:, None] * stride_os + off_d[None, :] * stride_od
    do_ptrs = dO + batch * stride_dob + head * stride_doh + off_m[:, None] * stride_dos + off_d[None, :] * stride_dod
    l_ptrs = L + batch * stride_lb + head * stride_lh + off_m * stride_ls
    
    # Inner loop over Queries
    for start_m in range(0, S, BLOCK_M):
        mask_m = (start_m + off_m) < S
        
        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
        o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
        do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
        l = tl.load(l_ptrs, mask=mask_m, other=0.0)
        
        d_val = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
        
        # We explicitly transpose the mathematical formulation to (BLOCK_N, BLOCK_M).
        # WGMMA instructions natively require `A` operands (k, v, P_T, ds_T) to be untransposed from registers
        # and flawlessly supports fetching transposed `B` operands (q.T, do.T) directly from SMEM!
        s_T = tl.dot(k, q.T) * scale
        
        P_T = tl.exp(s_T - l[None, :])
        mask_combined = mask_n[:, None] & mask_m[None, :]
        P_T = tl.where(mask_combined, P_T, 0.0)
        
        dv = tl.dot(P_T.to(do.dtype), do, acc=dv)
        
        dP_T = tl.dot(v, do.T)
        ds_T = P_T * (dP_T - d_val[None, :]) * scale
        
        dk = tl.dot(ds_T.to(q.dtype), q, acc=dk)
        
        q_ptrs += BLOCK_M * stride_qs
        o_ptrs += BLOCK_M * stride_os
        do_ptrs += BLOCK_M * stride_dos
        l_ptrs += BLOCK_M * stride_ls
        
    dk_ptrs = dK + batch * stride_dkb + head * stride_dkh + off_n[:, None] * stride_dks + off_d[None, :] * stride_dkd
    dv_ptrs = dV + batch * stride_dvb + head * stride_dvh + off_n[:, None] * stride_dvs + off_d[None, :] * stride_dvd
    
    tl.store(dk_ptrs, dk.to(dK.dtype.element_ty), mask=mask_n[:, None])
    tl.store(dv_ptrs, dv.to(dV.dtype.element_ty), mask=mask_n[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    with torch.cuda.device(Q.device):
        B, H, S, d = Q.shape
        scale = 1.0 / math.sqrt(d)
        
        # dQ accumulation pass
        grid_dq = (triton.cdiv(S, 128), B * H)
        _bwd_kernel_dq[grid_dq](
            Q, K, V, O, dO, L, dQ,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
            dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
            L.stride(0), L.stride(1), L.stride(2),
            H, S, scale,
            BLOCK_M=128, BLOCK_N=64, d=128,
            num_warps=8, num_stages=2
        )
        
        # dK / dV accumulation pass
        grid_dk_dv = (triton.cdiv(S, 64), B * H)
        _bwd_kernel_dk_dv[grid_dk_dv](
            Q, K, V, O, dO, L, dK, dV,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
            dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
            dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
            L.stride(0), L.stride(1), L.stride(2),
            H, S, scale,
            BLOCK_M=128, BLOCK_N=64, d=128,
            num_warps=8, num_stages=2
        )