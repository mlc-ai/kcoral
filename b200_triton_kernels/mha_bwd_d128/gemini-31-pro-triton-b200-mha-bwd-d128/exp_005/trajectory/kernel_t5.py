import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({
            'BM_DQ': 128, 'BN_DQ': 128,
            'BM_DK': 128, 'BN_DK': 128,
            'LOOP_STAGES': 2
        }, num_warps=8, num_stages=2),
        triton.Config({
            'BM_DQ': 128, 'BN_DQ': 64,
            'BM_DK': 64, 'BN_DK': 128,
            'LOOP_STAGES': 3
        }, num_warps=4, num_stages=3),
        triton.Config({
            'BM_DQ': 64, 'BN_DQ': 128,
            'BM_DK': 128, 'BN_DK': 64,
            'LOOP_STAGES': 3
        }, num_warps=4, num_stages=3),
        triton.Config({
            'BM_DQ': 64, 'BN_DQ': 64,
            'BM_DK': 64, 'BN_DK': 64,
            'LOOP_STAGES': 4
        }, num_warps=4, num_stages=4),
        triton.Config({
            'BM_DQ': 128, 'BN_DQ': 128,
            'BM_DK': 128, 'BN_DK': 128,
            'LOOP_STAGES': 3
        }, num_warps=8, num_stages=3),
    ],
    key=['S']
)
@triton.jit
def bwd_kernel(
    Q, K, V, O, dO, L,
    dQ, dK, dV,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    B, H, S, scale,
    d: tl.constexpr,
    BM_DQ: tl.constexpr, BN_DQ: tl.constexpr,
    BM_DK: tl.constexpr, BN_DK: tl.constexpr,
    LOOP_STAGES: tl.constexpr
):
    pid = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    num_kv_tiles = tl.cdiv(S, BN_DK)
    
    LOG2_E = 1.4426950408889634
    offs_d = tl.arange(0, d)
    
    if pid < num_kv_tiles:
        # =====================================================================
        # REGION 2: Exclusive owner for a dK and dV tile
        # =====================================================================
        pid_n = pid
        
        offs_n = pid_n * BN_DK + tl.arange(0, BN_DK)
        mask_n = offs_n < S
        
        k_ptrs = K + b * stride_kb + h * stride_kh + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        v_ptrs = V + b * stride_vb + h * stride_vh + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
        
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        
        # Pre-scale K to absorb softmax scale and log2(e) for the loop
        k_log2e = (k * (scale * LOG2_E)).to(k.dtype)
        
        dk_acc = tl.zeros((BN_DK, d), dtype=tl.float32)
        dv_acc = tl.zeros((BN_DK, d), dtype=tl.float32)
        
        offs_m = tl.arange(0, BM_DK)
        q_ptrs = Q + b * stride_qb + h * stride_qh + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
        o_ptrs = O + b * stride_ob + h * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
        do_ptrs = dO + b * stride_dob + h * stride_doh + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod
        l_ptrs = L + b * stride_lb + h * stride_lh + offs_m * stride_ls
        
        loop_q_tiles = tl.cdiv(S, BM_DK)
        for q_tile in tl.range(0, loop_q_tiles, num_stages=LOOP_STAGES):
            q_offs_m = q_tile * BM_DK + offs_m
            mask_m = q_offs_m < S
            
            q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
            o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
            do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
            l = tl.load(l_ptrs, mask=mask_m, other=0.0)
            
            delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
            l_log2e = l * LOG2_E
            
            scores_log2e = tl.dot(k_log2e, q.trans(1, 0))
            
            valid = mask_n[:, None] & mask_m[None, :]
            scores_log2e = tl.where(valid, scores_log2e, float("-inf"))
            
            p = tl.math.exp2(scores_log2e - l_log2e[None, :])
            p = tl.where(valid, p, 0.0)
            
            dv_acc += tl.dot(p.to(q.dtype), do)
            
            dp_t = tl.dot(v, do.trans(1, 0))
            ds_t = p * (dp_t - delta[None, :]) * scale
            ds_t = tl.where(valid, ds_t, 0.0)
            
            dk_acc += tl.dot(ds_t.to(q.dtype), q)
            
            q_ptrs += BM_DK * stride_qs
            o_ptrs += BM_DK * stride_os
            do_ptrs += BM_DK * stride_dos
            l_ptrs += BM_DK * stride_ls
            
        dk_ptrs = dK + b * stride_dkb + h * stride_dkh + offs_n[:, None] * stride_dks + offs_d[None, :] * stride_dkd
        dv_ptrs = dV + b * stride_dvb + h * stride_dvh + offs_n[:, None] * stride_dvs + offs_d[None, :] * stride_dvd
        
        tl.store(dk_ptrs, dk_acc.to(dK.dtype.element_ty), mask=mask_n[:, None])
        tl.store(dv_ptrs, dv_acc.to(dV.dtype.element_ty), mask=mask_n[:, None])
        
    else:
        # =====================================================================
        # REGION 1: Exclusive owner for a dQ tile
        # =====================================================================
        pid_m = pid - num_kv_tiles
        
        offs_m = pid_m * BM_DQ + tl.arange(0, BM_DQ)
        mask_m = offs_m < S
        
        q_ptrs = Q + b * stride_qb + h * stride_qh + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
        o_ptrs = O + b * stride_ob + h * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
        do_ptrs = dO + b * stride_dob + h * stride_doh + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod
        l_ptrs = L + b * stride_lb + h * stride_lh + offs_m * stride_ls
        
        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
        o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
        do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
        l = tl.load(l_ptrs, mask=mask_m, other=0.0)
        
        # Pre-scale Q to absorb softmax scale and log2(e) for the loop
        q_log2e = (q * (scale * LOG2_E)).to(q.dtype)
        
        delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
        l_log2e = l * LOG2_E
        
        dq_acc = tl.zeros((BM_DQ, d), dtype=tl.float32)
        
        offs_n = tl.arange(0, BN_DQ)
        k_ptrs = K + b * stride_kb + h * stride_kh + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        v_ptrs = V + b * stride_vb + h * stride_vh + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
        
        loop_kv_tiles = tl.cdiv(S, BN_DQ)
        for kv_tile in tl.range(0, loop_kv_tiles, num_stages=LOOP_STAGES):
            kv_offs_n = kv_tile * BN_DQ + offs_n
            mask_n = kv_offs_n < S
            
            k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
            v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
            
            scores_log2e = tl.dot(q_log2e, k.trans(1, 0))
            
            valid = mask_m[:, None] & mask_n[None, :]
            scores_log2e = tl.where(valid, scores_log2e, float("-inf"))
            
            p = tl.math.exp2(scores_log2e - l_log2e[:, None])
            p = tl.where(valid, p, 0.0)
            
            dp = tl.dot(do, v.trans(1, 0))
            ds = p * (dp - delta[:, None]) * scale
            ds = tl.where(valid, ds, 0.0)
            
            dq_acc += tl.dot(ds.to(q.dtype), k)
            
            k_ptrs += BN_DQ * stride_ks
            v_ptrs += BN_DQ * stride_vs
            
        dq_ptrs = dQ + b * stride_dqb + h * stride_dqh + offs_m[:, None] * stride_dqs + offs_d[None, :] * stride_dqd
        tl.store(dq_ptrs, dq_acc.to(dQ.dtype.element_ty), mask=mask_m[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes exact backward gradients for multi-head attention without causal masking.
    
    Operates on bf16 tensors with shape [B, H, S, d]. Output dQ, dK, and dV are fully stored
    using a unified flat-grid layout with split output ownership to maximize parallel compute,
    leverage efficient scheduling, and eliminate atomic operations.
    """
    with torch.cuda.device(Q.device):
        B_val, H_val, S_val, d_val = Q.shape
        scale = 1.0 / (d_val ** 0.5)
        stride_ls = L.stride(2) if L.dim() >= 3 else L.stride(-1)

        def grid(META):
            num_kv = triton.cdiv(S_val, META['BN_DK'])
            num_q = triton.cdiv(S_val, META['BM_DQ'])
            return (num_kv + num_q, B_val * H_val)

        bwd_kernel[grid](
            Q, K, V, O, dO, L,
            dQ, dK, dV,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
            L.stride(0), L.stride(1), stride_ls,
            dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
            dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
            dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
            B_val, H_val, S_val, scale,
            d=d_val
        )