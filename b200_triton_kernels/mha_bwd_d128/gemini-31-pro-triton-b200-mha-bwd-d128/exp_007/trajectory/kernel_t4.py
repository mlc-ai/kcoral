import math
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M_DQ': 128, 'BLOCK_N_DQ': 64,  'BLOCK_M_DKDV': 64,  'BLOCK_N_DKDV': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M_DQ': 64,  'BLOCK_N_DQ': 128, 'BLOCK_M_DKDV': 128, 'BLOCK_N_DKDV': 64},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M_DQ': 128, 'BLOCK_N_DQ': 128, 'BLOCK_M_DKDV': 128, 'BLOCK_N_DKDV': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M_DQ': 64,  'BLOCK_N_DQ': 64,  'BLOCK_M_DKDV': 64,  'BLOCK_N_DKDV': 64},  num_warps=4, num_stages=4),
    ],
    key=['S']
)
@triton.jit
def _bwd_kernel(
    Q, K, V, O, dO, LSE, dQ, dK, dV,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    stride_lseb, stride_lseh, stride_lses,
    S, softmax_scale,
    BLOCK_M_DQ: tl.constexpr, BLOCK_N_DQ: tl.constexpr,
    BLOCK_M_DKDV: tl.constexpr, BLOCK_N_DKDV: tl.constexpr,
    d: tl.constexpr, S_ALIGNED: tl.constexpr
):
    pid = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    num_kv_tiles = tl.cdiv(S, BLOCK_N_DKDV)
    
    off_q = pid_b * stride_qb + pid_h * stride_qh
    off_k = pid_b * stride_kb + pid_h * stride_kh
    off_v = pid_b * stride_vb + pid_h * stride_vh
    off_o = pid_b * stride_ob + pid_h * stride_oh
    off_do = pid_b * stride_dob + pid_h * stride_doh
    off_dq = pid_b * stride_dqb + pid_h * stride_dqh
    off_dk = pid_b * stride_dkb + pid_h * stride_dkh
    off_dv = pid_b * stride_dvb + pid_h * stride_dvh
    off_lse = pid_b * stride_lseb + pid_h * stride_lseh
    
    if pid < num_kv_tiles:
        # -------------------------------------------------------------
        # Region 1: dK / dV Owners (exclusively mapped)
        # -------------------------------------------------------------
        kv_tile = pid
        offs_n = kv_tile * BLOCK_N_DKDV + tl.arange(0, BLOCK_N_DKDV)
        offs_d = tl.arange(0, d)
        
        k_ptrs = K + off_k + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        v_ptrs = V + off_v + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
        
        if S_ALIGNED:
            k = tl.load(k_ptrs)
            v = tl.load(v_ptrs)
        else:
            mask_n = offs_n < S
            k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
            v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        
        dk = tl.zeros((BLOCK_N_DKDV, d), tl.float32)
        dv = tl.zeros((BLOCK_N_DKDV, d), tl.float32)
        
        offs_m = tl.arange(0, BLOCK_M_DKDV)
        q_ptrs = Q + off_q + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
        o_ptrs = O + off_o + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
        do_ptrs = dO + off_do + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod
        lse_ptrs = LSE + off_lse + offs_m * stride_lses
        
        for q_tile in range(tl.cdiv(S, BLOCK_M_DKDV)):
            if S_ALIGNED:
                q = tl.load(q_ptrs)
                o = tl.load(o_ptrs)
                do = tl.load(do_ptrs)
                lse = tl.load(lse_ptrs)
            else:
                mask_m = (q_tile * BLOCK_M_DKDV + offs_m) < S
                q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
                o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
                do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
                lse = tl.load(lse_ptrs, mask=mask_m, other=0.0)
            
            # Rowwise delta evaluates inline efficiently utilizing cache hit retention directly
            delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
            
            qk_t = tl.dot(k, q.T, out_dtype=tl.float32) * softmax_scale
            p_t = tl.math.exp(qk_t - lse[None, :])
            
            if not S_ALIGNED:
                mask_nm = mask_n[:, None] & mask_m[None, :]
                p_t = tl.where(mask_nm, p_t, 0.0)
            
            dv += tl.dot(p_t.to(q.dtype), do, out_dtype=tl.float32)
            
            dp_t = tl.dot(v, do.T, out_dtype=tl.float32)
            ds_t = p_t * (dp_t - delta[None, :]) * softmax_scale
            
            dk += tl.dot(ds_t.to(q.dtype), q, out_dtype=tl.float32)
            
            q_ptrs += BLOCK_M_DKDV * stride_qs
            o_ptrs += BLOCK_M_DKDV * stride_os
            do_ptrs += BLOCK_M_DKDV * stride_dos
            lse_ptrs += BLOCK_M_DKDV * stride_lses
            
        dk_ptrs = dK + off_dk + offs_n[:, None] * stride_dks + offs_d[None, :] * stride_dkd
        dv_ptrs = dV + off_dv + offs_n[:, None] * stride_dvs + offs_d[None, :] * stride_dvd
        
        if S_ALIGNED:
            tl.store(dk_ptrs, dk.to(dK.dtype.element_ty))
            tl.store(dv_ptrs, dv.to(dV.dtype.element_ty))
        else:
            tl.store(dk_ptrs, dk.to(dK.dtype.element_ty), mask=mask_n[:, None])
            tl.store(dv_ptrs, dv.to(dV.dtype.element_ty), mask=mask_n[:, None])
        
    else:
        # -------------------------------------------------------------
        # Region 2: dQ Owners (exclusively mapped)
        # -------------------------------------------------------------
        q_tile = pid - num_kv_tiles
        offs_m = q_tile * BLOCK_M_DQ + tl.arange(0, BLOCK_M_DQ)
        offs_d = tl.arange(0, d)
        
        q_ptrs = Q + off_q + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
        o_ptrs = O + off_o + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
        do_ptrs = dO + off_do + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod
        lse_ptrs = LSE + off_lse + offs_m * stride_lses
        
        if S_ALIGNED:
            q = tl.load(q_ptrs)
            o = tl.load(o_ptrs)
            do = tl.load(do_ptrs)
            lse = tl.load(lse_ptrs)
        else:
            mask_m = offs_m < S
            q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
            o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
            do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
            lse = tl.load(lse_ptrs, mask=mask_m, other=0.0)
        
        delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
        dq = tl.zeros((BLOCK_M_DQ, d), tl.float32)
        
        offs_n = tl.arange(0, BLOCK_N_DQ)
        k_ptrs = K + off_k + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        v_ptrs = V + off_v + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
        
        for kv_tile in range(tl.cdiv(S, BLOCK_N_DQ)):
            if S_ALIGNED:
                k = tl.load(k_ptrs)
                v = tl.load(v_ptrs)
            else:
                mask_n = (kv_tile * BLOCK_N_DQ + offs_n) < S
                k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
                v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
            
            qk = tl.dot(q, k.T, out_dtype=tl.float32) * softmax_scale
            p = tl.math.exp(qk - lse[:, None])
            
            if not S_ALIGNED:
                mask_mn = mask_m[:, None] & mask_n[None, :]
                p = tl.where(mask_mn, p, 0.0)
            
            dp = tl.dot(do, v.T, out_dtype=tl.float32)
            ds = p * (dp - delta[:, None]) * softmax_scale
            
            dq += tl.dot(ds.to(q.dtype), k, out_dtype=tl.float32)
            
            k_ptrs += BLOCK_N_DQ * stride_ks
            v_ptrs += BLOCK_N_DQ * stride_vs
            
        dq_ptrs = dQ + off_dq + offs_m[:, None] * stride_dqs + offs_d[None, :] * stride_dqd
        
        if S_ALIGNED:
            tl.store(dq_ptrs, dq.to(dQ.dtype.element_ty))
        else:
            tl.store(dq_ptrs, dq.to(dQ.dtype.element_ty), mask=mask_m[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes Standard-Triton Split Ownership Backward seamlessly abstracting atomics and allocating efficiently.
    """
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    softmax_scale = 1.0 / math.sqrt(d)
    
    # Static compile-time inference to strip loop masks generating dense SM computations
    S_ALIGNED = (S % 128 == 0)
    
    def grid_fn(META):
        num_kv = triton.cdiv(S, META['BLOCK_N_DKDV'])
        num_q = triton.cdiv(S, META['BLOCK_M_DQ'])
        return (num_kv + num_q, B, H)
        
    _bwd_kernel[grid_fn](
        Q, K, V, O, dO, L, dQ, dK, dV,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        S, softmax_scale,
        d=d, S_ALIGNED=S_ALIGNED
    )