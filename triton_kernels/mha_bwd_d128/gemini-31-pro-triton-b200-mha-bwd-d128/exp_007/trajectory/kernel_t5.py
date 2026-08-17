import math
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
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
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    d: tl.constexpr
):
    pid = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    num_kv_tiles = tl.cdiv(S, BLOCK_N)
    
    # Base offsets localized to each Head & Batch
    off_q = pid_b * stride_qb + pid_h * stride_qh
    off_k = pid_b * stride_kb + pid_h * stride_kh
    off_v = pid_b * stride_vb + pid_h * stride_vh
    off_o = pid_b * stride_ob + pid_h * stride_oh
    off_do = pid_b * stride_dob + pid_h * stride_doh
    off_dq = pid_b * stride_dqb + pid_h * stride_dqh
    off_dk = pid_b * stride_dkb + pid_h * stride_dkh
    off_dv = pid_b * stride_dvb + pid_h * stride_dvh
    off_lse = pid_b * stride_lseb + pid_h * stride_lseh
    
    offs_d = tl.arange(0, d)
    
    if pid < num_kv_tiles:
        # -------------------------------------------------------------
        # Region 1: dK / dV Owners (exclusively mapped)
        # -------------------------------------------------------------
        kv_tile = pid
        offs_n = tl.arange(0, BLOCK_N)
        mask_n = (kv_tile * BLOCK_N + offs_n) < S
        
        # Resident Key & Value chunks loaded exactly once
        k_ptrs = K + off_k + (kv_tile * BLOCK_N + offs_n)[:, None] * stride_ks + offs_d[None, :] * stride_kd
        v_ptrs = V + off_v + (kv_tile * BLOCK_N + offs_n)[:, None] * stride_vs + offs_d[None, :] * stride_vd
        
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        
        dk = tl.zeros((BLOCK_N, d), tl.float32)
        dv = tl.zeros((BLOCK_N, d), tl.float32)
        
        offs_m_base = tl.arange(0, BLOCK_M)
        q_ptrs = Q + off_q + offs_m_base[:, None] * stride_qs + offs_d[None, :] * stride_qd
        o_ptrs = O + off_o + offs_m_base[:, None] * stride_os + offs_d[None, :] * stride_od
        do_ptrs = dO + off_do + offs_m_base[:, None] * stride_dos + offs_d[None, :] * stride_dod
        lse_ptrs = LSE + off_lse + offs_m_base * stride_lses
        
        for q_tile in range(tl.cdiv(S, BLOCK_M)):
            # Masks guarantee 100% safe asynchronous TMA Prefetches when num_stages > 1
            mask_m = (q_tile * BLOCK_M + offs_m_base) < S
            q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
            o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
            do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
            lse = tl.load(lse_ptrs, mask=mask_m, other=0.0)
            
            # Rowwise delta evaluates cleanly inside FP32 registers 
            delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
            
            qk_t = tl.dot(k, q.T, out_dtype=tl.float32) * softmax_scale
            p_t = tl.math.exp(qk_t - lse[None, :])
            
            mask_nm = mask_n[:, None] & mask_m[None, :]
            p_t = tl.where(mask_nm, p_t, 0.0)
            
            dv += tl.dot(p_t.to(q.dtype), do, out_dtype=tl.float32)
            
            dp_t = tl.dot(v, do.T, out_dtype=tl.float32)
            ds_t = p_t * (dp_t - delta[None, :]) * softmax_scale
            
            dk += tl.dot(ds_t.to(q.dtype), q, out_dtype=tl.float32)
            
            # Linear Pointer advancements completely bypass recalculations 
            q_ptrs += BLOCK_M * stride_qs
            o_ptrs += BLOCK_M * stride_os
            do_ptrs += BLOCK_M * stride_dos
            lse_ptrs += BLOCK_M * stride_lses
            
        dk_ptrs = dK + off_dk + (kv_tile * BLOCK_N + offs_n)[:, None] * stride_dks + offs_d[None, :] * stride_dkd
        dv_ptrs = dV + off_dv + (kv_tile * BLOCK_N + offs_n)[:, None] * stride_dvs + offs_d[None, :] * stride_dvd
        
        tl.store(dk_ptrs, dk.to(k.dtype), mask=mask_n[:, None])
        tl.store(dv_ptrs, dv.to(v.dtype), mask=mask_n[:, None])
        
    else:
        # -------------------------------------------------------------
        # Region 2: dQ Owners (exclusively mapped)
        # -------------------------------------------------------------
        q_tile = pid - num_kv_tiles
        offs_m = tl.arange(0, BLOCK_M)
        mask_m = (q_tile * BLOCK_M + offs_m) < S
        
        q_ptrs = Q + off_q + (q_tile * BLOCK_M + offs_m)[:, None] * stride_qs + offs_d[None, :] * stride_qd
        o_ptrs = O + off_o + (q_tile * BLOCK_M + offs_m)[:, None] * stride_os + offs_d[None, :] * stride_od
        do_ptrs = dO + off_do + (q_tile * BLOCK_M + offs_m)[:, None] * stride_dos + offs_d[None, :] * stride_dod
        lse_ptrs = LSE + off_lse + (q_tile * BLOCK_M + offs_m) * stride_lses
        
        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
        o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
        do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
        lse = tl.load(lse_ptrs, mask=mask_m, other=0.0)
        
        delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
        dq = tl.zeros((BLOCK_M, d), tl.float32)
        
        offs_n_base = tl.arange(0, BLOCK_N)
        k_ptrs = K + off_k + offs_n_base[:, None] * stride_ks + offs_d[None, :] * stride_kd
        v_ptrs = V + off_v + offs_n_base[:, None] * stride_vs + offs_d[None, :] * stride_vd
        
        for kv_tile in range(tl.cdiv(S, BLOCK_N)):
            mask_n = (kv_tile * BLOCK_N + offs_n_base) < S
            k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
            v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
            
            qk = tl.dot(q, k.T, out_dtype=tl.float32) * softmax_scale
            p = tl.math.exp(qk - lse[:, None])
            
            mask_mn = mask_m[:, None] & mask_n[None, :]
            p = tl.where(mask_mn, p, 0.0)
            
            dp = tl.dot(do, v.T, out_dtype=tl.float32)
            ds = p * (dp - delta[:, None]) * softmax_scale
            
            dq += tl.dot(ds.to(q.dtype), k, out_dtype=tl.float32)
            
            k_ptrs += BLOCK_N * stride_ks
            v_ptrs += BLOCK_N * stride_vs
            
        dq_ptrs = dQ + off_dq + (q_tile * BLOCK_M + offs_m)[:, None] * stride_dqs + offs_d[None, :] * stride_dqd
        tl.store(dq_ptrs, dq.to(q.dtype), mask=mask_m[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes Standard-Triton B200 FA Backward perfectly maintaining SRAM bandwidth mappings whilst resolving global atomics conflicts natively.
    """
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    softmax_scale = 1.0 / math.sqrt(d)
    
    # Unifies the Split-Ownership directly into a Flat Program Configuration
    def grid_fn(META):
        num_kv = triton.cdiv(S, META['BLOCK_N'])
        num_q = triton.cdiv(S, META['BLOCK_M'])
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
        d=d
    )