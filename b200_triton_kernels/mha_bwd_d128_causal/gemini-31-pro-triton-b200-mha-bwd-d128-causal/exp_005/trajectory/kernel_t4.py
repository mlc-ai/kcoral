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
    stride_lb, stride_lh, stride_ls,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    B, H, S, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr
):
    start_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H
    
    # Cast to 64-bit to prevent any silent overflow on large pointer offsets
    off_b = pid_b.to(tl.int64)
    off_h = pid_h.to(tl.int64)
    
    m_start = start_m * BLOCK_M
    offs_m = m_start + tl.arange(0, BLOCK_M)
    mask_m = offs_m < S
    offs_d = tl.arange(0, D)
    
    q_ptrs = Q + off_b * stride_qb + off_h * stride_qh + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    do_ptrs = dO + off_b * stride_dob + off_h * stride_doh + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod
    o_ptrs = O + off_b * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    l_ptrs = L + off_b * stride_lb + off_h * stride_lh + offs_m * stride_ls
    
    # Load resident query block tensors
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
    o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
    lse = tl.load(l_ptrs, mask=mask_m, other=0.0)
    
    # Precalculate delta globally for the block avoiding repeated work
    delta_val = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
    
    dq = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    
    k_base = K + off_b * stride_kb + off_h * stride_kh + offs_d[None, :] * stride_kd
    v_base = V + off_b * stride_vb + off_h * stride_vh + offs_d[None, :] * stride_vd
    
    # Causal sequence limit
    max_offs_m = tl.minimum(S, m_start + BLOCK_M)
    end_n = (max_offs_m + BLOCK_N - 1) // BLOCK_N
    
    for start_n in range(0, end_n):
        offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S
        
        k_ptrs = k_base + offs_n[:, None] * stride_ks
        v_ptrs = v_base + offs_n[:, None] * stride_vs
        
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        
        scores = tl.dot(q, tl.trans(k)) * scale
        
        # Computing the mask unconditionally prevents divergence compilation bugs 
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        valid = causal_mask & mask_n[None, :] & mask_m[:, None]
        
        scores = tl.where(valid, scores, float("-inf"))
        p = tl.math.exp(scores - lse[:, None])
        p = tl.where(valid, p, 0.0)
        
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta_val[:, None]) * scale
        ds = tl.where(valid, ds, 0.0)
        
        dq += tl.dot(ds.to(q.dtype), k)
        
    dq_ptrs = dQ + off_b * stride_dqb + off_h * stride_dqh + offs_m[:, None] * stride_dqs + offs_d[None, :] * stride_dqd
    tl.store(dq_ptrs, dq.to(q.dtype), mask=mask_m[:, None])


@triton.jit
def _bwd_kernel_dk_dv(
    Q, K, V, O, dO, L, dK, dV,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    B, H, S, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr
):
    start_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H
    
    off_b = pid_b.to(tl.int64)
    off_h = pid_h.to(tl.int64)
    
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < S
    offs_d = tl.arange(0, D)
    
    k_ptrs = K + off_b * stride_kb + off_h * stride_kh + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = V + off_b * stride_vb + off_h * stride_vh + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
    
    k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
    
    dk = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    
    start_m_initial = (start_n * BLOCK_N) // BLOCK_M
    end_m = (S + BLOCK_M - 1) // BLOCK_M
    
    q_base = Q + off_b * stride_qb + off_h * stride_qh + offs_d[None, :] * stride_qd
    do_base = dO + off_b * stride_dob + off_h * stride_doh + offs_d[None, :] * stride_dod
    o_base = O + off_b * stride_ob + off_h * stride_oh + offs_d[None, :] * stride_od
    l_base = L + off_b * stride_lb + off_h * stride_lh
    
    for start_m in range(start_m_initial, end_m):
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S
        
        q_ptrs = q_base + offs_m[:, None] * stride_qs
        do_ptrs = do_base + offs_m[:, None] * stride_dos
        o_ptrs = o_base + offs_m[:, None] * stride_os
        l_ptrs = l_base + offs_m * stride_ls
        
        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
        do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
        o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
        lse = tl.load(l_ptrs, mask=mask_m, other=0.0)
        
        delta_val = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
        
        scores_t = tl.dot(k, tl.trans(q)) * scale
        
        causal_mask_t = offs_m[None, :] >= offs_n[:, None]
        valid_t = causal_mask_t & mask_n[:, None] & mask_m[None, :]
        
        scores_t = tl.where(valid_t, scores_t, float("-inf"))
        p_t = tl.math.exp(scores_t - lse[None, :])
        p_t = tl.where(valid_t, p_t, 0.0)
        
        dv += tl.dot(p_t.to(q.dtype), do)
        
        dp_t = tl.dot(v, tl.trans(do))
        ds_t = p_t * (dp_t - delta_val[None, :]) * scale
        ds_t = tl.where(valid_t, ds_t, 0.0)
        
        dk += tl.dot(ds_t.to(q.dtype), q)
        
    dk_ptrs = dK + off_b * stride_dkb + off_h * stride_dkh + offs_n[:, None] * stride_dks + offs_d[None, :] * stride_dkd
    dv_ptrs = dV + off_b * stride_dvb + off_h * stride_dvh + offs_n[:, None] * stride_dvs + offs_d[None, :] * stride_dvd
    
    tl.store(dk_ptrs, dk.to(q.dtype), mask=mask_n[:, None])
    tl.store(dv_ptrs, dv.to(q.dtype), mask=mask_n[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes causal Multi-Head Attention backward with split ownership strategy.
    
    Asymmetric 128x64 pipelines for query ownership blocks (Q, dQ, O, dO held fully resident).
    Asymmetric 64x128 pipelines for KV ownership blocks (K, V, dK, dV held fully resident).
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    scale = 1.0 / (D ** 0.5)
    
    BLOCK_M_DQ = 128
    BLOCK_N_DQ = 64
    grid_dq = (triton.cdiv(S, BLOCK_M_DQ), B * H)
    _bwd_kernel_dq[grid_dq](
        Q, K, V, O, dO, L, dQ,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        B, H, S, scale,
        BLOCK_M=BLOCK_M_DQ, BLOCK_N=BLOCK_N_DQ, D=D,
        num_warps=4, num_stages=3
    )
    
    BLOCK_M_DK = 64
    BLOCK_N_DK = 128
    grid_dkdv = (triton.cdiv(S, BLOCK_N_DK), B * H)
    _bwd_kernel_dk_dv[grid_dkdv](
        Q, K, V, O, dO, L, dK, dV,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        B, H, S, scale,
        BLOCK_M=BLOCK_M_DK, BLOCK_N=BLOCK_N_DK, D=D,
        num_warps=4, num_stages=3
    )