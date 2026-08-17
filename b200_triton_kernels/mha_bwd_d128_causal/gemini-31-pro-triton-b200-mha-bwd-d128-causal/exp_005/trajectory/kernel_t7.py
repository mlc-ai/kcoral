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
    
    # Cast base offsets to 64-bit to prevent any silent overflow on huge sequence batches
    off_b = (pid_bh // H).to(tl.int64)
    off_h = (pid_bh % H).to(tl.int64)
    
    m_start = start_m * BLOCK_M
    offs_m = m_start + tl.arange(0, BLOCK_M)
    mask_m = offs_m < S
    offs_d = tl.arange(0, D)
    
    # Pre-calculate offset bases
    q_ptrs = Q + off_b * stride_qb + off_h * stride_qh + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    do_ptrs = dO + off_b * stride_dob + off_h * stride_doh + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod
    o_ptrs = O + off_b * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    l_ptrs = L + off_b * stride_lb + off_h * stride_lh + offs_m * stride_ls
    
    # Load resident Q-block tensors
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
    o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
    lse = tl.load(l_ptrs, mask=mask_m, other=0.0)
    
    # Precalculate delta unconditionally for the resident Q-block
    delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
    
    dq = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    
    # Calculate bounds limits to securely prune processing for completely non-causal K blocks early
    k_base = K + off_b * stride_kb + off_h * stride_kh + offs_d[None, :] * stride_kd
    v_base = V + off_b * stride_vb + off_h * stride_vh + offs_d[None, :] * stride_vd
    
    max_offs_m = tl.minimum(S, m_start + BLOCK_M)
    end_n = (max_offs_m + BLOCK_N - 1) // BLOCK_N
    
    RCP_LN2 = 1.4426950408889634
    
    for start_n in range(0, end_n):
        n_start = start_n * BLOCK_N
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S
        
        k_ptrs = k_base + offs_n[:, None] * stride_ks
        v_ptrs = v_base + offs_n[:, None] * stride_vs
        
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        
        # Native Hardware Tensor Core Mathematics mappings accumulating internally in FP32
        scores = tl.dot(q, k.T, out_dtype=tl.float32) * scale
        
        causal = offs_m[:, None] >= offs_n[None, :]
        valid = causal & mask_m[:, None] & mask_n[None, :]
        
        scores = tl.where(valid, scores, float("-inf"))
        p = tl.math.exp2((scores - lse[:, None]) * RCP_LN2)
        p = tl.where(valid, p, 0.0)
        
        dp = tl.dot(do, v.T, out_dtype=tl.float32)
        ds = p * (dp - delta[:, None]) * scale
        ds = tl.where(valid, ds, 0.0)
        
        dq += tl.dot(ds.to(tl.bfloat16), k, out_dtype=tl.float32)
        
    dq_ptrs = dQ + off_b * stride_dqb + off_h * stride_dqh + offs_m[:, None] * stride_dqs + offs_d[None, :] * stride_dqd
    tl.store(dq_ptrs, dq.to(tl.bfloat16), mask=mask_m[:, None])


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
    
    off_b = (pid_bh // H).to(tl.int64)
    off_h = (pid_bh % H).to(tl.int64)
    
    n_start = start_n * BLOCK_N
    offs_n = n_start + tl.arange(0, BLOCK_N)
    mask_n = offs_n < S
    offs_d = tl.arange(0, D)
    
    k_ptrs = K + off_b * stride_kb + off_h * stride_kh + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = V + off_b * stride_vb + off_h * stride_vh + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
    
    # Store KV ownership structures resident inside distributed SM100 warp registers
    k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
    
    dk = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    
    # Compute limit iteration boundary safely utilizing the strictly causally-valid diagonal origin
    start_m_initial = n_start // BLOCK_M
    end_m = (S + BLOCK_M - 1) // BLOCK_M
    
    q_base = Q + off_b * stride_qb + off_h * stride_qh + offs_d[None, :] * stride_qd
    do_base = dO + off_b * stride_dob + off_h * stride_doh + offs_d[None, :] * stride_dod
    o_base = O + off_b * stride_ob + off_h * stride_oh + offs_d[None, :] * stride_od
    l_base = L + off_b * stride_lb + off_h * stride_lh
    
    RCP_LN2 = 1.4426950408889634
    
    for start_m in range(start_m_initial, end_m):
        m_start = start_m * BLOCK_M
        offs_m = m_start + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S
        
        q_ptrs = q_base + offs_m[:, None] * stride_qs
        do_ptrs = do_base + offs_m[:, None] * stride_dos
        o_ptrs = o_base + offs_m[:, None] * stride_os
        l_ptrs = l_base + offs_m * stride_ls
        
        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
        do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
        o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
        lse = tl.load(l_ptrs, mask=mask_m, other=0.0)
        
        delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
        
        scores_t = tl.dot(k, q.T, out_dtype=tl.float32) * scale
        
        causal = offs_m[None, :] >= offs_n[:, None]
        valid_t = causal & mask_n[:, None] & mask_m[None, :]
        
        scores_t = tl.where(valid_t, scores_t, float("-inf"))
        p_t = tl.math.exp2((scores_t - lse[None, :]) * RCP_LN2)
        p_t = tl.where(valid_t, p_t, 0.0)
        
        dv += tl.dot(p_t.to(tl.bfloat16), do, out_dtype=tl.float32)
        
        dp_t = tl.dot(v, do.T, out_dtype=tl.float32)
        ds_t = p_t * (dp_t - delta[None, :]) * scale
        ds_t = tl.where(valid_t, ds_t, 0.0)
        
        dk += tl.dot(ds_t.to(tl.bfloat16), q, out_dtype=tl.float32)
        
    dk_ptrs = dK + off_b * stride_dkb + off_h * stride_dkh + offs_n[:, None] * stride_dks + offs_d[None, :] * stride_dkd
    dv_ptrs = dV + off_b * stride_dvb + off_h * stride_dvh + offs_n[:, None] * stride_dvs + offs_d[None, :] * stride_dvd
    
    tl.store(dk_ptrs, dk.to(tl.bfloat16), mask=mask_n[:, None])
    tl.store(dv_ptrs, dv.to(tl.bfloat16), mask=mask_n[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes causal Multi-Head Attention backward with Blackwell decoupled ownership strategy.
    Implements decoupled 64x64 blocks coupled with num_warps=8 configuration effectively 
    keeping peak allocations per thread fully inside the native SM limit envelope avoiding memcheck triggers. 
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    scale = 1.0 / (D ** 0.5)
    
    # Using symmetrical block limits slices allocations neatly avoiding memory spilling faults completely
    BLOCK_M = 64
    BLOCK_N = 64
    
    grid_dq = (triton.cdiv(S, BLOCK_M), B * H)
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
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, D=D,
        num_warps=8, num_stages=3
    )
    
    grid_dkdv = (triton.cdiv(S, BLOCK_N), B * H)
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
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, D=D,
        num_warps=8, num_stages=3
    )