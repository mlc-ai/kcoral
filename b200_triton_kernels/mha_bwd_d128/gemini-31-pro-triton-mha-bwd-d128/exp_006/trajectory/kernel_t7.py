import torch
import triton
import triton.language as tl

@triton.jit
def bwd_kernel_dq(
    Q, K, V, O, dO, L, dQ,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    S, scale: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, d: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, d)
    mask_m = offs_m < S
    
    # Outer loop variables for M block
    q_ptrs = Q + pid_b * stride_qb + pid_h * stride_qh + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    do_ptrs = dO + pid_b * stride_dob + pid_h * stride_doh + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod
    o_ptrs = O + pid_b * stride_ob + pid_h * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    l_ptrs = L + pid_b * stride_lb + pid_h * stride_lh + offs_m * stride_ls
    
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
    o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
    l = tl.load(l_ptrs, mask=mask_m, other=0.0)
    
    # Precompute rowsum(dO * O) in FP32
    d_val = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
    
    # Explicitly remove `o` from registers
    del o
    
    # Initialize FP32 accumulator for dQ
    dq = tl.zeros((BLOCK_M, d), tl.float32)
    
    k_ptrs_base = K + pid_b * stride_kb + pid_h * stride_kh + offs_d[None, :] * stride_kd
    v_ptrs_base = V + pid_b * stride_vb + pid_h * stride_vh + offs_d[None, :] * stride_vd
    
    offs_n_init = tl.arange(0, BLOCK_N)
    num_n_blocks = tl.cdiv(S, BLOCK_N)
    
    for n in range(num_n_blocks):
        offs_n = n * BLOCK_N + offs_n_init
        mask_n = offs_n < S
        
        k_ptrs = k_ptrs_base + offs_n[:, None] * stride_ks
        v_ptrs = v_ptrs_base + offs_n[:, None] * stride_vs
        
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        
        # S = Q @ K^T
        s = tl.dot(q, tl.trans(k))
        s = s * scale
        
        # P = exp(S - L)
        p = tl.exp(s - l[:, None])
        p = tl.where(mask_m[:, None] & mask_n[None, :], p, 0.0)
        
        # dP = dO @ V^T
        dp = tl.dot(do, tl.trans(v))
        
        # dS = P * (dP - D) * scale
        ds = (p * (dp - d_val[:, None]) * scale).to(q.dtype)
        
        # dQ += dS @ K
        dq = tl.dot(ds, k, dq)
        
    dq_ptrs = dQ + pid_b * stride_dqb + pid_h * stride_dqh + offs_m[:, None] * stride_dqs + offs_d[None, :] * stride_dqd
    tl.store(dq_ptrs, dq.to(q.dtype), mask=mask_m[:, None])


@triton.jit
def bwd_kernel_dk_dv(
    Q, K, V, O, dO, L, dK, dV,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    S, scale: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, d: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, d)
    mask_n = offs_n < S
    
    # Outer loop variables for N block
    k_ptrs = K + pid_b * stride_kb + pid_h * stride_kh + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = V + pid_b * stride_vb + pid_h * stride_vh + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
    
    k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
    
    dk = tl.zeros((BLOCK_N, d), tl.float32)
    dv = tl.zeros((BLOCK_N, d), tl.float32)
    
    q_ptrs_base = Q + pid_b * stride_qb + pid_h * stride_qh + offs_d[None, :] * stride_qd
    do_ptrs_base = dO + pid_b * stride_dob + pid_h * stride_doh + offs_d[None, :] * stride_dod
    o_ptrs_base = O + pid_b * stride_ob + pid_h * stride_oh + offs_d[None, :] * stride_od
    l_ptrs_base = L + pid_b * stride_lb + pid_h * stride_lh
    
    offs_m_init = tl.arange(0, BLOCK_M)
    num_m_blocks = tl.cdiv(S, BLOCK_M)
    
    for m in range(num_m_blocks):
        offs_m = m * BLOCK_M + offs_m_init
        mask_m = offs_m < S
        
        q_ptrs = q_ptrs_base + offs_m[:, None] * stride_qs
        do_ptrs = do_ptrs_base + offs_m[:, None] * stride_dos
        o_ptrs = o_ptrs_base + offs_m[:, None] * stride_os
        l_ptrs = l_ptrs_base + offs_m * stride_ls
        
        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
        do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
        o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
        l = tl.load(l_ptrs, mask=mask_m, other=0.0)
        
        d_val = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
        del o
        
        # TRANSPOSED MATH trick: Avert SMEM transpose
        # s_t = K @ Q^T logically becomes [BLOCK_N, BLOCK_M]
        s_t = tl.dot(k, tl.trans(q))
        s_t = s_t * scale
        
        p_t = tl.exp(s_t - l[None, :])
        p_t = tl.where(mask_n[:, None] & mask_m[None, :], p_t, 0.0)
        
        # dp_t = V @ dO^T logically [BLOCK_N, BLOCK_M]
        dp_t = tl.dot(v, tl.trans(do))
        
        ds_t = (p_t * (dp_t - d_val[None, :]) * scale).to(q.dtype)
        p_t_bf16 = p_t.to(q.dtype)
        
        # Accelerate via optimally packed tensor ordering.
        dv = tl.dot(p_t_bf16, do, dv)
        dk = tl.dot(ds_t, q, dk)
        
    dk_ptrs = dK + pid_b * stride_dkb + pid_h * stride_dkh + offs_n[:, None] * stride_dks + offs_d[None, :] * stride_dkd
    dv_ptrs = dV + pid_b * stride_dvb + pid_h * stride_dvh + offs_n[:, None] * stride_dvs + offs_d[None, :] * stride_dvd
    
    tl.store(dk_ptrs, dk.to(k.dtype), mask=mask_n[:, None])
    tl.store(dv_ptrs, dv.to(v.dtype), mask=mask_n[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes FlashAttention-3 backward pass efficiently.
    Uses dedicated kernels for destination-passing.
    """
    with torch.cuda.device(Q.device):
        B, H, S, d = Q.shape
        scale = 1.0 / (d ** 0.5)

        # Tune grid and pipeline buffers carefully for Hopper SM90a 
        # (128x64 outer x inner leverages exactly 192KB of the 228KB limit perfectly)
        BLOCK_M_DQ = 128
        BLOCK_N_DQ = 64
        grid_dq = (triton.cdiv(S, BLOCK_M_DQ), H, B)

        bwd_kernel_dq[grid_dq](
            Q, K, V, O, dO, L, dQ,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
            L.stride(0), L.stride(1), L.stride(2),
            dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
            S, scale,
            BLOCK_M=BLOCK_M_DQ, BLOCK_N=BLOCK_N_DQ, d=d,
            num_warps=8, num_stages=3
        )

        # Inverting sizes yields symmetric buffer limits.
        # (64x128 avoids register spilling across accumulators for K and V, maxes out at ~192 regs).
        BLOCK_M_DK = 64
        BLOCK_N_DK = 128
        grid_dk_dv = (triton.cdiv(S, BLOCK_N_DK), H, B)

        bwd_kernel_dk_dv[grid_dk_dv](
            Q, K, V, O, dO, L, dK, dV,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
            L.stride(0), L.stride(1), L.stride(2),
            dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
            dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
            S, scale,
            BLOCK_M=BLOCK_M_DK, BLOCK_N=BLOCK_N_DK, d=d,
            num_warps=8, num_stages=3
        )