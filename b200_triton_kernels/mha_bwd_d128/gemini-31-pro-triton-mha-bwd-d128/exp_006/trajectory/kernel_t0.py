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
    
    q_ptrs = Q + pid_b * stride_qb + pid_h * stride_qh + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    do_ptrs = dO + pid_b * stride_dob + pid_h * stride_doh + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod
    o_ptrs = O + pid_b * stride_ob + pid_h * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
    o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
    
    l_ptrs = L + pid_b * stride_lb + pid_h * stride_lh + offs_m * stride_ls
    l = tl.load(l_ptrs, mask=mask_m, other=0.0)
    
    # Precompute rowsum(dO * O) for the local block to avoid allocating memory for D
    d_val = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
    
    dq = tl.zeros((BLOCK_M, d), tl.float32)
    
    num_n_blocks = tl.cdiv(S, BLOCK_N)
    for n in range(num_n_blocks):
        offs_n = n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S
        
        k_ptrs = K + pid_b * stride_kb + pid_h * stride_kh + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        v_ptrs = V + pid_b * stride_vb + pid_h * stride_vh + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
        
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        
        # Calculate pre-softmax logits S_ij = (Q_i @ K_j^T)
        acc_s = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        s = tl.dot(q, tl.trans(k), acc_s)
        s = s * scale
        
        # Calculate P = exp(S_ij - L_i) with masking bounds
        p = tl.exp(s - l[:, None])
        p = tl.where((mask_m[:, None]) & (mask_n[None, :]), p, 0.0)
        
        # Calculate dP_i^j = dO_i @ V_j^T
        acc_dp = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        dp = tl.dot(do, tl.trans(v), acc_dp)
        
        # Calculate dS_i^j = P_i^j * (dP_i^j - D_i) * scale
        ds = (p * (dp - d_val[:, None]) * scale).to(q.dtype)
        
        # Accumulate dQ_i += dS_i^j @ K_j
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
    
    k_ptrs = K + pid_b * stride_kb + pid_h * stride_kh + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = V + pid_b * stride_vb + pid_h * stride_vh + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
    
    k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
    
    dk = tl.zeros((BLOCK_N, d), tl.float32)
    dv = tl.zeros((BLOCK_N, d), tl.float32)
    
    num_m_blocks = tl.cdiv(S, BLOCK_M)
    for m in range(num_m_blocks):
        offs_m = m * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S
        
        q_ptrs = Q + pid_b * stride_qb + pid_h * stride_qh + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
        do_ptrs = dO + pid_b * stride_dob + pid_h * stride_doh + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod
        o_ptrs = O + pid_b * stride_ob + pid_h * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
        
        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
        do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
        o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
        
        # Calculate pre-softmax logits S_ij = (Q_i @ K_j^T)
        acc_s = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        s = tl.dot(q, tl.trans(k), acc_s)
        s = s * scale
        
        l_ptrs = L + pid_b * stride_lb + pid_h * stride_lh + offs_m * stride_ls
        l = tl.load(l_ptrs, mask=mask_m, other=0.0)
        
        # Calculate P = exp(S_ij - L_i) with masking bounds
        p = tl.exp(s - l[:, None])
        p = tl.where((mask_m[:, None]) & (mask_n[None, :]), p, 0.0)
        
        # Calculate dP_i^j = dO_i @ V_j^T
        acc_dp = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        dp = tl.dot(do, tl.trans(v), acc_dp)
        
        # On-the-fly rowsum to avoid storing D on HBM 
        d_val = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
        
        # Calculate dS_i^j = P_i^j * (dP_i^j - D_i) * scale
        ds = (p * (dp - d_val[:, None]) * scale).to(q.dtype)
        p = p.to(q.dtype)
        
        # Accumulate dV_j += P_i^j^T @ dO_i
        dv = tl.dot(tl.trans(p), do, dv)
        # Accumulate dK_j += dS_i^j^T @ Q_i
        dk = tl.dot(tl.trans(ds), q, dk)
        
    dk_ptrs = dK + pid_b * stride_dkb + pid_h * stride_dkh + offs_n[:, None] * stride_dks + offs_d[None, :] * stride_dkd
    dv_ptrs = dV + pid_b * stride_dvb + pid_h * stride_dvh + offs_n[:, None] * stride_dvs + offs_d[None, :] * stride_dvd
    
    tl.store(dk_ptrs, dk.to(k.dtype), mask=mask_n[:, None])
    tl.store(dv_ptrs, dv.to(v.dtype), mask=mask_n[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Entry point for Multi-Head Attention Backward.
    Expects Q, K, V, O, dO (bf16), L (fp32 logsumexp).
    Stores results in pre-allocated dQ, dK, dV (bf16).
    """
    with torch.cuda.device(Q.device):
        B, H, S, d = Q.shape
        scale = 1.0 / (d ** 0.5)

        # Process dQ blocks
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

        # Process dK and dV blocks
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
            num_warps=8, num_stages=2
        )