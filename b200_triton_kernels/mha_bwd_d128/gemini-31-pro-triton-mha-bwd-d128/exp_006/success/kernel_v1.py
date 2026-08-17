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
    S, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, d: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, d)
    mask_m = offs_m < S
    
    # Load Q, dO, O, and L for the current M block once (outer loop)
    q_ptrs = Q + pid_b * stride_qb + pid_h * stride_qh + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    do_ptrs = dO + pid_b * stride_dob + pid_h * stride_doh + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod
    o_ptrs = O + pid_b * stride_ob + pid_h * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    l_ptrs = L + pid_b * stride_lb + pid_h * stride_lh + offs_m * stride_ls
    
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
    o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
    l = tl.load(l_ptrs, mask=mask_m, other=0.0)
    
    # Precompute D = rowsum(dO * O) in fp32
    d_val = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
    
    dq = tl.zeros((BLOCK_M, d), tl.float32)
    
    k_ptrs = K + pid_b * stride_kb + pid_h * stride_kh + offs_d[None, :] * stride_kd
    v_ptrs = V + pid_b * stride_vb + pid_h * stride_vh + offs_d[None, :] * stride_vd
    
    num_n_blocks = tl.cdiv(S, BLOCK_N)
    for n in range(num_n_blocks):
        offs_n = n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S
        
        curr_k_ptrs = k_ptrs + offs_n[:, None] * stride_ks
        curr_v_ptrs = v_ptrs + offs_n[:, None] * stride_vs
        
        k = tl.load(curr_k_ptrs, mask=mask_n[:, None], other=0.0)
        v = tl.load(curr_v_ptrs, mask=mask_n[:, None], other=0.0)
        
        # Calculate pre-softmax logits S = Q @ K^T
        s = tl.dot(q, tl.trans(k))
        s = s * scale
        
        # Calculate P = exp(S - L)
        p = tl.exp(s - l[:, None])
        p = tl.where(mask_m[:, None] & mask_n[None, :], p, 0.0)
        
        # Calculate dP = dO @ V^T
        dp = tl.dot(do, tl.trans(v))
        
        # Calculate dS = P * (dP - D) * scale
        ds = p * (dp - d_val[:, None]) * scale
        ds_bf16 = ds.to(q.dtype)
        
        # Accumulate dQ += dS @ K
        dq = tl.dot(ds_bf16, k, dq)
        
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
    S, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, d: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, d)
    mask_n = offs_n < S
    
    # Load K and V for the current N block once (outer loop)
    k_ptrs = K + pid_b * stride_kb + pid_h * stride_kh + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = V + pid_b * stride_vb + pid_h * stride_vh + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
    
    k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
    
    dk = tl.zeros((BLOCK_N, d), tl.float32)
    dv = tl.zeros((BLOCK_N, d), tl.float32)
    
    q_ptrs = Q + pid_b * stride_qb + pid_h * stride_qh + offs_d[None, :] * stride_qd
    do_ptrs = dO + pid_b * stride_dob + pid_h * stride_doh + offs_d[None, :] * stride_dod
    o_ptrs = O + pid_b * stride_ob + pid_h * stride_oh + offs_d[None, :] * stride_od
    l_ptrs = L + pid_b * stride_lb + pid_h * stride_lh
    
    num_m_blocks = tl.cdiv(S, BLOCK_M)
    for m in range(num_m_blocks):
        offs_m = m * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S
        
        curr_q_ptrs = q_ptrs + offs_m[:, None] * stride_qs
        curr_do_ptrs = do_ptrs + offs_m[:, None] * stride_dos
        curr_o_ptrs = o_ptrs + offs_m[:, None] * stride_os
        curr_l_ptrs = l_ptrs + offs_m * stride_ls
        
        q = tl.load(curr_q_ptrs, mask=mask_m[:, None], other=0.0)
        do = tl.load(curr_do_ptrs, mask=mask_m[:, None], other=0.0)
        o = tl.load(curr_o_ptrs, mask=mask_m[:, None], other=0.0)
        l = tl.load(curr_l_ptrs, mask=mask_m, other=0.0)
        
        # Calculate D for this M block on the fly
        d_val = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
        
        # Calculate pre-softmax logits S = Q @ K^T
        s = tl.dot(q, tl.trans(k))
        s = s * scale
        
        # Calculate P = exp(S - L)
        p = tl.exp(s - l[:, None])
        p = tl.where(mask_m[:, None] & mask_n[None, :], p, 0.0)
        
        # Calculate dP = dO @ V^T
        dp = tl.dot(do, tl.trans(v))
        
        # Calculate dS = P * (dP - D) * scale
        ds = p * (dp - d_val[:, None]) * scale
        
        p_bf16 = p.to(q.dtype)
        ds_bf16 = ds.to(q.dtype)
        
        # Accumulate dV += P^T @ dO
        dv = tl.dot(tl.trans(p_bf16), do, dv)
        # Accumulate dK += dS^T @ Q
        dk = tl.dot(tl.trans(ds_bf16), q, dk)
        
    dk_ptrs = dK + pid_b * stride_dkb + pid_h * stride_dkh + offs_n[:, None] * stride_dks + offs_d[None, :] * stride_dkd
    dv_ptrs = dV + pid_b * stride_dvb + pid_h * stride_dvh + offs_n[:, None] * stride_dvs + offs_d[None, :] * stride_dvd
    
    tl.store(dk_ptrs, dk.to(k.dtype), mask=mask_n[:, None])
    tl.store(dv_ptrs, dv.to(v.dtype), mask=mask_n[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes FlashAttention-3 backward pass (destination-passing variant).
    """
    with torch.cuda.device(Q.device):
        B, H, S, d = Q.shape
        scale = 1.0 / (d ** 0.5)

        # 128 (inner) x 64 (outer) fits precisely into H100 CTA resource budgets 
        # (registers and SMEM) while exploiting WGMMA tensor cores.
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
            num_warps=4, num_stages=3
        )

        # For dK/dV, outer loop is N. 
        # Using M=64, N=128 ensures peak WGMMA tile shapes and stays safely within 
        # Hopper's 228KB CTA shared memory limit when buffering Q, dO, O for pipelining.
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
            num_warps=4, num_stages=2
        )