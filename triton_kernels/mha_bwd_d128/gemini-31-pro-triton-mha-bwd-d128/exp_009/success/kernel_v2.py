import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
    ],
    key=['seqlen_q']
)
@triton.jit
def bwd_kernel_dq(
    Q, K, V, L, O, dO, dQ,
    sm_scale,
    seqlen_q,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    stride_lb, stride_lh, stride_ls,
    H,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D_HEAD: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    b_idx = pid_bh // H
    h_idx = pid_bh % H
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D_HEAD)
    
    mask_m = offs_m < seqlen_q
    mask_m_2d = mask_m[:, None]
    
    # Base pointers for the current M block
    q_ptrs = Q + b_idx * stride_qb + h_idx * stride_qh + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    do_ptrs = dO + b_idx * stride_dob + h_idx * stride_doh + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod
    o_ptrs = O + b_idx * stride_ob + h_idx * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    l_ptrs = L + b_idx * stride_lb + h_idx * stride_lh + offs_m * stride_ls
    
    # Load Q, dO, O, and L for this M block (outside the loop)
    q = tl.load(q_ptrs, mask=mask_m_2d, other=0.0)
    do = tl.load(do_ptrs, mask=mask_m_2d, other=0.0)
    o = tl.load(o_ptrs, mask=mask_m_2d, other=0.0)
    l_i = tl.load(l_ptrs, mask=mask_m, other=0.0)
    
    # Precompute D_i = sum(O_i * dO_i, dim=-1) for the current M block
    D_i = tl.sum(tl.cast(o, tl.float32) * tl.cast(do, tl.float32), axis=1)
    
    # Initialize dQ accumulator
    dq = tl.zeros([BLOCK_M, D_HEAD], dtype=tl.float32)
    
    # Starting pointers for the N loop
    offs_n = tl.arange(0, BLOCK_N)
    k_ptrs = K + b_idx * stride_kb + h_idx * stride_kh + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = V + b_idx * stride_vb + h_idx * stride_vh + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
    
    # Pipelined loop over N (K and V blocks)
    for start_n in tl.range(0, seqlen_q, BLOCK_N):
        mask_n = (start_n + offs_n) < seqlen_q
        mask_n_2d = mask_n[:, None]
        
        k = tl.load(k_ptrs, mask=mask_n_2d, other=0.0)
        v = tl.load(v_ptrs, mask=mask_n_2d, other=0.0)
        
        # S_ij = Q_i @ K_j^T
        qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * sm_scale
        
        # P_ij = exp(S_ij - L_i)
        p = tl.exp(qk - l_i[:, None])
        mask_mn = mask_m[:, None] & mask_n[None, :]
        p = tl.where(mask_mn, p, 0.0)
        
        # dP_ij = dO_i @ V_j^T
        dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32)
        
        # dS_ij = P_ij * (dP_ij - D_i)
        ds = p * (dp - D_i[:, None]) * sm_scale
        ds = tl.where(mask_mn, ds, 0.0)
        
        # dQ_i += dS_ij @ K_j
        dq += tl.dot(ds.to(tl.bfloat16), k, out_dtype=tl.float32)
        
        # Advance pointers
        k_ptrs += BLOCK_N * stride_ks
        v_ptrs += BLOCK_N * stride_vs
        
    # Store dQ
    dq_ptrs = dQ + b_idx * stride_dqb + h_idx * stride_dqh + offs_m[:, None] * stride_dqs + offs_d[None, :] * stride_dqd
    tl.store(dq_ptrs, dq.to(tl.bfloat16), mask=mask_m_2d)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 128, 'BLOCK_M': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_N': 64, 'BLOCK_M': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_N': 128, 'BLOCK_M': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_N': 64, 'BLOCK_M': 64}, num_stages=4, num_warps=4),
    ],
    key=['seqlen_q']
)
@triton.jit
def bwd_kernel_dk_dv(
    Q, K, V, L, O, dO, dK, dV,
    sm_scale,
    seqlen_q,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    stride_lb, stride_lh, stride_ls,
    H,
    BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr, D_HEAD: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    b_idx = pid_bh // H
    h_idx = pid_bh % H
    
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D_HEAD)
    
    mask_n = offs_n < seqlen_q
    mask_n_2d = mask_n[:, None]
    
    # Load K and V for this N block (outside the loop)
    k_ptrs = K + b_idx * stride_kb + h_idx * stride_kh + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = V + b_idx * stride_vb + h_idx * stride_vh + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
    
    k = tl.load(k_ptrs, mask=mask_n_2d, other=0.0)
    v = tl.load(v_ptrs, mask=mask_n_2d, other=0.0)
    
    # Initialize dK and dV accumulators
    dk = tl.zeros([BLOCK_N, D_HEAD], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, D_HEAD], dtype=tl.float32)
    
    # Starting pointers for the M loop
    offs_m = tl.arange(0, BLOCK_M)
    q_ptrs = Q + b_idx * stride_qb + h_idx * stride_qh + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    do_ptrs = dO + b_idx * stride_dob + h_idx * stride_doh + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod
    o_ptrs = O + b_idx * stride_ob + h_idx * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    l_ptrs = L + b_idx * stride_lb + h_idx * stride_lh + offs_m * stride_ls
    
    # Pipelined loop over M (Q, dO, O blocks)
    for start_m in tl.range(0, seqlen_q, BLOCK_M):
        mask_m = (start_m + offs_m) < seqlen_q
        mask_m_2d = mask_m[:, None]
        
        q = tl.load(q_ptrs, mask=mask_m_2d, other=0.0)
        do = tl.load(do_ptrs, mask=mask_m_2d, other=0.0)
        o = tl.load(o_ptrs, mask=mask_m_2d, other=0.0)
        l_i = tl.load(l_ptrs, mask=mask_m, other=0.0)
        
        # Recompute D_i for the current M block
        D_i = tl.sum(tl.cast(o, tl.float32) * tl.cast(do, tl.float32), axis=1)
        
        # S_ij = Q_i @ K_j^T
        qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * sm_scale
        
        # P_ij = exp(S_ij - L_i)
        p = tl.exp(qk - l_i[:, None])
        mask_mn = mask_m[:, None] & mask_n[None, :]
        p = tl.where(mask_mn, p, 0.0)
        
        # dV_j += P_ij^T @ dO_i
        dv += tl.dot(tl.trans(p.to(tl.bfloat16)), do, out_dtype=tl.float32)
        
        # dP_ij = dO_i @ V_j^T
        dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32)
        
        # dS_ij = P_ij * (dP_ij - D_i)
        ds = p * (dp - D_i[:, None]) * sm_scale
        ds = tl.where(mask_mn, ds, 0.0)
        
        # dK_j += dS_ij^T @ Q_i
        dk += tl.dot(tl.trans(ds.to(tl.bfloat16)), q, out_dtype=tl.float32)
        
        # Advance pointers
        q_ptrs += BLOCK_M * stride_qs
        do_ptrs += BLOCK_M * stride_dos
        o_ptrs += BLOCK_M * stride_os
        l_ptrs += BLOCK_M * stride_ls
        
    # Store dK and dV
    dk_ptrs = dK + b_idx * stride_dkb + h_idx * stride_dkh + offs_n[:, None] * stride_dks + offs_d[None, :] * stride_dkd
    dv_ptrs = dV + b_idx * stride_dvb + h_idx * stride_dvh + offs_n[:, None] * stride_dvs + offs_d[None, :] * stride_dvd
    
    tl.store(dk_ptrs, dk.to(tl.bfloat16), mask=mask_n_2d)
    tl.store(dv_ptrs, dv.to(tl.bfloat16), mask=mask_n_2d)

def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes the backward pass for multi-head attention.
    Destination-passing semantics: overwrites pre-allocated dQ, dK, dV tensors in-place.
    """
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape

    if S == 0:
        return

    sm_scale = 1.0 / (d ** 0.5)

    stride_lb = L.stride(0)
    stride_lh = L.stride(1)
    stride_ls = L.stride(2)

    # 1) Compute dQ (grid spans M blocks)
    grid_dq = lambda META: (triton.cdiv(S, META['BLOCK_M']), B * H)
    bwd_kernel_dq[grid_dq](
        Q, K, V, L, O, dO, dQ,
        sm_scale,
        S,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        stride_lb, stride_lh, stride_ls,
        H,
        D_HEAD=d
    )

    # 2) Compute dK and dV (grid spans N blocks)
    grid_dkdv = lambda META: (triton.cdiv(S, META['BLOCK_N']), B * H)
    bwd_kernel_dk_dv[grid_dkdv](
        Q, K, V, L, O, dO, dK, dV,
        sm_scale,
        S,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        stride_lb, stride_lh, stride_ls,
        H,
        D_HEAD=d
    )