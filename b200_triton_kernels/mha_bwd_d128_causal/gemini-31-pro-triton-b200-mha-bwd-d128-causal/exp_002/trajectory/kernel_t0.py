import torch
import triton
import triton.language as tl

@triton.jit
def bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dQ_ptr,
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
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, d)
    
    q_base = Q_ptr + b * stride_qb + h * stride_qh
    o_base = O_ptr + b * stride_ob + h * stride_oh
    do_base = dO_ptr + b * stride_dob + h * stride_doh
    
    q_ptrs = q_base + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    o_ptrs = o_base + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    do_ptrs = do_base + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod
    
    l_base = L_ptr + b * stride_lb + h * stride_lh
    l_ptrs = l_base + offs_m * stride_ls

    mask_m = offs_m < S
    mask_md = mask_m[:, None]

    # Load outer-loop structures 
    q = tl.load(q_ptrs, mask=mask_md, other=0.0)
    o = tl.load(o_ptrs, mask=mask_md, other=0.0).to(tl.float32)
    do = tl.load(do_ptrs, mask=mask_md, other=0.0)
    
    # Precompute factor D for this Q-block dynamically
    D = tl.sum(o * do.to(tl.float32), axis=1)
    L = tl.load(l_ptrs, mask=mask_m, other=0.0)

    acc_dq = tl.zeros((BLOCK_M, d), dtype=tl.float32)

    k_base = K_ptr + b * stride_kb + h * stride_kh
    v_base = V_ptr + b * stride_vb + h * stride_vh
    
    offs_n_init = tl.arange(0, BLOCK_N)
    k_ptrs = k_base + offs_n_init[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = v_base + offs_n_init[:, None] * stride_vs + offs_d[None, :] * stride_vd
    
    max_j = tl.minimum(pid_m * BLOCK_M + BLOCK_M, S)
    num_blocks = (max_j + BLOCK_N - 1) // BLOCK_N

    for n_idx in tl.range(0, num_blocks, num_stages=3):
        current_n = n_idx * BLOCK_N + offs_n_init
        mask_n = current_n < S
        mask_nd = mask_n[:, None]
        
        k = tl.load(k_ptrs, mask=mask_nd, other=0.0)
        v = tl.load(v_ptrs, mask=mask_nd, other=0.0)
        
        s_ij = tl.dot(q, k.T, out_dtype=tl.float32) * scale
        
        causal_mask = current_n[None, :] <= offs_m[:, None]
        valid_mask = causal_mask & mask_n[None, :] & mask_m[:, None]
        
        p_ij = tl.exp(s_ij - L[:, None])
        p_ij = tl.where(valid_mask, p_ij, 0.0)
        
        dp_ij = tl.dot(do, v.T, out_dtype=tl.float32)
        
        ds_ij = p_ij * (dp_ij - D[:, None])
        ds_ij_scaled = ds_ij * scale
        ds_ij_scaled_bf16 = ds_ij_scaled.to(tl.bfloat16)
        
        acc_dq = tl.dot(ds_ij_scaled_bf16, k, acc_dq)
        
        k_ptrs += BLOCK_N * stride_ks
        v_ptrs += BLOCK_N * stride_vs

    dq_base = dQ_ptr + b * stride_dqb + h * stride_dqh
    dq_ptrs = dq_base + offs_m[:, None] * stride_dqs + offs_d[None, :] * stride_dqd
    tl.store(dq_ptrs, acc_dq.to(tl.bfloat16), mask=mask_md)


@triton.jit
def bwd_dk_dv_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dK_ptr, dV_ptr,
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
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, d)
    
    k_base = K_ptr + b * stride_kb + h * stride_kh
    v_base = V_ptr + b * stride_vb + h * stride_vh
    
    k_ptrs = k_base + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = v_base + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
    
    mask_n = offs_n < S
    mask_nd = mask_n[:, None]
    
    k = tl.load(k_ptrs, mask=mask_nd, other=0.0)
    v = tl.load(v_ptrs, mask=mask_nd, other=0.0)
    
    acc_dk = tl.zeros((BLOCK_N, d), dtype=tl.float32)
    acc_dv = tl.zeros((BLOCK_N, d), dtype=tl.float32)
    
    q_base = Q_ptr + b * stride_qb + h * stride_qh
    o_base = O_ptr + b * stride_ob + h * stride_oh
    do_base = dO_ptr + b * stride_dob + h * stride_doh
    l_base = L_ptr + b * stride_lb + h * stride_lh
    
    offs_m_init = tl.arange(0, BLOCK_M)
    
    start_m_idx = (pid_n * BLOCK_N) // BLOCK_M
    num_m_blocks = (S + BLOCK_M - 1) // BLOCK_M
    
    q_ptrs = q_base + (start_m_idx * BLOCK_M + offs_m_init)[:, None] * stride_qs + offs_d[None, :] * stride_qd
    o_ptrs = o_base + (start_m_idx * BLOCK_M + offs_m_init)[:, None] * stride_os + offs_d[None, :] * stride_od
    do_ptrs = do_base + (start_m_idx * BLOCK_M + offs_m_init)[:, None] * stride_dos + offs_d[None, :] * stride_dod
    l_ptrs = l_base + (start_m_idx * BLOCK_M + offs_m_init) * stride_ls
    
    for m_idx in tl.range(start_m_idx, num_m_blocks, num_stages=3):
        current_m = m_idx * BLOCK_M + offs_m_init
        mask_m = current_m < S
        mask_md = mask_m[:, None]
        
        q = tl.load(q_ptrs, mask=mask_md, other=0.0)
        o = tl.load(o_ptrs, mask=mask_md, other=0.0).to(tl.float32)
        do = tl.load(do_ptrs, mask=mask_md, other=0.0)
        L = tl.load(l_ptrs, mask=mask_m, other=0.0)
        
        # Precompute factor D for this temporal section natively dynamically
        D = tl.sum(o * do.to(tl.float32), axis=1) 
        
        s_ij = tl.dot(q, k.T, out_dtype=tl.float32) * scale
        
        causal_mask = offs_n[None, :] <= current_m[:, None]
        valid_mask = causal_mask & mask_n[None, :] & mask_m[:, None]
        
        p_ij = tl.exp(s_ij - L[:, None])
        p_ij = tl.where(valid_mask, p_ij, 0.0)
        
        p_ij_bf16 = p_ij.to(tl.bfloat16)
        acc_dv = tl.dot(p_ij_bf16.T, do, acc_dv)
        
        dp_ij = tl.dot(do, v.T, out_dtype=tl.float32)
        
        ds_ij = p_ij * (dp_ij - D[:, None])
        ds_ij_scaled = ds_ij * scale
        ds_ij_scaled_bf16 = ds_ij_scaled.to(tl.bfloat16)
        
        acc_dk = tl.dot(ds_ij_scaled_bf16.T, q, acc_dk)
        
        q_ptrs += BLOCK_M * stride_qs
        o_ptrs += BLOCK_M * stride_os
        do_ptrs += BLOCK_M * stride_dos
        l_ptrs += BLOCK_M * stride_ls

    dk_base = dK_ptr + b * stride_dkb + h * stride_dkh
    dv_base = dV_ptr + b * stride_dvb + h * stride_dvh
    
    dk_ptrs = dk_base + offs_n[:, None] * stride_dks + offs_d[None, :] * stride_dkd
    dv_ptrs = dv_base + offs_n[:, None] * stride_dvs + offs_d[None, :] * stride_dvd
    
    tl.store(dk_ptrs, acc_dk.to(tl.bfloat16), mask=mask_nd)
    tl.store(dv_ptrs, acc_dv.to(tl.bfloat16), mask=mask_nd)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Triton entry point for backward multi-head causal attention evaluating gradients w.r.t Q, K, and V.
    Leverages Blackwell SM100 limits appropriately.
    """
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    scale = 1.0 / (d ** 0.5)

    # Sanitize inputs optionally carrying unneeded unsqueezed inner dimensions
    if L.dim() == 4:
        L = L.squeeze(-1)

    BLOCK_M = 128
    BLOCK_N = 128
    
    grid_dq = (triton.cdiv(S, BLOCK_M), B * H)
    bwd_dq_kernel[grid_dq](
        Q, K, V, O, dO, L, dQ,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        S, scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, d=128,
        num_warps=8, num_stages=3
    )

    grid_dk_dv = (triton.cdiv(S, BLOCK_N), B * H)
    bwd_dk_dv_kernel[grid_dk_dv](
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
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, d=128,
        num_warps=8, num_stages=3
    )