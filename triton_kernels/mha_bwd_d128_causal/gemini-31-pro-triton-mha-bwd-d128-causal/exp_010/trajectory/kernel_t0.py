import math
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
    B, H, S, sm_scale,
    d: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    start_m = pid_m * BLOCK_M
    if start_m >= S:
        return

    off_q = pid_b * stride_qb + pid_h * stride_qh
    off_k = pid_b * stride_kb + pid_h * stride_kh
    off_v = pid_b * stride_vb + pid_h * stride_vh
    off_o = pid_b * stride_ob + pid_h * stride_oh
    off_do = pid_b * stride_dob + pid_h * stride_doh
    off_l = pid_b * stride_lb + pid_h * stride_lh
    off_dq = pid_b * stride_dqb + pid_h * stride_dqh

    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, d)
    mask_m = offs_m < S

    # Pointers for the current row block of Q, O, dO, L
    q_ptrs = Q + off_q + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    o_ptrs = O + off_o + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    do_ptrs = dO + off_do + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod
    l_ptrs = L + off_l + offs_m * stride_ls

    q_i = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    o_i = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
    do_i = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
    l_i = tl.load(l_ptrs, mask=mask_m, other=0.0)

    # Compute rowsum(dO_i * O_i)
    Di = tl.sum(do_i.to(tl.float32) * o_i.to(tl.float32), axis=1)

    dq_acc = tl.zeros((BLOCK_M, d), dtype=tl.float32)

    # For causal attention, we only need to iterate over K and V blocks where n <= m.
    # The max row index in this block is start_m + BLOCK_M - 1.
    max_n = tl.minimum(S, start_m + BLOCK_M)
    num_steps = tl.cdiv(max_n, BLOCK_N)

    k_ptrs = K + off_k + tl.arange(0, BLOCK_N)[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = V + off_v + tl.arange(0, BLOCK_N)[:, None] * stride_vs + offs_d[None, :] * stride_vd

    for step in range(num_steps):
        start_n = step * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S

        k_j = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        v_j = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

        # S_ij = (Q_i @ K_j^T) * scale
        s_ij = tl.dot(q_i, tl.trans(k_j), out_dtype=tl.float32) * sm_scale

        causal_mask = offs_m[:, None] >= offs_n[None, :]
        valid_mask = mask_m[:, None] & mask_n[None, :] & causal_mask

        # Apply masking and exponentiate
        s_ij = tl.where(valid_mask, s_ij, float('-inf'))
        p_ij = tl.exp(s_ij - l_i[:, None])

        # dP_ij = dO_i @ V_j^T
        dp_ij = tl.dot(do_i, tl.trans(v_j), out_dtype=tl.float32)

        # dS_ij = P_ij * (dP_ij - D_i) * scale
        ds_ij = p_ij * (dp_ij - Di[:, None]) * sm_scale

        # dQ_i += dS_ij @ K_j
        dq_acc = tl.dot(ds_ij.to(tl.bfloat16), k_j, acc=dq_acc, out_dtype=tl.float32)

        k_ptrs += BLOCK_N * stride_ks
        v_ptrs += BLOCK_N * stride_vs

    dq_ptrs = dQ + off_dq + offs_m[:, None] * stride_dqs + offs_d[None, :] * stride_dqd
    tl.store(dq_ptrs, dq_acc.to(dQ.dtype.element_ty), mask=mask_m[:, None])


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
    B, H, S, sm_scale,
    d: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    start_n = pid_n * BLOCK_N
    if start_n >= S:
        return

    off_q = pid_b * stride_qb + pid_h * stride_qh
    off_k = pid_b * stride_kb + pid_h * stride_kh
    off_v = pid_b * stride_vb + pid_h * stride_vh
    off_o = pid_b * stride_ob + pid_h * stride_oh
    off_do = pid_b * stride_dob + pid_h * stride_doh
    off_l = pid_b * stride_lb + pid_h * stride_lh
    off_dk = pid_b * stride_dkb + pid_h * stride_dkh
    off_dv = pid_b * stride_dvb + pid_h * stride_dvh

    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, d)
    mask_n = offs_n < S

    # Load K_j and V_j
    k_ptrs = K + off_k + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = V + off_v + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd

    k_j = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
    v_j = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

    dk_acc = tl.zeros((BLOCK_N, d), dtype=tl.float32)
    dv_acc = tl.zeros((BLOCK_N, d), dtype=tl.float32)

    # For causal attention, we only need to iterate over Q blocks where m >= n.
    start_m_initial = (start_n // BLOCK_M) * BLOCK_M
    if start_m_initial < S:
        num_steps = tl.cdiv(S - start_m_initial, BLOCK_M)
    else:
        num_steps = 0

    q_ptrs = Q + off_q + (start_m_initial + tl.arange(0, BLOCK_M))[:, None] * stride_qs + offs_d[None, :] * stride_qd
    o_ptrs = O + off_o + (start_m_initial + tl.arange(0, BLOCK_M))[:, None] * stride_os + offs_d[None, :] * stride_od
    do_ptrs = dO + off_do + (start_m_initial + tl.arange(0, BLOCK_M))[:, None] * stride_dos + offs_d[None, :] * stride_dod
    l_ptrs = L + off_l + (start_m_initial + tl.arange(0, BLOCK_M)) * stride_ls

    for step in range(num_steps):
        start_m = start_m_initial + step * BLOCK_M
        offs_m = start_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S

        q_i = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
        o_i = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
        do_i = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
        l_i = tl.load(l_ptrs, mask=mask_m, other=0.0)

        # Compute rowsum(dO_i * O_i) on the fly
        Di = tl.sum(do_i.to(tl.float32) * o_i.to(tl.float32), axis=1)

        s_ij = tl.dot(q_i, tl.trans(k_j), out_dtype=tl.float32) * sm_scale

        causal_mask = offs_m[:, None] >= offs_n[None, :]
        valid_mask = mask_m[:, None] & mask_n[None, :] & causal_mask

        s_ij = tl.where(valid_mask, s_ij, float('-inf'))
        p_ij = tl.exp(s_ij - l_i[:, None])

        dp_ij = tl.dot(do_i, tl.trans(v_j), out_dtype=tl.float32)

        ds_ij = p_ij * (dp_ij - Di[:, None]) * sm_scale

        # dV_j += P_ij^T @ dO_i
        dv_acc = tl.dot(tl.trans(p_ij.to(tl.bfloat16)), do_i, acc=dv_acc, out_dtype=tl.float32)

        # dK_j += dS_ij^T @ Q_i
        dk_acc = tl.dot(tl.trans(ds_ij.to(tl.bfloat16)), q_i, acc=dk_acc, out_dtype=tl.float32)

        q_ptrs += BLOCK_M * stride_qs
        o_ptrs += BLOCK_M * stride_os
        do_ptrs += BLOCK_M * stride_dos
        l_ptrs += BLOCK_M * stride_ls

    dk_ptrs = dK + off_dk + offs_n[:, None] * stride_dks + offs_d[None, :] * stride_dkd
    dv_ptrs = dV + off_dv + offs_n[:, None] * stride_dvs + offs_d[None, :] * stride_dvd

    tl.store(dk_ptrs, dk_acc.to(dK.dtype.element_ty), mask=mask_n[:, None])
    tl.store(dv_ptrs, dv_acc.to(dV.dtype.element_ty), mask=mask_n[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes the backward pass of causal multi-head attention.
    Receives all input tensors in definition order, followed by preallocated output tensors.
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, d = Q.shape
    sm_scale = 1.0 / math.sqrt(d)
    
    BLOCK_M = 64
    BLOCK_N = 64
    
    grid_dq = (triton.cdiv(S, BLOCK_M), B, H)
    grid_dk_dv = (triton.cdiv(S, BLOCK_N), B, H)
    
    bwd_kernel_dq[grid_dq](
        Q, K, V, O, dO, L, dQ,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        B, H, S, sm_scale,
        d=d, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=4, num_stages=3
    )
    
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
        B, H, S, sm_scale,
        d=d, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=4, num_stages=3
    )