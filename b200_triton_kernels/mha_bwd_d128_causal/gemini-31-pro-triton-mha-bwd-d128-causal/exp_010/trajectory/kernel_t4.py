import math
import torch
import triton
import triton.language as tl

def get_dq_configs():
    return [
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
    ]

def get_dkdv_configs():
    return [
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
    ]

@triton.jit
def precompute_D_kernel(
    O, dO, dQ,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    S, d: tl.constexpr, BLOCK_M: tl.constexpr
):
    """
    Precomputes D_i = sum(dO_i * O_i) to avoid O(N^2) HBM reads of O_i in the dk_dv kernel.
    Safely stages the float32 result into the first elements of the uninitialized bfloat16 dQ tensor,
    which bwd_kernel_dk_dv will read, before bwd_kernel_dq overwrites dQ completely.
    """
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    start_m = pid_m * BLOCK_M
    if start_m >= S:
        return

    off_o = pid_b * stride_ob + pid_h * stride_oh
    off_do = pid_b * stride_dob + pid_h * stride_doh
    off_dq = pid_b * stride_dqb + pid_h * stride_dqh

    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, d)
    mask_m = offs_m < S

    o_ptrs = O + off_o + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    do_ptrs = dO + off_do + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod

    o_i = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
    do_i = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)

    d_i = tl.sum(do_i.to(tl.float32) * o_i.to(tl.float32), axis=1)

    dq_ptrs = dQ + off_dq + offs_m * stride_dqs
    # Stage into the 4 bytes available per bfloat16 row head
    dq_ptrs_f32 = dq_ptrs.to(tl.pointer_type(tl.float32))
    tl.store(dq_ptrs_f32, d_i, mask=mask_m)


@triton.autotune(configs=get_dkdv_configs(), key=['S'])
@triton.jit
def bwd_kernel_dk_dv(
    Q, K, V, dO, L, dQ_for_D, dK, dV,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
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
    off_do = pid_b * stride_dob + pid_h * stride_doh
    off_dq = pid_b * stride_dqb + pid_h * stride_dqh
    off_l = pid_b * stride_lb + pid_h * stride_lh
    off_dk = pid_b * stride_dkb + pid_h * stride_dkh
    off_dv = pid_b * stride_dvb + pid_h * stride_dvh

    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, d)
    mask_n = offs_n < S

    k_ptrs = K + off_k + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = V + off_v + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd

    k_j = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
    v_j = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

    dk_acc = tl.zeros((BLOCK_N, d), dtype=tl.float32)
    dv_acc = tl.zeros((BLOCK_N, d), dtype=tl.float32)

    # For causal attention, we only iterate Q blocks where m >= n
    start_m_initial = (start_n // BLOCK_M) * BLOCK_M
    if start_m_initial < S:
        num_steps = tl.cdiv(S - start_m_initial, BLOCK_M)
    else:
        num_steps = 0

    q_ptrs = Q + off_q + (start_m_initial + tl.arange(0, BLOCK_M))[:, None] * stride_qs + offs_d[None, :] * stride_qd
    do_ptrs = dO + off_do + (start_m_initial + tl.arange(0, BLOCK_M))[:, None] * stride_dos + offs_d[None, :] * stride_dod
    l_ptrs = L + off_l + (start_m_initial + tl.arange(0, BLOCK_M)) * stride_ls
    
    # bf16 pointer tracking
    dq_ptrs = dQ_for_D + off_dq + (start_m_initial + tl.arange(0, BLOCK_M)) * stride_dqs

    for step in tl.range(num_steps):
        start_m = start_m_initial + step * BLOCK_M
        offs_m = start_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S

        q_i = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
        do_i = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
        l_i = tl.load(l_ptrs, mask=mask_m, other=0.0)
        
        # Cast to fp32 strictly on the loaded pointer iteration
        dq_ptrs_f32 = dq_ptrs.to(tl.pointer_type(tl.float32))
        d_i = tl.load(dq_ptrs_f32, mask=mask_m, other=0.0)

        s_ij = tl.dot(q_i, tl.trans(k_j), out_dtype=tl.float32) * sm_scale

        is_causal_step = (start_m < start_n + BLOCK_N)
        if is_causal_step:
            valid = offs_m[:, None] >= offs_n[None, :]
            s_ij = tl.where(valid, s_ij, float('-inf'))

        if start_m + BLOCK_M > S or start_n + BLOCK_N > S:
            valid_bnd = (offs_m[:, None] < S) & (offs_n[None, :] < S)
            s_ij = tl.where(valid_bnd, s_ij, float('-inf'))

        p_ij = tl.exp(s_ij - l_i[:, None])

        dp_ij = tl.dot(do_i, tl.trans(v_j), out_dtype=tl.float32)
        ds_ij = p_ij * (dp_ij - d_i[:, None]) * sm_scale

        dv_acc = tl.dot(tl.trans(p_ij.to(tl.bfloat16)), do_i, acc=dv_acc, out_dtype=tl.float32)
        dk_acc = tl.dot(tl.trans(ds_ij.to(tl.bfloat16)), q_i, acc=dk_acc, out_dtype=tl.float32)

        q_ptrs += BLOCK_M * stride_qs
        do_ptrs += BLOCK_M * stride_dos
        l_ptrs += BLOCK_M * stride_ls
        dq_ptrs += BLOCK_M * stride_dqs

    dk_out_ptrs = dK + off_dk + offs_n[:, None] * stride_dks + offs_d[None, :] * stride_dkd
    dv_out_ptrs = dV + off_dv + offs_n[:, None] * stride_dvs + offs_d[None, :] * stride_dvd

    tl.store(dk_out_ptrs, dk_acc.to(dK.dtype.element_ty), mask=mask_n[:, None])
    tl.store(dv_out_ptrs, dv_acc.to(dV.dtype.element_ty), mask=mask_n[:, None])


@triton.autotune(configs=get_dq_configs(), key=['S'])
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

    q_ptrs = Q + off_q + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    o_ptrs = O + off_o + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    do_ptrs = dO + off_do + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod
    l_ptrs = L + off_l + offs_m * stride_ls

    q_i = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    o_i = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
    do_i = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
    l_i = tl.load(l_ptrs, mask=mask_m, other=0.0)

    # Compute D_i on the fly; no need to read from memory staging here since O_i is fetched once.
    d_i = tl.sum(do_i.to(tl.float32) * o_i.to(tl.float32), axis=1)

    dq_acc = tl.zeros((BLOCK_M, d), dtype=tl.float32)

    k_ptrs = K + off_k + tl.arange(0, BLOCK_N)[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = V + off_v + tl.arange(0, BLOCK_N)[:, None] * stride_vs + offs_d[None, :] * stride_vd

    max_n = tl.minimum(S, start_m + BLOCK_M)
    num_steps = tl.cdiv(max_n, BLOCK_N)

    for step in tl.range(num_steps):
        start_n = step * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S

        k_j = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        v_j = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

        s_ij = tl.dot(q_i, tl.trans(k_j), out_dtype=tl.float32) * sm_scale

        is_causal_step = (start_n + BLOCK_N > start_m)
        if is_causal_step:
            valid = offs_m[:, None] >= offs_n[None, :]
            s_ij = tl.where(valid, s_ij, float('-inf'))

        if start_m + BLOCK_M > S or start_n + BLOCK_N > S:
            valid_bnd = (offs_m[:, None] < S) & (offs_n[None, :] < S)
            s_ij = tl.where(valid_bnd, s_ij, float('-inf'))

        p_ij = tl.exp(s_ij - l_i[:, None])
        dp_ij = tl.dot(do_i, tl.trans(v_j), out_dtype=tl.float32)
        ds_ij = p_ij * (dp_ij - d_i[:, None]) * sm_scale

        dq_acc = tl.dot(ds_ij.to(tl.bfloat16), k_j, acc=dq_acc, out_dtype=tl.float32)

        k_ptrs += BLOCK_N * stride_ks
        v_ptrs += BLOCK_N * stride_vs

    dq_out_ptrs = dQ + off_dq + offs_m[:, None] * stride_dqs + offs_d[None, :] * stride_dqd
    tl.store(dq_out_ptrs, dq_acc.to(dQ.dtype.element_ty), mask=mask_m[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes the backward pass of causal multi-head attention.
    Receives all input tensors in definition order, followed by preallocated output tensors.
    """
    torch.cuda.set_device(Q.device)

    B, H, S, d = Q.shape
    sm_scale = float(1.0 / math.sqrt(d))
    
    # 1. Precompute `D_i = rowsum(dO * O)` and stage it safely in the memory
    # space of the dQ tensor, to be consumed by the inner loop of bwd_kernel_dk_dv.
    # We do this to avoid an O(N^2) memory read for O in the dk_dv kernel.
    grid_precompute = (triton.cdiv(S, 128), B, H)
    precompute_D_kernel[grid_precompute](
        O, dO, dQ,
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        S, d=d, BLOCK_M=128
    )
    
    # 2. Reconstruct `dK` and `dV`. Sequential CUDA stream execution safely prevents
    # reading the partially written dQ prior to its final overwriting.
    grid_dk_dv = lambda META: (triton.cdiv(S, META['BLOCK_N']), B, H)
    bwd_kernel_dk_dv[grid_dk_dv](
        Q, K, V, dO, L, dQ, dK, dV,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        B, H, S, sm_scale,
        d=d
    )
    
    # 3. Compute `dQ` overriding the same elements loaded temporarily during execution.
    grid_dq = lambda META: (triton.cdiv(S, META['BLOCK_M']), B, H)
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
        d=d
    )