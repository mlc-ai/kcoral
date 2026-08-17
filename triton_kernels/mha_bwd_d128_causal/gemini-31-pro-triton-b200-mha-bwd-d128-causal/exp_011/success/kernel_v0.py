import torch
import triton
import triton.language as tl

def get_autotune_configs_dq():
    return [
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
    ]

@triton.autotune(
    configs=get_autotune_configs_dq(),
    key=['S']
)
@triton.jit
def _bwd_dq_kernel(
    Q, K, V, O, dO, L, dQ,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    S, sm_scale, H,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    start_m = pid_m * BLOCK_M
    if start_m >= S:
        return

    batch_idx = pid_bh // H
    head_idx = pid_bh % H

    q_ptr = Q + batch_idx * stride_qb + head_idx * stride_qh
    k_ptr = K + batch_idx * stride_kb + head_idx * stride_kh
    v_ptr = V + batch_idx * stride_vb + head_idx * stride_vh
    o_ptr = O + batch_idx * stride_ob + head_idx * stride_oh
    do_ptr = dO + batch_idx * stride_dob + head_idx * stride_doh
    l_ptr = L + batch_idx * stride_lb + head_idx * stride_lh
    dq_ptr = dQ + batch_idx * stride_dqb + head_idx * stride_dqh

    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    mask_m = offs_m < S

    q_ptrs = q_ptr + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    o_ptrs = o_ptr + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    do_ptrs = do_ptr + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod
    l_ptrs = l_ptr + offs_m * stride_ls

    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
    do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
    l = tl.load(l_ptrs, mask=mask_m, other=0.0)

    delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)

    dq_acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    max_n = tl.minimum(S, start_m + BLOCK_M)
    n_steps = tl.cdiv(max_n, BLOCK_N)

    for n in range(n_steps):
        start_n = n * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S

        k_ptrs = k_ptr + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        v_ptrs = v_ptr + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd

        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

        qk_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        qk = tl.dot(q, k.T, acc=qk_acc) * sm_scale

        causal_mask = offs_m[:, None] >= offs_n[None, :]
        valid_mask = mask_m[:, None] & mask_n[None, :] & causal_mask
        qk = tl.where(valid_mask, qk, float("-inf"))

        p = tl.math.exp2((qk - l[:, None]) * 1.4426950408889634)
        p = tl.where(valid_mask, p, 0.0)

        dp_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        dp = tl.dot(do, v.T, acc=dp_acc)

        ds = p * (dp - delta[:, None]) * sm_scale
        ds = tl.where(valid_mask, ds, 0.0)

        dq_acc = tl.dot(ds.to(q.dtype), k, acc=dq_acc)

    dq_ptrs = dq_ptr + offs_m[:, None] * stride_dqs + offs_d[None, :] * stride_dqd
    tl.store(dq_ptrs, dq_acc.to(dq_ptr.dtype.element_ty), mask=mask_m[:, None])


def get_autotune_configs_dkv():
    return [
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
    ]

@triton.autotune(
    configs=get_autotune_configs_dkv(),
    key=['S']
)
@triton.jit
def _bwd_dkv_kernel(
    Q, K, V, O, dO, L, dK, dV,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    S, sm_scale, H,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)

    start_n = pid_n * BLOCK_N
    if start_n >= S:
        return

    batch_idx = pid_bh // H
    head_idx = pid_bh % H

    q_ptr = Q + batch_idx * stride_qb + head_idx * stride_qh
    k_ptr = K + batch_idx * stride_kb + head_idx * stride_kh
    v_ptr = V + batch_idx * stride_vb + head_idx * stride_vh
    o_ptr = O + batch_idx * stride_ob + head_idx * stride_oh
    do_ptr = dO + batch_idx * stride_dob + head_idx * stride_doh
    l_ptr = L + batch_idx * stride_lb + head_idx * stride_lh
    dk_ptr = dK + batch_idx * stride_dkb + head_idx * stride_dkh
    dv_ptr = dV + batch_idx * stride_dvb + head_idx * stride_dvh

    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    mask_n = offs_n < S

    k_ptrs = k_ptr + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = v_ptr + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd

    k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

    dk_acc = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    dv_acc = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)

    start_m_initial = (start_n // BLOCK_M) * BLOCK_M
    n_steps = tl.cdiv(S - start_m_initial, BLOCK_M)

    for m in range(n_steps):
        start_m = start_m_initial + m * BLOCK_M
        offs_m = start_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S
        
        q_ptrs = q_ptr + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
        o_ptrs = o_ptr + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
        do_ptrs = do_ptr + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod
        l_ptrs = l_ptr + offs_m * stride_ls

        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
        o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
        do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
        l = tl.load(l_ptrs, mask=mask_m, other=0.0)
        
        delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)

        qk_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        qk = tl.dot(q, k.T, acc=qk_acc) * sm_scale
        
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        valid_mask = mask_m[:, None] & mask_n[None, :] & causal_mask
        qk = tl.where(valid_mask, qk, float("-inf"))
        
        p = tl.math.exp2((qk - l[:, None]) * 1.4426950408889634)
        p = tl.where(valid_mask, p, 0.0)
        
        dp_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        dp = tl.dot(do, v.T, acc=dp_acc)
        
        ds = p * (dp - delta[:, None]) * sm_scale
        ds = tl.where(valid_mask, ds, 0.0)
        
        dk_acc = tl.dot(ds.T.to(q.dtype), q, acc=dk_acc)
        dv_acc = tl.dot(p.T.to(q.dtype), do, acc=dv_acc)

    dk_ptrs = dk_ptr + offs_n[:, None] * stride_dks + offs_d[None, :] * stride_dkd
    dv_ptrs = dv_ptr + offs_n[:, None] * stride_dvs + offs_d[None, :] * stride_dvd
    
    tl.store(dk_ptrs, dk_acc.to(dk_ptr.dtype.element_ty), mask=mask_n[:, None])
    tl.store(dv_ptrs, dv_acc.to(dv_ptr.dtype.element_ty), mask=mask_n[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes the backward pass for causal multi-head attention.
    Each result is written into the supplied output tensors without reallocation.
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, d = Q.shape
    sm_scale = 1.0 / (d ** 0.5)

    grid_dq = lambda META: (triton.cdiv(S, META['BLOCK_M']), B * H)
    _bwd_dq_kernel[grid_dq](
        Q, K, V, O, dO, L, dQ,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        S, sm_scale, H,
        BLOCK_D=128
    )

    grid_dkv = lambda META: (triton.cdiv(S, META['BLOCK_N']), B * H)
    _bwd_dkv_kernel[grid_dkv](
        Q, K, V, O, dO, L, dK, dV,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        S, sm_scale, H,
        BLOCK_D=128
    )