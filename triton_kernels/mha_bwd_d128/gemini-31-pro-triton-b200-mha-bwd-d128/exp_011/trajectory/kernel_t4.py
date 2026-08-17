import math
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
    ],
    key=['S'],
)
@triton.jit
def bwd_dq_kernel(
    Q, K, V, O, dO, L, dQ,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    scale, S,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    q_ptrs = Q + pid_b * stride_qb + pid_h * stride_qh + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    o_ptrs = O + pid_b * stride_ob + pid_h * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    do_ptrs = dO + pid_b * stride_dob + pid_h * stride_doh + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod
    l_ptrs = L + pid_b * stride_lb + pid_h * stride_lh + offs_m * stride_ls

    mask_m = offs_m < S
    mask_md = mask_m[:, None]

    # Load Q, O, dO once per query block
    q = tl.load(q_ptrs, mask=mask_md, other=0.0)
    o = tl.load(o_ptrs, mask=mask_md, other=0.0)
    do = tl.load(do_ptrs, mask=mask_md, other=0.0)
    l = tl.load(l_ptrs, mask=mask_m, other=0.0)

    # Precompute rowwise delta for this query block
    delta = tl.sum(do.to(tl.float32) * o.to(tl.float32), axis=1)
    
    # Initialize dQ accumulator
    dq = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # Set up pointers for KV loop
    k_ptrs = K + pid_b * stride_kb + pid_h * stride_kh + tl.arange(0, BLOCK_N)[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = V + pid_b * stride_vb + pid_h * stride_vh + tl.arange(0, BLOCK_N)[:, None] * stride_vs + offs_d[None, :] * stride_vd

    for n0 in range(tl.cdiv(S, BLOCK_N)):
        offs_n = n0 * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S
        mask_nd = mask_n[:, None]

        k = tl.load(k_ptrs, mask=mask_nd, other=0.0)
        v = tl.load(v_ptrs, mask=mask_nd, other=0.0)

        scores = tl.dot(q, k.T) * scale
        mask_mn = mask_m[:, None] & mask_n[None, :]
        scores = tl.where(mask_mn, scores, float("-inf"))

        p = tl.exp(scores - l[:, None])
        p = tl.where(mask_mn, p, 0.0)

        dp = tl.dot(do, v.T)
        ds = p * (dp - delta[:, None]) * scale
        ds = tl.where(mask_mn, ds, 0.0)

        dq = tl.dot(ds.to(q.dtype), k, dq)

        # Advance pointers
        k_ptrs += BLOCK_N * stride_ks
        v_ptrs += BLOCK_N * stride_vs

    dq_ptrs = dQ + pid_b * stride_dqb + pid_h * stride_dqh + offs_m[:, None] * stride_dqs + offs_d[None, :] * stride_dqd
    tl.store(dq_ptrs, dq.to(q.dtype), mask=mask_md)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
    ],
    key=['S'],
)
@triton.jit
def bwd_dkv_kernel(
    Q, K, V, O, dO, L, dK, dV,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    scale, S,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    mask_n = offs_n < S
    mask_nd = mask_n[:, None]

    k_ptrs = K + pid_b * stride_kb + pid_h * stride_kh + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = V + pid_b * stride_vb + pid_h * stride_vh + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd

    # Load K, V once per key/value block
    k = tl.load(k_ptrs, mask=mask_nd, other=0.0)
    v = tl.load(v_ptrs, mask=mask_nd, other=0.0)

    # Initialize accumulators
    dk = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)

    # Set up pointers for Q/O/dO/L loop
    q_ptrs = Q + pid_b * stride_qb + pid_h * stride_qh + tl.arange(0, BLOCK_M)[:, None] * stride_qs + offs_d[None, :] * stride_qd
    o_ptrs = O + pid_b * stride_ob + pid_h * stride_oh + tl.arange(0, BLOCK_M)[:, None] * stride_os + offs_d[None, :] * stride_od
    do_ptrs = dO + pid_b * stride_dob + pid_h * stride_doh + tl.arange(0, BLOCK_M)[:, None] * stride_dos + offs_d[None, :] * stride_dod
    l_ptrs = L + pid_b * stride_lb + pid_h * stride_lh + tl.arange(0, BLOCK_M) * stride_ls

    for m0 in range(tl.cdiv(S, BLOCK_M)):
        offs_m = m0 * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S
        mask_md = mask_m[:, None]

        q = tl.load(q_ptrs, mask=mask_md, other=0.0)
        o = tl.load(o_ptrs, mask=mask_md, other=0.0)
        do = tl.load(do_ptrs, mask=mask_md, other=0.0)
        l = tl.load(l_ptrs, mask=mask_m, other=0.0)

        delta = tl.sum(do.to(tl.float32) * o.to(tl.float32), axis=1)

        scores = tl.dot(q, k.T) * scale
        mask_mn = mask_m[:, None] & mask_n[None, :]
        scores = tl.where(mask_mn, scores, float("-inf"))

        p = tl.exp(scores - l[:, None])
        p = tl.where(mask_mn, p, 0.0)

        dp = tl.dot(do, v.T)
        ds = p * (dp - delta[:, None]) * scale
        ds = tl.where(mask_mn, ds, 0.0)

        dv = tl.dot(p.T.to(q.dtype), do, dv)
        dk = tl.dot(ds.T.to(q.dtype), q, dk)

        # Advance pointers
        q_ptrs += BLOCK_M * stride_qs
        o_ptrs += BLOCK_M * stride_os
        do_ptrs += BLOCK_M * stride_dos
        l_ptrs += BLOCK_M * stride_ls

    dk_ptrs = dK + pid_b * stride_dkb + pid_h * stride_dkh + offs_n[:, None] * stride_dks + offs_d[None, :] * stride_dkd
    dv_ptrs = dV + pid_b * stride_dvb + pid_h * stride_dvh + offs_n[:, None] * stride_dvs + offs_d[None, :] * stride_dvd

    tl.store(dk_ptrs, dk.to(k.dtype), mask=mask_nd)
    tl.store(dv_ptrs, dv.to(v.dtype), mask=mask_nd)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes SDPA backward pass (dQ, dK, dV) without atomics.
    Uses separate split-ownership kernels to natively parallelize math,
    leverage hardware FP32 accumulators, and maximize L2 cache reuse.
    """
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    scale = 1.0 / math.sqrt(d)

    # 1) Compute dQ tile by iterating over keys and values
    grid_dq = lambda META: (triton.cdiv(S, META['BLOCK_M']), H, B)
    bwd_dq_kernel[grid_dq](
        Q, K, V, O, dO, L, dQ,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        scale, S,
        BLOCK_D=d
    )

    # 2) Compute dK, dV tile by iterating over queries, forward outputs, and gradients
    grid_dkv = lambda META: (triton.cdiv(S, META['BLOCK_N']), H, B)
    bwd_dkv_kernel[grid_dkv](
        Q, K, V, O, dO, L, dK, dV,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        scale, S,
        BLOCK_D=d
    )