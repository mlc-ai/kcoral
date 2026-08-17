import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
    ],
    key=['S']
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
    B, H, S, scale,
    d: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, d)

    mask_m = offs_m < S

    # Pointers to the Q tile
    q_ptrs = Q + b * stride_qb + h * stride_qh + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    o_ptrs = O + b * stride_ob + h * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    do_ptrs = dO + b * stride_dob + h * stride_doh + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod
    l_ptrs = L + b * stride_lb + h * stride_lh + offs_m * stride_ls

    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
    do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
    l = tl.load(l_ptrs, mask=mask_m, other=0.0)

    # Precompute rowwise delta for the Q tile
    delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)

    dq_acc = tl.zeros((BLOCK_M, d), dtype=tl.float32)

    num_kv_tiles = tl.cdiv(S, BLOCK_N)
    LOG2_E = 1.4426950408889634

    for kv_tile in range(num_kv_tiles):
        kv_offs_n = kv_tile * BLOCK_N + offs_n
        mask_n = kv_offs_n < S
        
        k_ptrs = K + b * stride_kb + h * stride_kh + kv_offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        v_ptrs = V + b * stride_vb + h * stride_vh + kv_offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
        
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        
        scores = tl.dot(q, k.trans(1, 0)) * scale
        
        valid = mask_m[:, None] & mask_n[None, :]
        scores = tl.where(valid, scores, float("-inf"))
        
        p = tl.math.exp2((scores - l[:, None]) * LOG2_E)
        # Suppress NaNs from fully masked rows
        p = tl.where(valid, p, 0.0)
        
        dp = tl.dot(do, v.trans(1, 0))
        
        ds = p * (dp - delta[:, None]) * scale
        ds = tl.where(valid, ds, 0.0)
        
        dq_acc += tl.dot(ds.to(q.dtype), k)

    dq_ptrs = dQ + b * stride_dqb + h * stride_dqh + offs_m[:, None] * stride_dqs + offs_d[None, :] * stride_dqd
    tl.store(dq_ptrs, dq_acc.to(dQ.dtype.element_ty), mask=mask_m[:, None])


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
    ],
    key=['S']
)
@triton.jit
def bwd_dk_dv_kernel(
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
    d: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, d)

    mask_n = offs_n < S

    # Pointers to the KV tile
    k_ptrs = K + b * stride_kb + h * stride_kh + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = V + b * stride_vb + h * stride_vh + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd

    k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

    dk_acc = tl.zeros((BLOCK_N, d), dtype=tl.float32)
    dv_acc = tl.zeros((BLOCK_N, d), dtype=tl.float32)

    num_q_tiles = tl.cdiv(S, BLOCK_M)
    LOG2_E = 1.4426950408889634

    for q_tile in range(num_q_tiles):
        q_offs_m = q_tile * BLOCK_M + offs_m
        mask_m = q_offs_m < S
        
        q_ptrs = Q + b * stride_qb + h * stride_qh + q_offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
        o_ptrs = O + b * stride_ob + h * stride_oh + q_offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
        do_ptrs = dO + b * stride_dob + h * stride_doh + q_offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod
        l_ptrs = L + b * stride_lb + h * stride_lh + q_offs_m * stride_ls

        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
        o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
        do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
        l = tl.load(l_ptrs, mask=mask_m, other=0.0)

        # Precompute rowwise delta for this Q tile
        delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)

        scores = tl.dot(k, q.trans(1, 0)) * scale

        valid = mask_n[:, None] & mask_m[None, :]
        scores = tl.where(valid, scores, float("-inf"))

        p = tl.math.exp2((scores - l[None, :]) * LOG2_E)
        # Suppress NaNs from fully masked rows
        p = tl.where(valid, p, 0.0)

        dv_acc += tl.dot(p.to(q.dtype), do)

        dp_t = tl.dot(v, do.trans(1, 0))

        ds_t = p * (dp_t - delta[None, :]) * scale
        ds_t = tl.where(valid, ds_t, 0.0)

        dk_acc += tl.dot(ds_t.to(q.dtype), q)

    dk_ptrs = dK + b * stride_dkb + h * stride_dkh + offs_n[:, None] * stride_dks + offs_d[None, :] * stride_dkd
    dv_ptrs = dV + b * stride_dvb + h * stride_dvh + offs_n[:, None] * stride_dvs + offs_d[None, :] * stride_dvd

    tl.store(dk_ptrs, dk_acc.to(dK.dtype.element_ty), mask=mask_n[:, None])
    tl.store(dv_ptrs, dv_acc.to(dV.dtype.element_ty), mask=mask_n[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes backward gradients for multi-head attention without causal masking.
    
    Operates on bf16 tensors with shape [B, H, S, d]. Output dQ, dK, and dV are fully stored
    using an exclusive ownership split without requiring atomic accumulations.
    """
    torch.cuda.set_device(Q.device)
    B_val, H_val, S_val, d_val = Q.shape
    
    # Scale is 1/sqrt(d)
    scale = 1.0 / (d_val ** 0.5)

    # Calculate LSE stride along sequence length
    stride_ls = L.stride(2) if L.dim() >= 3 else L.stride(-1)

    # Launch Region 1: Q-tile owners calculate dQ
    def grid_dq(META):
        return (triton.cdiv(S_val, META['BLOCK_M']), B_val * H_val)

    bwd_dq_kernel[grid_dq](
        Q, K, V, O, dO, L, dQ,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        L.stride(0), L.stride(1), stride_ls,
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        B_val, H_val, S_val, scale,
        d=128
    )

    # Launch Region 2: KV-tile owners calculate dK and dV
    def grid_dk(META):
        return (triton.cdiv(S_val, META['BLOCK_N']), B_val * H_val)

    bwd_dk_dv_kernel[grid_dk](
        Q, K, V, O, dO, L, dK, dV,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        L.stride(0), L.stride(1), stride_ls,
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        B_val, H_val, S_val, scale,
        d=128
    )