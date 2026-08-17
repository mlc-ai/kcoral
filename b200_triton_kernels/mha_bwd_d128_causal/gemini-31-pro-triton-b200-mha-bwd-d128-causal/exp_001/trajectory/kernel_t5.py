import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
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
    S,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, d: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)

    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, d)

    q_ptrs = Q + pid_b * stride_qb + pid_h * stride_qh + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    o_ptrs = O + pid_b * stride_ob + pid_h * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    do_ptrs = dO + pid_b * stride_dob + pid_h * stride_doh + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod
    l_ptrs = L + pid_b * stride_lb + pid_h * stride_lh + offs_m * stride_ls

    mask_m = offs_m < S
    
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
    do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
    l = tl.load(l_ptrs, mask=mask_m, other=0.0)

    # Precalculate sum(O * dO, dim=-1) for the current Q block
    Di = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
    dq = tl.zeros((BLOCK_M, d), dtype=tl.float32)
    scale = 0.08838834764831845  # 1.0 / sqrt(128)

    max_n = start_m + BLOCK_M
    if max_n > S:
        max_n = S

    # Calculate limit for completely off-diagonal blocks to skip causal masking
    limit_n = (start_m // BLOCK_N) * BLOCK_N
    if limit_n > max_n:
        limit_n = max_n

    k_ptrs = K + pid_b * stride_kb + pid_h * stride_kh + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = V + pid_b * stride_vb + pid_h * stride_vh + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd

    # 1. Off-diagonal K/V blocks (fully below the diagonal)
    for start_n in range(0, limit_n, BLOCK_N):
        curr_n = start_n + offs_n
        mask_n = curr_n < S
        
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

        s = tl.dot(q, k.T, out_dtype=tl.float32) * scale
        
        # Only boundary checks required, no causal masking
        mask = mask_m[:, None] & mask_n[None, :]
        s = tl.where(mask, s, float('-inf'))
        p = tl.exp(s - l[:, None])

        ds = tl.dot(do, v.T, out_dtype=tl.float32)
        dp = p * (ds - Di[:, None]) * scale

        dq += tl.dot(dp.to(q.dtype), k, out_dtype=tl.float32)
        
        k_ptrs += BLOCK_N * stride_ks
        v_ptrs += BLOCK_N * stride_vs

    # 2. Diagonal K/V blocks (intersects the causal boundary)
    for start_n in range(limit_n, max_n, BLOCK_N):
        curr_n = start_n + offs_n
        mask_n = curr_n < S
        
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

        s = tl.dot(q, k.T, out_dtype=tl.float32) * scale
        
        causal_mask = offs_m[:, None] >= curr_n[None, :]
        mask = causal_mask & mask_m[:, None] & mask_n[None, :]
        s = tl.where(mask, s, float('-inf'))

        p = tl.exp(s - l[:, None])

        ds = tl.dot(do, v.T, out_dtype=tl.float32)
        dp = p * (ds - Di[:, None]) * scale

        dq += tl.dot(dp.to(q.dtype), k, out_dtype=tl.float32)
        
        k_ptrs += BLOCK_N * stride_ks
        v_ptrs += BLOCK_N * stride_vs

    dq_ptrs = dQ + pid_b * stride_dqb + pid_h * stride_dqh + offs_m[:, None] * stride_dqs + offs_d[None, :] * stride_dqd
    tl.store(dq_ptrs, dq.to(q.dtype), mask=mask_m[:, None])


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
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
    S,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, d: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)

    start_n = pid_n * BLOCK_N
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, d)
    mask_n = offs_n < S

    k_ptrs = K + pid_b * stride_kb + pid_h * stride_kh + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = V + pid_b * stride_vb + pid_h * stride_vh + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd

    k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

    dk = tl.zeros((BLOCK_N, d), dtype=tl.float32)
    dv = tl.zeros((BLOCK_N, d), dtype=tl.float32)
    scale = 0.08838834764831845  # 1.0 / sqrt(128)

    # Causal masking: Q blocks with M < N can't see this K/V block, so start M at the causal boundary
    start_m_block = (start_n // BLOCK_M) * BLOCK_M
    if start_m_block < 0:
        start_m_block = 0
        
    limit_m = start_n + BLOCK_N
    if limit_m > S:
        limit_m = S

    q_ptrs = Q + pid_b * stride_qb + pid_h * stride_qh + (start_m_block + offs_m)[:, None] * stride_qs + offs_d[None, :] * stride_qd
    o_ptrs = O + pid_b * stride_ob + pid_h * stride_oh + (start_m_block + offs_m)[:, None] * stride_os + offs_d[None, :] * stride_od
    do_ptrs = dO + pid_b * stride_dob + pid_h * stride_doh + (start_m_block + offs_m)[:, None] * stride_dos + offs_d[None, :] * stride_dod
    l_ptrs = L + pid_b * stride_lb + pid_h * stride_lh + (start_m_block + offs_m) * stride_ls

    # 1. Diagonal Q blocks (requires causal mask checks)
    for curr_m in range(start_m_block, limit_m, BLOCK_M):
        mask_m = (curr_m + offs_m) < S
        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
        o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
        do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
        l_val = tl.load(l_ptrs, mask=mask_m, other=0.0)

        Di = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)

        # Transpose implicitly handled natively by Tensor Cores.
        s_T = tl.dot(k, q.T, out_dtype=tl.float32) * scale
        
        causal_mask = (curr_m + offs_m)[None, :] >= offs_n[:, None]
        mask = causal_mask & mask_m[None, :] & mask_n[:, None]
        s_T = tl.where(mask, s_T, float('-inf'))

        p_T = tl.exp(s_T - l_val[None, :])

        dv += tl.dot(p_T.to(v.dtype), do, out_dtype=tl.float32)

        ds_T = tl.dot(v, do.T, out_dtype=tl.float32)
        dp_T = p_T * (ds_T - Di[None, :]) * scale

        dk += tl.dot(dp_T.to(k.dtype), q, out_dtype=tl.float32)

        q_ptrs += BLOCK_M * stride_qs
        o_ptrs += BLOCK_M * stride_os
        do_ptrs += BLOCK_M * stride_dos
        l_ptrs += BLOCK_M * stride_ls

    # 2. Off-diagonal Q blocks (no causal mask check required)
    for curr_m in range(limit_m, S, BLOCK_M):
        mask_m = (curr_m + offs_m) < S
        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
        o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
        do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
        l_val = tl.load(l_ptrs, mask=mask_m, other=0.0)

        Di = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)

        s_T = tl.dot(k, q.T, out_dtype=tl.float32) * scale
        
        mask = mask_m[None, :] & mask_n[:, None]
        s_T = tl.where(mask, s_T, float('-inf'))

        p_T = tl.exp(s_T - l_val[None, :])

        dv += tl.dot(p_T.to(v.dtype), do, out_dtype=tl.float32)

        ds_T = tl.dot(v, do.T, out_dtype=tl.float32)
        dp_T = p_T * (ds_T - Di[None, :]) * scale

        dk += tl.dot(dp_T.to(k.dtype), q, out_dtype=tl.float32)

        q_ptrs += BLOCK_M * stride_qs
        o_ptrs += BLOCK_M * stride_os
        do_ptrs += BLOCK_M * stride_dos
        l_ptrs += BLOCK_M * stride_ls

    dk_ptrs = dK + pid_b * stride_dkb + pid_h * stride_dkh + offs_n[:, None] * stride_dks + offs_d[None, :] * stride_dkd
    dv_ptrs = dV + pid_b * stride_dvb + pid_h * stride_dvh + offs_n[:, None] * stride_dvs + offs_d[None, :] * stride_dvd

    tl.store(dk_ptrs, dk.to(k.dtype), mask=mask_n[:, None])
    tl.store(dv_ptrs, dv.to(v.dtype), mask=mask_n[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes causal multi-head attention backward pass natively on NVIDIA Blackwell.
    Results are written into preallocated dQ, dK, dV output buffers.
    """
    with torch.cuda.device(Q.device):
        B, H, S, d = Q.shape
        lse = L.view(B, H, S)
        
        grid_dq = lambda META: (triton.cdiv(S, META['BLOCK_M']), H, B)
        bwd_dq_kernel[grid_dq](
            Q, K, V, O, dO, lse, dQ,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
            S, d=d
        )

        grid_dk_dv = lambda META: (triton.cdiv(S, META['BLOCK_N']), H, B)
        bwd_dk_dv_kernel[grid_dk_dv](
            Q, K, V, O, dO, lse, dK, dV,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
            dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
            S, d=d
        )
        
        return dQ, dK, dV