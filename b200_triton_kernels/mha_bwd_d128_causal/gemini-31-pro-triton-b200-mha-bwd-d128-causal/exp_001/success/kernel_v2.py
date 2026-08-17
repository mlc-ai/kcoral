import torch
import triton
import triton.language as tl

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

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
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

    # Precompute row-wise dot product of O and dO in FP32
    Di = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)

    dq = tl.zeros((BLOCK_M, d), dtype=tl.float32)
    scale = 0.08838834764831845  # 1.0 / sqrt(128)

    max_n = (pid_m + 1) * BLOCK_M
    if max_n > S:
        max_n = S

    num_steps = (max_n + BLOCK_N - 1) // BLOCK_N
    if num_steps < 0:
        num_steps = 0

    k_ptrs = K + pid_b * stride_kb + pid_h * stride_kh + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = V + pid_b * stride_vb + pid_h * stride_vh + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd

    for step in range(num_steps):
        start_n = step * BLOCK_N
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

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
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

    # Causal masking: Q elements with M < N can't see this K/V block, so start M at the causal boundary
    start_m = (pid_n * BLOCK_N // BLOCK_M) * BLOCK_M
    if start_m < 0:
        start_m = 0
        
    num_steps = (S - start_m + BLOCK_M - 1) // BLOCK_M
    if num_steps < 0:
        num_steps = 0
    
    q_ptrs = Q + pid_b * stride_qb + pid_h * stride_qh + (start_m + offs_m)[:, None] * stride_qs + offs_d[None, :] * stride_qd
    o_ptrs = O + pid_b * stride_ob + pid_h * stride_oh + (start_m + offs_m)[:, None] * stride_os + offs_d[None, :] * stride_od
    do_ptrs = dO + pid_b * stride_dob + pid_h * stride_doh + (start_m + offs_m)[:, None] * stride_dos + offs_d[None, :] * stride_dod
    l_ptrs = L + pid_b * stride_lb + pid_h * stride_lh + (start_m + offs_m) * stride_ls

    for step in range(num_steps):
        curr_m = start_m + step * BLOCK_M
        mask_m = (curr_m + offs_m) < S

        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
        o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
        do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
        l = tl.load(l_ptrs, mask=mask_m, other=0.0)

        Di = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)

        s_T = tl.dot(k, q.T, out_dtype=tl.float32) * scale
        
        causal_mask = (curr_m + offs_m)[None, :] >= offs_n[:, None]
        mask = causal_mask & mask_n[:, None] & mask_m[None, :]
        s_T = tl.where(mask, s_T, float('-inf'))

        p_T = tl.exp(s_T - l[None, :])

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
    Computes causal multi-head attention backward pass.
    Results are written into dQ, dK, dV.
    """
    with torch.cuda.device(Q.device):
        B, H, S, d = Q.shape
        
        # Ensure correct shape for L (B, H, S)
        lse = L.view(B, H, S)

        # Launch parameters tuned to fit well within B200 228 KB SM shared memory
        bwd_dq_kernel[(triton.cdiv(S, 128), H, B)](
            Q, K, V, O, dO, lse, dQ,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
            S,
            BLOCK_M=128, BLOCK_N=64, d=d,
            num_warps=8, num_stages=3
        )

        bwd_dk_dv_kernel[(triton.cdiv(S, 128), H, B)](
            Q, K, V, O, dO, lse, dK, dV,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
            dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
            S,
            BLOCK_M=64, BLOCK_N=128, d=d,
            num_warps=8, num_stages=3
        )
        
        return dQ, dK, dV