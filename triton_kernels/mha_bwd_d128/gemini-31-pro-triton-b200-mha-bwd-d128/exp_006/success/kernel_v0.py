import torch
import triton
import triton.language as tl

autotune_configs = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
]


@triton.autotune(configs=autotune_configs, key=['S'])
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
    S, d: tl.constexpr,
    softmax_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_d = tl.arange(0, d)
    mask_m = off_m < S
    
    # Q, O, dO pointers
    q_ptrs = Q + pid_b * stride_qb + pid_h * stride_qh + off_m[:, None] * stride_qs + off_d[None, :] * stride_qd
    o_ptrs = O + pid_b * stride_ob + pid_h * stride_oh + off_m[:, None] * stride_os + off_d[None, :] * stride_od
    do_ptrs = dO + pid_b * stride_dob + pid_h * stride_doh + off_m[:, None] * stride_dos + off_d[None, :] * stride_dod
    l_ptrs = L + pid_b * stride_lb + pid_h * stride_lh + off_m * stride_ls

    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
    do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
    l = tl.load(l_ptrs, mask=mask_m, other=0.0)

    # Compute rowwise delta: [BLOCK_M]
    delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)

    dq = tl.zeros((BLOCK_M, d), dtype=tl.float32)

    off_n = tl.arange(0, BLOCK_N)
    k_ptrs = K + pid_b * stride_kb + pid_h * stride_kh + off_n[:, None] * stride_ks + off_d[None, :] * stride_kd
    v_ptrs = V + pid_b * stride_vb + pid_h * stride_vh + off_n[:, None] * stride_vs + off_d[None, :] * stride_vd

    num_kv_tiles = tl.cdiv(S, BLOCK_N)
    for kv_idx in range(0, num_kv_tiles):
        curr_n = kv_idx * BLOCK_N + off_n
        mask_n = curr_n < S
        valid = mask_m[:, None] & mask_n[None, :]

        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

        # scores: [BLOCK_M, d] @ [d, BLOCK_N] -> [BLOCK_M, BLOCK_N]
        scores = tl.dot(q, k.T, out_dtype=tl.float32) * softmax_scale
        scores = tl.where(valid, scores, -float("inf"))
        
        p = tl.math.exp(scores - l[:, None])
        p = tl.where(valid, p, 0.0)
        
        # dp: [BLOCK_M, d] @ [d, BLOCK_N] -> [BLOCK_M, BLOCK_N]
        dp = tl.dot(do, v.T, out_dtype=tl.float32)
        
        ds = p * (dp - delta[:, None]) * softmax_scale
        ds = tl.where(valid, ds, 0.0)
        
        # ds: [BLOCK_M, BLOCK_N], k: [BLOCK_N, d] -> dq: [BLOCK_M, d]
        dq = tl.dot(ds.to(q.dtype), k, acc=dq)
        
        k_ptrs += BLOCK_N * stride_ks
        v_ptrs += BLOCK_N * stride_vs

    dq_ptrs = dQ + pid_b * stride_dqb + pid_h * stride_dqh + off_m[:, None] * stride_dqs + off_d[None, :] * stride_dqd
    tl.store(dq_ptrs, dq.to(q.dtype), mask=mask_m[:, None])


@triton.autotune(configs=autotune_configs, key=['S'])
@triton.jit
def bwd_dkdv_kernel(
    Q, K, V, O, dO, L, dK, dV,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    S, d: tl.constexpr,
    softmax_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    off_d = tl.arange(0, d)
    mask_n = off_n < S

    # K and V pointers for the resident KV tile
    k_ptrs = K + pid_b * stride_kb + pid_h * stride_kh + off_n[:, None] * stride_ks + off_d[None, :] * stride_kd
    v_ptrs = V + pid_b * stride_vb + pid_h * stride_vh + off_n[:, None] * stride_vs + off_d[None, :] * stride_vd
    
    k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

    dk = tl.zeros((BLOCK_N, d), dtype=tl.float32)
    dv = tl.zeros((BLOCK_N, d), dtype=tl.float32)

    off_m = tl.arange(0, BLOCK_M)
    q_ptrs = Q + pid_b * stride_qb + pid_h * stride_qh + off_m[:, None] * stride_qs + off_d[None, :] * stride_qd
    o_ptrs = O + pid_b * stride_ob + pid_h * stride_oh + off_m[:, None] * stride_os + off_d[None, :] * stride_od
    do_ptrs = dO + pid_b * stride_dob + pid_h * stride_doh + off_m[:, None] * stride_dos + off_d[None, :] * stride_dod
    l_ptrs = L + pid_b * stride_lb + pid_h * stride_lh + off_m * stride_ls

    num_q_tiles = tl.cdiv(S, BLOCK_M)
    for q_idx in range(0, num_q_tiles):
        curr_m = q_idx * BLOCK_M + off_m
        mask_m = curr_m < S
        valid = mask_n[:, None] & mask_m[None, :]

        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
        o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
        do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
        l = tl.load(l_ptrs, mask=mask_m, other=0.0)

        # Compute rowwise delta: [BLOCK_M]
        delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)

        # scores_t: [BLOCK_N, d] @ [d, BLOCK_M] -> [BLOCK_N, BLOCK_M]
        scores_t = tl.dot(k, q.T, out_dtype=tl.float32) * softmax_scale
        scores_t = tl.where(valid, scores_t, -float("inf"))

        p_t = tl.math.exp(scores_t - l[None, :])
        p_t = tl.where(valid, p_t, 0.0)

        # dv += p_t @ do (p_t is [BLOCK_N, BLOCK_M], do is [BLOCK_M, d])
        dv = tl.dot(p_t.to(q.dtype), do, acc=dv)

        # dp_t: [BLOCK_N, d] @ [d, BLOCK_M] -> [BLOCK_N, BLOCK_M]
        dp_t = tl.dot(v, do.T, out_dtype=tl.float32)

        ds_t = p_t * (dp_t - delta[None, :]) * softmax_scale
        ds_t = tl.where(valid, ds_t, 0.0)

        # dk += ds_t @ q (ds_t is [BLOCK_N, BLOCK_M], q is [BLOCK_M, d])
        dk = tl.dot(ds_t.to(q.dtype), q, acc=dk)

        q_ptrs += BLOCK_M * stride_qs
        o_ptrs += BLOCK_M * stride_os
        do_ptrs += BLOCK_M * stride_dos
        l_ptrs += BLOCK_M * stride_ls

    dk_ptrs = dK + pid_b * stride_dkb + pid_h * stride_dkh + off_n[:, None] * stride_dks + off_d[None, :] * stride_dkd
    dv_ptrs = dV + pid_b * stride_dvb + pid_h * stride_dvh + off_n[:, None] * stride_dvs + off_d[None, :] * stride_dvd

    tl.store(dk_ptrs, dk.to(k.dtype), mask=mask_n[:, None])
    tl.store(dv_ptrs, dv.to(v.dtype), mask=mask_n[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Compute destination-passing multi-head attention backward without atomics.
    """
    with torch.cuda.device(Q.device):
        B, H, S, d = Q.shape
        softmax_scale = 1.0 / (d ** 0.5)

        # Dispatch dQ reduction kernel
        grid_dq = lambda META: (triton.cdiv(S, META['BLOCK_M']), B, H)
        bwd_dq_kernel[grid_dq](
            Q, K, V, O, dO, L, dQ,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
            L.stride(0), L.stride(1), L.stride(2),
            dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
            S, d, softmax_scale
        )

        # Dispatch dK/dV reduction kernel
        grid_dkdv = lambda META: (triton.cdiv(S, META['BLOCK_N']), B, H)
        bwd_dkdv_kernel[grid_dkdv](
            Q, K, V, O, dO, L, dK, dV,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
            L.stride(0), L.stride(1), L.stride(2),
            dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
            dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
            S, d, softmax_scale
        )