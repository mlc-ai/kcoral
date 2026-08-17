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
    B: tl.constexpr, H: tl.constexpr, d: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
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

    # Compute row-wise dot product of O and dO in FP32
    o_f32 = o.to(tl.float32)
    do_f32 = do.to(tl.float32)
    D_i = tl.sum(o_f32 * do_f32, axis=1)

    dq = tl.zeros((BLOCK_SIZE, d), dtype=tl.float32)
    scale = 0.08838834764831845  # 1.0 / sqrt(128)

    max_n = (pid_m + 1) * BLOCK_SIZE
    if max_n > S:
        max_n = S
    num_steps = (max_n + BLOCK_SIZE - 1) // BLOCK_SIZE

    for step in range(num_steps):
        start_n = step * BLOCK_SIZE
        offs_n = start_n + tl.arange(0, BLOCK_SIZE)
        mask_n = offs_n < S

        k_ptrs = K + pid_b * stride_kb + pid_h * stride_kh + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        v_ptrs = V + pid_b * stride_vb + pid_h * stride_vh + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd

        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

        s = tl.dot(q, k.T, out_dtype=tl.float32) * scale
        
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        mask = causal_mask & mask_m[:, None] & mask_n[None, :]
        s = tl.where(mask, s, float('-inf'))

        p = tl.exp(s - l[:, None])

        ds = tl.dot(do, v.T, out_dtype=tl.float32)
        dp = p * (ds - D_i[:, None]) * scale

        dq += tl.dot(dp.to(q.dtype), k, out_dtype=tl.float32)

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
    B: tl.constexpr, H: tl.constexpr, d: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_d = tl.arange(0, d)
    mask_n = offs_n < S

    k_ptrs = K + pid_b * stride_kb + pid_h * stride_kh + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = V + pid_b * stride_vb + pid_h * stride_vh + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd

    k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

    dk = tl.zeros((BLOCK_SIZE, d), dtype=tl.float32)
    dv = tl.zeros((BLOCK_SIZE, d), dtype=tl.float32)

    scale = 0.08838834764831845  # 1.0 / sqrt(128)

    num_q_steps = (S + BLOCK_SIZE - 1) // BLOCK_SIZE
    start_step = pid_n
    
    for step in range(start_step, num_q_steps):
        start_m = step * BLOCK_SIZE
        offs_m = start_m + tl.arange(0, BLOCK_SIZE)
        mask_m = offs_m < S

        q_ptrs = Q + pid_b * stride_qb + pid_h * stride_qh + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
        o_ptrs = O + pid_b * stride_ob + pid_h * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
        do_ptrs = dO + pid_b * stride_dob + pid_h * stride_doh + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod
        l_ptrs = L + pid_b * stride_lb + pid_h * stride_lh + offs_m * stride_ls

        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
        o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
        do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
        l = tl.load(l_ptrs, mask=mask_m, other=0.0)

        o_f32 = o.to(tl.float32)
        do_f32 = do.to(tl.float32)
        D_i = tl.sum(o_f32 * do_f32, axis=1)

        s = tl.dot(q, k.T, out_dtype=tl.float32) * scale
        
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        mask = causal_mask & mask_m[:, None] & mask_n[None, :]
        s = tl.where(mask, s, float('-inf'))

        p = tl.exp(s - l[:, None])

        dv += tl.dot(p.to(q.dtype).T, do, out_dtype=tl.float32)

        ds = tl.dot(do, v.T, out_dtype=tl.float32)
        dp = p * (ds - D_i[:, None]) * scale

        dk += tl.dot(dp.to(q.dtype).T, q, out_dtype=tl.float32)

    dk_ptrs = dK + pid_b * stride_dkb + pid_h * stride_dkh + offs_n[:, None] * stride_dks + offs_d[None, :] * stride_dkd
    dv_ptrs = dV + pid_b * stride_dvb + pid_h * stride_dvh + offs_n[:, None] * stride_dvs + offs_d[None, :] * stride_dvd

    tl.store(dk_ptrs, dk.to(k.dtype), mask=mask_n[:, None])
    tl.store(dv_ptrs, dv.to(v.dtype), mask=mask_n[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    with torch.cuda.device(Q.device):
        B, H, S, d = Q.shape
        BLOCK_SIZE = 64

        grid_dq = (triton.cdiv(S, BLOCK_SIZE), H, B)
        bwd_dq_kernel[grid_dq](
            Q, K, V, O, dO, L, dQ,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
            L.stride(0), L.stride(1), L.stride(2),
            dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
            S,
            B=B, H=H, d=d,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
            num_stages=3
        )

        grid_dk_dv = (triton.cdiv(S, BLOCK_SIZE), H, B)
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
            S,
            B=B, H=H, d=d,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
            num_stages=3
        )