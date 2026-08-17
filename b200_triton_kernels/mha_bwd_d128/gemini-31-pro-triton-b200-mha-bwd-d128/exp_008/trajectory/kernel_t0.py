import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=3, num_warps=8),
    ],
    key=['seq_len']
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
    seq_len,
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    batch = pid_bh // 48
    head = pid_bh % 48

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    mask_m = offs_m < seq_len

    q_ptrs = Q + batch * stride_qb + head * stride_qh + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    o_ptrs = O + batch * stride_ob + head * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    do_ptrs = dO + batch * stride_dob + head * stride_doh + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod
    l_ptrs = L + batch * stride_lb + head * stride_lh + offs_m * stride_ls

    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
    do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
    l = tl.load(l_ptrs, mask=mask_m, other=0.0)

    # Compute rowwise delta = sum(dO * O, axis=1) in FP32
    do_fp32 = tl.cast(do, tl.float32)
    o_fp32 = tl.cast(o, tl.float32)
    delta = tl.sum(do_fp32 * o_fp32, axis=1)

    dq = tl.zeros((BLOCK_M, D), dtype=tl.float32)
    
    k_ptrs_base = K + batch * stride_kb + head * stride_kh
    v_ptrs_base = V + batch * stride_vb + head * stride_vh

    num_kv_blocks = tl.cdiv(seq_len, BLOCK_N)
    for n_idx in range(0, num_kv_blocks):
        offs_n = n_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < seq_len

        k_ptrs = k_ptrs_base + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        v_ptrs = v_ptrs_base + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd

        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

        s = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        s = tl.dot(q, tl.trans(k), s)
        s = s * sm_scale

        valid = mask_m[:, None] & mask_n[None, :]
        s = tl.where(valid, s, float('-inf'))

        p = tl.math.exp(s - l[:, None])
        p = tl.where(valid, p, 0.0)

        dp = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        dp = tl.dot(do, tl.trans(v), dp)

        ds = p * (dp - delta[:, None]) * sm_scale
        ds = tl.where(valid, ds, 0.0)
        ds_bf16 = tl.cast(ds, tl.bfloat16)

        dq = tl.dot(ds_bf16, k, dq)

    dq_ptrs = dQ + batch * stride_dqb + head * stride_dqh + offs_m[:, None] * stride_dqs + offs_d[None, :] * stride_dqd
    tl.store(dq_ptrs, tl.cast(dq, tl.bfloat16), mask=mask_m[:, None])


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=3, num_warps=8),
    ],
    key=['seq_len']
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
    seq_len,
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)

    batch = pid_bh // 48
    head = pid_bh % 48

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    mask_n = offs_n < seq_len

    k_ptrs = K + batch * stride_kb + head * stride_kh + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = V + batch * stride_vb + head * stride_vh + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd

    k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

    dk = tl.zeros((BLOCK_N, D), dtype=tl.float32)
    dv = tl.zeros((BLOCK_N, D), dtype=tl.float32)

    q_ptrs_base = Q + batch * stride_qb + head * stride_qh
    do_ptrs_base = dO + batch * stride_dob + head * stride_doh
    o_ptrs_base = O + batch * stride_ob + head * stride_oh
    l_ptrs_base = L + batch * stride_lb + head * stride_lh

    num_q_blocks = tl.cdiv(seq_len, BLOCK_M)
    for m_idx in range(0, num_q_blocks):
        offs_m = m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < seq_len

        q_ptrs = q_ptrs_base + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
        do_ptrs = do_ptrs_base + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod
        o_ptrs = o_ptrs_base + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
        l_ptrs = l_ptrs_base + offs_m * stride_ls

        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
        do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
        o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
        l = tl.load(l_ptrs, mask=mask_m, other=0.0)

        # Compute delta locally per query block streamed in memory
        do_fp32 = tl.cast(do, tl.float32)
        o_fp32 = tl.cast(o, tl.float32)
        delta = tl.sum(do_fp32 * o_fp32, axis=1)

        s = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        s = tl.dot(q, tl.trans(k), s)
        s = s * sm_scale

        valid = mask_m[:, None] & mask_n[None, :]
        s = tl.where(valid, s, float('-inf'))

        p = tl.math.exp(s - l[:, None])
        p = tl.where(valid, p, 0.0)

        dp = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        dp = tl.dot(do, tl.trans(v), dp)

        ds = p * (dp - delta[:, None]) * sm_scale
        ds = tl.where(valid, ds, 0.0)
        ds_bf16 = tl.cast(ds, tl.bfloat16)
        p_bf16 = tl.cast(p, tl.bfloat16)

        dk = tl.dot(tl.trans(ds_bf16), q, dk)
        dv = tl.dot(tl.trans(p_bf16), do, dv)

    dk_ptrs = dK + batch * stride_dkb + head * stride_dkh + offs_n[:, None] * stride_dks + offs_d[None, :] * stride_dkd
    dv_ptrs = dV + batch * stride_dvb + head * stride_dvh + offs_n[:, None] * stride_dvs + offs_d[None, :] * stride_dvd

    tl.store(dk_ptrs, tl.cast(dk, tl.bfloat16), mask=mask_n[:, None])
    tl.store(dv_ptrs, tl.cast(dv, tl.bfloat16), mask=mask_n[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Standard standard-Triton non-causal multi-head attention backward kernel implementation.
    Receives preallocated tensors in definition order. Uses two passes to gracefully segregate gradients.
    """
    torch.cuda.set_device(Q.device)
    
    # Grid Shape setup: B, H, S, d
    B, H, S, d = Q.shape
    sm_scale = 1.0 / (d ** 0.5)
    
    grid_dq = lambda META: (triton.cdiv(S, META['BLOCK_M']), B * H)
    bwd_dq_kernel[grid_dq](
        Q, K, V, O, dO, L, dQ,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        seq_len=S, sm_scale=sm_scale, D=d
    )

    grid_dkv = lambda META: (triton.cdiv(S, META['BLOCK_N']), B * H)
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
        seq_len=S, sm_scale=sm_scale, D=d
    )