import torch
import triton
import triton.language as tl
import math


@triton.jit
def _dq_kernel(
    q_ptr, k_ptr, v_ptr, do_ptr, o_ptr, l_ptr, dq_ptr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lb, stride_lh, stride_ls,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    S, D, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    mask_m = offs_m[:, None] < S
    mask_d = offs_d[None, :] < D
    mask_md = mask_m & mask_d

    q_base = pid_b * stride_qb + pid_h * stride_qh
    k_base = pid_b * stride_kb + pid_h * stride_kh
    v_base = pid_b * stride_vb + pid_h * stride_vh
    do_base = pid_b * stride_dob + pid_h * stride_doh
    o_base = pid_b * stride_ob + pid_h * stride_oh
    l_base = pid_b * stride_lb + pid_h * stride_lh
    dq_base = pid_b * stride_dqb + pid_h * stride_dqh

    Q_tile = tl.load(q_ptr + q_base + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd,
                     mask=mask_md, other=0.0).to(tl.float32)
    dO_tile = tl.load(do_ptr + do_base + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod,
                      mask=mask_md, other=0.0).to(tl.float32)
    O_tile = tl.load(o_ptr + o_base + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od,
                     mask=mask_md, other=0.0).to(tl.float32)
    L_vals = tl.load(l_ptr + l_base + offs_m * stride_ls, mask=(offs_m < S), other=0.0)
    D_vals = tl.sum(dO_tile * O_tile, axis=1)

    acc_dq = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    num_n_blocks = tl.cdiv(S, BLOCK_N)
    for pi in range(num_n_blocks):
        offs_n = pi * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S

        K_tile = tl.load(k_ptr + k_base + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd,
                         mask=mask_n[:, None] & mask_d, other=0.0).to(tl.float32)
        V_tile = tl.load(v_ptr + v_base + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd,
                         mask=mask_n[:, None] & mask_d, other=0.0).to(tl.float32)

        scores = tl.dot(Q_tile, K_tile.T) * scale
        causal_mask = (offs_n[None, :] <= offs_m[:, None])
        attn_mask = causal_mask & mask_n[None, :] & mask_m
        P = tl.where(attn_mask, tl.exp(scores - L_vals[:, None]), 0.0)

        dP = tl.dot(dO_tile, V_tile.T)
        dS = P * (dP - D_vals[:, None]) * scale
        acc_dq = tl.dot(dS, K_tile, acc=acc_dq)

    tl.store(dq_ptr + dq_base + offs_m[:, None] * stride_dqs + offs_d[None, :] * stride_dqd,
             acc_dq.to(tl.bfloat16), mask=mask_md)


@triton.jit
def _dk_kernel(
    q_ptr, k_ptr, v_ptr, do_ptr, o_ptr, l_ptr, dk_ptr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lb, stride_lh, stride_ls,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    S, D, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    mask_n = offs_n[:, None] < S
    mask_d = offs_d[None, :] < D
    mask_nd = mask_n & mask_d

    q_base = pid_b * stride_qb + pid_h * stride_qh
    k_base = pid_b * stride_kb + pid_h * stride_kh
    v_base = pid_b * stride_vb + pid_h * stride_vh
    do_base = pid_b * stride_dob + pid_h * stride_doh
    o_base = pid_b * stride_ob + pid_h * stride_oh
    l_base = pid_b * stride_lb + pid_h * stride_lh
    dk_base = pid_b * stride_dkb + pid_h * stride_dkh

    K_tile = tl.load(k_ptr + k_base + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd,
                     mask=mask_nd, other=0.0).to(tl.float32)
    V_tile = tl.load(v_ptr + v_base + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd,
                     mask=mask_nd, other=0.0).to(tl.float32)

    acc_dk = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)

    num_m_blocks = tl.cdiv(S, BLOCK_M)
    for pm in range(num_m_blocks):
        offs_m = pm * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m[:, None] < S

        Q_tile = tl.load(q_ptr + q_base + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd,
                         mask=mask_m & mask_d, other=0.0).to(tl.float32)
        dO_tile = tl.load(do_ptr + do_base + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod,
                          mask=mask_m & mask_d, other=0.0).to(tl.float32)
        O_tile = tl.load(o_ptr + o_base + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od,
                         mask=mask_m & mask_d, other=0.0).to(tl.float32)
        L_vals = tl.load(l_ptr + l_base + offs_m * stride_ls, mask=(offs_m < S), other=0.0)
        D_vals = tl.sum(dO_tile * O_tile, axis=1)

        scores = tl.dot(Q_tile, K_tile.T) * scale
        causal_mask = (offs_n[None, :] <= offs_m[:, None])
        attn_mask = causal_mask & mask_n.T & mask_m
        P = tl.where(attn_mask, tl.exp(scores - L_vals[:, None]), 0.0)

        dP = tl.dot(dO_tile, V_tile.T)
        dS = P * (dP - D_vals[:, None]) * scale
        acc_dk = tl.dot(dS.T, Q_tile, acc=acc_dk)

    tl.store(dk_ptr + dk_base + offs_n[:, None] * stride_dks + offs_d[None, :] * stride_dkd,
             acc_dk.to(tl.bfloat16), mask=mask_nd)


@triton.jit
def _dv_kernel(
    q_ptr, k_ptr, v_ptr, do_ptr, o_ptr, l_ptr, dv_ptr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lb, stride_lh, stride_ls,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    S, D, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    mask_n = offs_n[:, None] < S
    mask_d = offs_d[None, :] < D
    mask_nd = mask_n & mask_d

    q_base = pid_b * stride_qb + pid_h * stride_qh
    k_base = pid_b * stride_kb + pid_h * stride_kh
    v_base = pid_b * stride_vb + pid_h * stride_vh
    do_base = pid_b * stride_dob + pid_h * stride_doh
    o_base = pid_b * stride_ob + pid_h * stride_oh
    l_base = pid_b * stride_lb + pid_h * stride_lh
    dv_base = pid_b * stride_dvb + pid_h * stride_dvh

    V_tile = tl.load(v_ptr + v_base + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd,
                     mask=mask_nd, other=0.0).to(tl.float32)

    acc_dv = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)

    num_m_blocks = tl.cdiv(S, BLOCK_M)
    for pm in range(num_m_blocks):
        offs_m = pm * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m[:, None] < S

        Q_tile = tl.load(q_ptr + q_base + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd,
                         mask=mask_m & mask_d, other=0.0).to(tl.float32)
        K_tile = tl.load(k_ptr + k_base + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd,
                         mask=mask_nd, other=0.0).to(tl.float32)
        dO_tile = tl.load(do_ptr + do_base + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod,
                          mask=mask_m & mask_d, other=0.0).to(tl.float32)
        O_tile = tl.load(o_ptr + o_base + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od,
                         mask=mask_m & mask_d, other=0.0).to(tl.float32)
        L_vals = tl.load(l_ptr + l_base + offs_m * stride_ls, mask=(offs_m < S), other=0.0)
        D_vals = tl.sum(dO_tile * O_tile, axis=1)

        scores = tl.dot(Q_tile, K_tile.T) * scale
        causal_mask = (offs_n[None, :] <= offs_m[:, None])
        attn_mask = causal_mask & mask_n.T & mask_m
        P = tl.where(attn_mask, tl.exp(scores - L_vals[:, None]), 0.0)

        acc_dv = tl.dot(P.T, dO_tile, acc=acc_dv)

    tl.store(dv_ptr + dv_base + offs_n[:, None] * stride_dvs + offs_d[None, :] * stride_dvd,
             acc_dv.to(tl.bfloat16), mask=mask_nd)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Causal MHA backward: compute dQ, dK, dV into preallocated outputs."""
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    scale = 1.0 / math.sqrt(D)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = 128

    num_m_blocks = triton.cdiv(S, BLOCK_M)
    num_n_blocks = triton.cdiv(S, BLOCK_N)

    grid_dq = (num_m_blocks, B, H)
    grid_dk = (num_n_blocks, B, H)
    grid_dv = (num_n_blocks, B, H)

    def strides4(t):
        return t.stride(0), t.stride(1), t.stride(2), t.stride(3)

    sq = strides4(Q)
    sk = strides4(K)
    sv = strides4(V)
    sdo = strides4(dO)
    so = strides4(O)
    sl = (L.stride(0), L.stride(1), L.stride(2))
    sdq = strides4(dQ)
    sdk = strides4(dK)
    sdv = strides4(dV)

    _dq_kernel[grid_dq](
        Q, K, V, dO, O, L, dQ,
        *sq, *sk, *sv, *sdo, *so, *sl, *sdq,
        S, D, scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=4, num_stages=3,
    )

    _dk_kernel[grid_dk](
        Q, K, V, dO, O, L, dK,
        *sq, *sk, *sv, *sdo, *so, *sl, *sdk,
        S, D, scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=4, num_stages=3,
    )

    _dv_kernel[grid_dv](
        Q, K, V, dO, O, L, dV,
        *sq, *sk, *sv, *sdo, *so, *sl, *sdv,
        S, D, scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=4, num_stages=3,
    )