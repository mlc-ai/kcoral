import torch
import triton
import triton.language as tl
import math


@triton.jit
def _dQ_kernel(
    Q, K, V, O, dO, L, dQ,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    stride_lsb, stride_lsh, stride_lss,
    B, H, S,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    batch = pid_bh // H
    head = pid_bh % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    m_boundary = offs_m < S
    m_d_mask = m_boundary[:, None]

    q_off = batch * stride_qb + head * stride_qh
    k_off = batch * stride_kb + head * stride_kh
    v_off = batch * stride_vb + head * stride_vh
    o_off = batch * stride_ob + head * stride_oh
    do_off = batch * stride_dob + head * stride_doh
    l_off = batch * stride_lsb + head * stride_lsh
    dq_off = batch * stride_dqb + head * stride_dqh

    # Load Q tile [BLOCK_M, BLOCK_D]
    q_ptrs = Q + q_off + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    Q_tile = tl.load(q_ptrs, mask=m_d_mask, other=0.0)

    # Load dO tile [BLOCK_M, BLOCK_D]
    do_ptrs = dO + do_off + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod
    dO_tile = tl.load(do_ptrs, mask=m_d_mask, other=0.0)

    # Load O tile [BLOCK_M, BLOCK_D] (needed for D computation)
    o_ptrs = O + o_off + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    O_tile = tl.load(o_ptrs, mask=m_d_mask, other=0.0)

    # D[i] = sum_k(dO[i,k] * O[i,k]) -> [BLOCK_M]
    D = tl.sum(dO_tile * O_tile, axis=1)

    # Load L for this Q block -> [BLOCK_M]
    l_ptrs = L + l_off + offs_m * stride_lss
    L_tile = tl.load(l_ptrs, mask=m_boundary, other=float("inf"))

    # Accumulator for dQ [BLOCK_M, BLOCK_D]
    dQ_acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    # Iterate over all KV blocks
    num_n_blocks = tl.cdiv(S, BLOCK_N)
    for pid_n in range(num_n_blocks):
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_boundary = offs_n < S
        n_d_mask = n_boundary[:, None]

        # Load K tile [BLOCK_N, BLOCK_D]
        k_ptrs = K + k_off + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        K_tile = tl.load(k_ptrs, mask=n_d_mask, other=0.0)

        # Load V tile [BLOCK_N, BLOCK_D]
        v_ptrs = V + v_off + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
        V_tile = tl.load(v_ptrs, mask=n_d_mask, other=0.0)

        # S = Q @ K^T * scale -> [BLOCK_M, BLOCK_N]
        S_tile = tl.dot(Q_tile, K_tile.T) * scale

        # P = exp(S - L) -> [BLOCK_M, BLOCK_N]
        P = tl.exp(S_tile - L_tile[:, None])

        # dP = dO @ V^T -> [BLOCK_M, BLOCK_N]
        dP = tl.dot(dO_tile, V_tile.T)

        # dS = P * (dP - D) * scale -> [BLOCK_M, BLOCK_N]
        dS = P * (dP - D[:, None]) * scale

        # dQ += dS @ K -> [BLOCK_M, BLOCK_D]
        dQ_acc = tl.dot(dS.to(tl.bfloat16), K_tile, dQ_acc)

    # Store dQ result
    dq_ptrs = dQ + dq_off + offs_m[:, None] * stride_dqs + offs_d[None, :] * stride_dqd
    tl.store(dq_ptrs, dQ_acc.to(tl.bfloat16), mask=m_d_mask)


@triton.jit
def _dKV_kernel(
    Q, K, V, O, dO, L, dK, dV,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    stride_lsb, stride_lsh, stride_lss,
    B, H, S,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_n = tl.program_id(1)

    batch = pid_bh // H
    head = pid_bh % H

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    n_boundary = offs_n < S
    n_d_mask = n_boundary[:, None]

    k_off = batch * stride_kb + head * stride_kh
    v_off = batch * stride_vb + head * stride_vh
    dk_off = batch * stride_dkb + head * stride_dkh
    dv_off = batch * stride_dvb + head * stride_dvh

    # Load K and V tiles upfront
    k_ptrs = K + k_off + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    K_tile = tl.load(k_ptrs, mask=n_d_mask, other=0.0)

    v_ptrs = V + v_off + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
    V_tile = tl.load(v_ptrs, mask=n_d_mask, other=0.0)

    # Accumulators [BLOCK_N, BLOCK_D] in fp32
    dK_acc = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    dV_acc = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)

    num_m_blocks = tl.cdiv(S, BLOCK_M)

    for pid_m in range(num_m_blocks):
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        m_boundary = offs_m < S
        m_d_mask = m_boundary[:, None]

        # Load Q tile
        q_off_bh = batch * stride_qb + head * stride_qh
        q_ptrs = Q + q_off_bh + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
        Q_tile = tl.load(q_ptrs, mask=m_d_mask, other=0.0)

        # Load dO tile
        do_off_bh = batch * stride_dob + head * stride_doh
        do_ptrs = dO + do_off_bh + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod
        dO_tile = tl.load(do_ptrs, mask=m_d_mask, other=0.0)

        # Load O tile (for D computation)
        o_off_bh = batch * stride_ob + head * stride_oh
        o_ptrs = O + o_off_bh + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
        O_tile = tl.load(o_ptrs, mask=m_d_mask, other=0.0)

        # D[i] = sum(dO[i,k] * O[i,k])
        D = tl.sum(dO_tile * O_tile, axis=1)

        # Load L
        l_off_bh = batch * stride_lsb + head * stride_lsh
        l_ptrs = L + l_off_bh + offs_m * stride_lss
        L_tile = tl.load(l_ptrs, mask=m_boundary, other=float("inf"))

        # S = Q @ K^T * scale
        S_tile = tl.dot(Q_tile, K_tile.T) * scale

        # P = exp(S - L)
        P = tl.exp(S_tile - L_tile[:, None])

        # dP = dO @ V^T
        dP = tl.dot(dO_tile, V_tile.T)

        # dS = P * (dP - D) * scale
        dS = P * (dP - D[:, None]) * scale

        # dK += dS^T @ Q
        dK_acc = tl.dot(dS.T.to(tl.bfloat16), Q_tile, dK_acc)

        # dV += P^T @ dO
        dV_acc = tl.dot(P.T.to(tl.bfloat16), dO_tile, dV_acc)

    # Store dK
    dk_ptrs = dK + dk_off + offs_n[:, None] * stride_dks + offs_d[None, :] * stride_dkd
    tl.store(dk_ptrs, dK_acc.to(tl.bfloat16), mask=n_d_mask)

    # Store dV
    dv_ptrs = dV + dv_off + offs_n[:, None] * stride_dvs + offs_d[None, :] * stride_dvd
    tl.store(dv_ptrs, dV_acc.to(tl.bfloat16), mask=n_d_mask)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Multi-head attention backward pass."""
    torch.cuda.set_device(Q.device)

    B, H, S, D = Q.shape
    scale = 1.0 / math.sqrt(D)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = 128

    BH = B * H
    num_m_blocks = triton.cdiv(S, BLOCK_M)
    num_n_blocks = triton.cdiv(S, BLOCK_N)

    qs = Q.stride()
    ks = K.stride()
    vs = V.stride()
    os_s = O.stride()
    dos = dO.stride()
    dqs = dQ.stride()
    dks = dK.stride()
    dvs = dV.stride()
    ls = L.stride()

    # Kernel 1: compute dQ
    if num_m_blocks > 0:
        grid_dQ = (BH, num_m_blocks)
        _dQ_kernel[grid_dQ](
            Q, K, V, O, dO, L, dQ,
            qs[0], qs[1], qs[2], qs[3],
            ks[0], ks[1], ks[2], ks[3],
            vs[0], vs[1], vs[2], vs[3],
            os_s[0], os_s[1], os_s[2], os_s[3],
            dos[0], dos[1], dos[2], dos[3],
            dqs[0], dqs[1], dqs[2], dqs[3],
            ls[0], ls[1], ls[2],
            B, H, S,
            scale,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
            num_warps=4, num_stages=3,
        )

    # Kernel 2: compute dK and dV together
    if num_n_blocks > 0:
        grid_dKV = (BH, num_n_blocks)
        _dKV_kernel[grid_dKV](
            Q, K, V, O, dO, L, dK, dV,
            qs[0], qs[1], qs[2], qs[3],
            ks[0], ks[1], ks[2], ks[3],
            vs[0], vs[1], vs[2], vs[3],
            os_s[0], os_s[1], os_s[2], os_s[3],
            dos[0], dos[1], dos[2], dos[3],
            dks[0], dks[1], dks[2], dks[3],
            dvs[0], dvs[1], dvs[2], dvs[3],
            ls[0], ls[1], ls[2],
            B, H, S,
            scale,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
            num_warps=4, num_stages=3,
        )