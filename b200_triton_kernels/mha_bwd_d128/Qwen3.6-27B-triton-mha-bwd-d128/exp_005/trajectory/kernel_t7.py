import torch
import triton
import triton.language as tl


@triton.jit
def _mha_bwd_dq_kernel(
    Q, K, V, dO, O, L,
    dQ,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_dOb, stride_dOh, stride_dOs, stride_dOd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lb, stride_lh, stride_ls,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    B, H, S, d,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute dQ: each (program m, program bh) owns a unique output tile."""
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    pid_b = pid_bh // H
    pid_h = pid_bh % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    mask_m = offs_m < S
    mask_d = offs_d < d

    q_base = pid_b * stride_qb + pid_h * stride_qh
    k_base = pid_b * stride_kb + pid_h * stride_kh
    v_base = pid_b * stride_vb + pid_h * stride_vh
    do_base = pid_b * stride_dOb + pid_h * stride_dOh
    o_base = pid_b * stride_ob + pid_h * stride_oh
    l_base = pid_b * stride_lb + pid_h * stride_lh
    dq_base = pid_b * stride_dqb + pid_h * stride_dqh

    # Load Q tile once — reused across entire KV loop
    Q_tile = tl.load(
        Q + q_base + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd,
        mask=(mask_m[:, None] & mask_d[None, :]), other=0.0,
    )

    # Load dO and O for dP and D computation
    dO_tile = tl.load(
        dO + do_base + offs_m[:, None] * stride_dOs + offs_d[None, :] * stride_dOd,
        mask=(mask_m[:, None] & mask_d[None, :]), other=0.0,
    )

    O_tile = tl.load(
        O + o_base + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od,
        mask=(mask_m[:, None] & mask_d[None, :]), other=0.0,
    )

    # D[i] = sum_d(dO[i,d] * O[i,d])  shape [BLOCK_M, 1]
    D_tile = tl.sum(dO_tile * O_tile, axis=1)[:, None]

    L_tile = tl.load(
        L + l_base + offs_m * stride_ls,
        mask=mask_m, other=0.0,
    )[:, None]

    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    num_kv = tl.cdiv(S, BLOCK_N)
    for blk_n in range(num_kv):
        n_abs = blk_n * BLOCK_N + offs_n
        mask_n = n_abs < S

        K_tile = tl.load(
            K + k_base + n_abs[:, None] * stride_ks + offs_d[None, :] * stride_kd,
            mask=(mask_n[:, None] & mask_d[None, :]), other=0.0,
        )

        V_tile = tl.load(
            V + v_base + n_abs[:, None] * stride_vs + offs_d[None, :] * stride_vd,
            mask=(mask_n[:, None] & mask_d[None, :]), other=0.0,
        )

        # S[i,j] = Q[i,:] · K[j,:]^T · scale  →  [BLOCK_M, BLOCK_N]
        S_mat = tl.dot(Q_tile, K_tile.T) * scale
        P_mat = tl.exp(S_mat - L_tile)
        dP_mat = tl.dot(dO_tile, V_tile.T)
        dS_mat = P_mat * (dP_mat - D_tile) * scale

        # dQ += dS @ K
        acc = acc + tl.dot(dS_mat.to(tl.bfloat16), K_tile)

    # Store dQ
    tl.store(
        dQ + dq_base + offs_m[:, None] * stride_dqs + offs_d[None, :] * stride_dqd,
        acc.to(tl.bfloat16),
        mask=(mask_m[:, None] & mask_d[None, :]),
    )


@triton.jit
def _mha_bwd_dkv_kernel(
    Q, K, V, dO, O, L,
    dK, dV,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_dOb, stride_dOh, stride_dOs, stride_dOd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lb, stride_lh, stride_ls,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    B, H, S, d,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute dK and dV: each (program n, program bh) owns unique output tiles."""
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)

    pid_b = pid_bh // H
    pid_h = pid_bh % H

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    mask_n = offs_n < S
    mask_d = offs_d < d

    q_base = pid_b * stride_qb + pid_h * stride_qh
    k_base = pid_b * stride_kb + pid_h * stride_kh
    v_base = pid_b * stride_vb + pid_h * stride_vh
    do_base = pid_b * stride_dOb + pid_h * stride_dOh
    o_base = pid_b * stride_ob + pid_h * stride_oh
    l_base = pid_b * stride_lb + pid_h * stride_lh
    dk_base = pid_b * stride_dkb + pid_h * stride_dkh
    dv_base = pid_b * stride_dvb + pid_h * stride_dvh

    # Load K and V tiles once — reused across entire Q loop
    K_tile = tl.load(
        K + k_base + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd,
        mask=(mask_n[:, None] & mask_d[None, :]), other=0.0,
    )

    V_tile = tl.load(
        V + v_base + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd,
        mask=(mask_n[:, None] & mask_d[None, :]), other=0.0,
    )

    acc_dK = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    acc_dV = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)

    num_q = tl.cdiv(S, BLOCK_M)
    for blk_m in range(num_q):
        m_abs = blk_m * BLOCK_M + offs_m
        mask_m = m_abs < S

        Q_tile = tl.load(
            Q + q_base + m_abs[:, None] * stride_qs + offs_d[None, :] * stride_qd,
            mask=(mask_m[:, None] & mask_d[None, :]), other=0.0,
        )

        O_tile = tl.load(
            O + o_base + m_abs[:, None] * stride_os + offs_d[None, :] * stride_od,
            mask=(mask_m[:, None] & mask_d[None, :]), other=0.0,
        )

        dO_tile = tl.load(
            dO + do_base + m_abs[:, None] * stride_dOs + offs_d[None, :] * stride_dOd,
            mask=(mask_m[:, None] & mask_d[None, :]), other=0.0,
        )

        D_tile = tl.sum(dO_tile * O_tile, axis=1)[:, None]
        L_tile = tl.load(
            L + l_base + m_abs * stride_ls,
            mask=mask_m, other=0.0,
        )[:, None]

        S_mat = tl.dot(Q_tile, K_tile.T) * scale
        P_mat = tl.exp(S_mat - L_tile)
        dP_mat = tl.dot(dO_tile, V_tile.T)
        dS_mat = P_mat * (dP_mat - D_tile) * scale

        # dK += dS^T @ Q
        acc_dK = acc_dK + tl.dot(dS_mat.T.to(tl.bfloat16), Q_tile)
        # dV += P^T @ dO
        acc_dV = acc_dV + tl.dot(P_mat.T.to(tl.bfloat16), dO_tile)

    # Store dK
    tl.store(
        dK + dk_base + offs_n[:, None] * stride_dks + offs_d[None, :] * stride_dkd,
        acc_dK.to(tl.bfloat16),
        mask=(mask_n[:, None] & mask_d[None, :]),
    )

    # Store dV
    tl.store(
        dV + dv_base + offs_n[:, None] * stride_dvs + offs_d[None, :] * stride_dvd,
        acc_dV.to(tl.bfloat16),
        mask=(mask_n[:, None] & mask_d[None, :]),
    )


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Multi-head attention backward: compute dQ, dK, dV."""
    torch.cuda.set_device(Q.device)

    B, H, S, d = Q.shape
    scale = 1.0 / float(d ** 0.5)

    qs = tuple(int(s) for s in Q.stride())
    ks = tuple(int(s) for s in K.stride())
    vs = tuple(int(s) for s in V.stride())
    dos = tuple(int(s) for s in dO.stride())
    os_ = tuple(int(s) for s in O.stride())
    dqs = tuple(int(s) for s in dQ.stride())
    dks = tuple(int(s) for s in dK.stride())
    dvs = tuple(int(s) for s in dV.stride())
    ls = (int(L.stride(0)), int(L.stride(1)), int(L.stride(2)))

    bh_total = B * H
    BLOCK_D = d  # fixed head dim, passed as constexpr

    # Tile config: 64x64 seq, full d dimension
    BLOCK_M = 64
    BLOCK_N = 64

    num_bm = triton.cdiv(S, BLOCK_M)
    num_bn = triton.cdiv(S, BLOCK_N)

    # Launch dQ kernel
    _mha_bwd_dq_kernel[(num_bm, bh_total)](
        Q, K, V, dO, O, L, dQ,
        *qs, *ks, *vs, *dos, *os_,
        ls[0], ls[1], ls[2],
        *dqs,
        B, H, S, d, scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=4, num_stages=3,
    )

    # Launch dKV kernel
    _mha_bwd_dkv_kernel[(num_bn, bh_total)](
        Q, K, V, dO, O, L, dK, dV,
        *qs, *ks, *vs, *dos, *os_,
        ls[0], ls[1], ls[2],
        *dks, *dvs,
        B, H, S, d, scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=4, num_stages=3,
    )