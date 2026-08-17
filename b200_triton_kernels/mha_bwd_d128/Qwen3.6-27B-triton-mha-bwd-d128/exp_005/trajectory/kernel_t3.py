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
    q_ptrs = q_base + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    Q_tile = tl.load(q_ptrs, mask=(mask_m[:, None] & mask_d[None, :]), other=0.0)

    # Load dO and O for dP and D computation
    do_ptrs = do_base + offs_m[:, None] * stride_dOs + offs_d[None, :] * stride_dOd
    dO_tile = tl.load(do_ptrs, mask=(mask_m[:, None] & mask_d[None, :]), other=0.0)

    o_ptrs = o_base + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    O_tile = tl.load(o_ptrs, mask=(mask_m[:, None] & mask_d[None, :]), other=0.0)

    D_tile = tl.sum(dO_tile * O_tile, axis=1)[:, None]

    L_ptrs = l_base + offs_m * stride_ls
    L_tile = tl.load(L_ptrs, mask=mask_m, other=0.0)[:, None]

    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    num_kv = tl.cdiv(S, BLOCK_N)
    for blk_n in range(num_kv):
        n_abs = blk_n * BLOCK_N + offs_n
        mask_n = n_abs < S

        k_ptrs = k_base + n_abs[:, None] * stride_ks + offs_d[None, :] * stride_kd
        K_tile = tl.load(k_ptrs, mask=(mask_n[:, None] & mask_d[None, :]), other=0.0)

        v_ptrs = v_base + n_abs[:, None] * stride_vs + offs_d[None, :] * stride_vd
        V_tile = tl.load(v_ptrs, mask=(mask_n[:, None] & mask_d[None, :]), other=0.0)

        # S[i,j] = Q[i,:] · K[j,:]^T · scale  →  [BLOCK_M, BLOCK_N]
        S_mat = tl.dot(Q_tile, K_tile.T) * scale
        # Softmax probability
        P_mat = tl.exp(S_mat - L_tile)
        # dP[i,j] = dO[i,:] · V[j,:]^T
        dP_mat = tl.dot(dO_tile, V_tile.T)
        # Backprop through softmax, propagate scale: dS = P·(dP-D)·scale
        dS_mat = P_mat * (dP_mat - D_tile) * scale

        # dQ += dS @ K  →  [BLOCK_M, d]
        acc = acc + tl.dot(dS_mat.to(tl.bfloat16), K_tile)

    # Store dQ
    dq_ptrs = dq_base + offs_m[:, None] * stride_dqs + offs_d[None, :] * stride_dqd
    tl.store(dq_ptrs, acc.to(tl.bfloat16), mask=(mask_m[:, None] & mask_d[None, :]))


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
    k_ptrs = k_base + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    K_tile = tl.load(k_ptrs, mask=(mask_n[:, None] & mask_d[None, :]), other=0.0)

    v_ptrs = v_base + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
    V_tile = tl.load(v_ptrs, mask=(mask_n[:, None] & mask_d[None, :]), other=0.0)

    acc_dK = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    acc_dV = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)

    num_q = tl.cdiv(S, BLOCK_M)
    for blk_m in range(num_q):
        m_abs = blk_m * BLOCK_M + offs_m
        mask_m = m_abs < S

        q_ptrs = q_base + m_abs[:, None] * stride_qs + offs_d[None, :] * stride_qd
        Q_tile = tl.load(q_ptrs, mask=(mask_m[:, None] & mask_d[None, :]), other=0.0)

        o_ptrs = o_base + m_abs[:, None] * stride_os + offs_d[None, :] * stride_od
        O_tile = tl.load(o_ptrs, mask=(mask_m[:, None] & mask_d[None, :]), other=0.0)

        do_ptrs = do_base + m_abs[:, None] * stride_dOs + offs_d[None, :] * stride_dOd
        dO_tile = tl.load(do_ptrs, mask=(mask_m[:, None] & mask_d[None, :]), other=0.0)

        D_tile = tl.sum(dO_tile * O_tile, axis=1)[:, None]
        L_ptrs = l_base + m_abs * stride_ls
        L_tile = tl.load(L_ptrs, mask=mask_m, other=0.0)[:, None]

        S_mat = tl.dot(Q_tile, K_tile.T) * scale
        P_mat = tl.exp(S_mat - L_tile)
        dP_mat = tl.dot(dO_tile, V_tile.T)
        dS_mat = P_mat * (dP_mat - D_tile) * scale

        # dK += dS^T @ Q  →  [BLOCK_N, d]
        acc_dK = acc_dK + tl.dot(dS_mat.T.to(tl.bfloat16), Q_tile)
        # dV += P^T @ dO  →  [BLOCK_N, d]
        acc_dV = acc_dV + tl.dot(P_mat.T.to(tl.bfloat16), dO_tile)

    # Store dK
    dk_ptrs = dk_base + offs_n[:, None] * stride_dks + offs_d[None, :] * stride_dkd
    tl.store(dk_ptrs, acc_dK.to(tl.bfloat16), mask=(mask_n[:, None] & mask_d[None, :]))

    # Store dV
    dv_ptrs = dv_base + offs_n[:, None] * stride_dvs + offs_d[None, :] * stride_dvd
    tl.store(dv_ptrs, acc_dV.to(tl.bfloat16), mask=(mask_n[:, None] & mask_d[None, :]))


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
    BLOCK_D = d  # fixed head dimension as constexpr

    # Hardcoded tile sizes tuned for d=128, large S
    BLOCK_M = 64
    BLOCK_N = 64

    num_bm = triton.cdiv(S, BLOCK_M)
    num_bn = triton.cdiv(S, BLOCK_N)

    # Launch dQ kernel — each program owns one QUERY tile, no races
    _mha_bwd_dq_kernel[(num_bm, bh_total)](
        Q, K, V, dO, O, L, dQ,
        *qs, *ks, *vs, *dos, *os_,
        ls[0], ls[1], ls[2],
        *dqs,
        B, H, S, d, scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=4, num_stages=3,
    )

    # Launch dKV kernel — each program owns one KV tile, no races
    _mha_bwd_dkv_kernel[(num_bn, bh_total)](
        Q, K, V, dO, O, L, dK, dV,
        *qs, *ks, *vs, *dos, *os_,
        ls[0], ls[1], ls[2],
        *dks, *dvs,
        B, H, S, d, scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=4, num_stages=3,
    )