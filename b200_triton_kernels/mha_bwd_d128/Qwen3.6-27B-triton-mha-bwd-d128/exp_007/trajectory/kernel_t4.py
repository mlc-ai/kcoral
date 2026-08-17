import torch
import triton
import triton.language as tl


@triton.jit
def _dQ_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, O_ptr, L_ptr, dQ_ptr,
    B, H, S,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lb, stride_lh, stride_ls,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    tau,
    TILE_M: tl.constexpr,
    TILE_K: tl.constexpr,
    HEAD_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)

    m_off = pid_m * TILE_M + tl.arange(0, TILE_M)
    m_mask = m_off < S
    d_off = tl.arange(0, HEAD_D)

    # Compute base offsets for this (b, h) pair
    q_base = pid_b * stride_qb + pid_h * stride_qh
    Q_tile = tl.load(
        Q_ptr + q_base + m_off[:, None] * stride_qs + d_off[None, :] * stride_qd,
        mask=m_mask[:, None], other=0.0,
    ).to(tl.bfloat16)

    do_base = pid_b * stride_dob + pid_h * stride_doh
    o_base = pid_b * stride_ob + pid_h * stride_oh
    dO_tile = tl.load(
        dO_ptr + do_base + m_off[:, None] * stride_dos + d_off[None, :] * stride_dod,
        mask=m_mask[:, None], other=0.0,
    ).to(tl.bfloat16)
    O_tile = tl.load(
        O_ptr + o_base + m_off[:, None] * stride_os + d_off[None, :] * stride_od,
        mask=m_mask[:, None], other=0.0,
    ).to(tl.bfloat16)

    # D_row[b] = sum_d(dO[b,d] * O[b,d]) accumulated in fp32
    D_row = tl.sum((dO_tile * O_tile).to(tl.float32), axis=1)

    l_base = pid_b * stride_lb + pid_h * stride_lh
    L_row = tl.load(L_ptr + l_base + m_off, mask=m_mask, other=0.0).to(tl.float32)

    dQ_acc = tl.zeros((TILE_M, HEAD_D), dtype=tl.float32)

    num_k_tiles = tl.cdiv(S, TILE_K)
    for k_idx in range(num_k_tiles):
        n_off = k_idx * TILE_K + tl.arange(0, TILE_K)
        n_mask = n_off < S

        k_base = pid_b * stride_kb + pid_h * stride_kh
        v_base = pid_b * stride_vb + pid_h * stride_vh
        K_tile = tl.load(
            K_ptr + k_base + n_off[:, None] * stride_ks + d_off[None, :] * stride_kd,
            mask=n_mask[:, None], other=0.0,
        ).to(tl.bfloat16)
        V_tile = tl.load(
            V_ptr + v_base + n_off[:, None] * stride_vs + d_off[None, :] * stride_vd,
            mask=n_mask[:, None], other=0.0,
        ).to(tl.bfloat16)

        # S = Q @ K^T * tau -> [TILE_M, TILE_K] fp32
        S_mat = tl.dot(Q_tile, K_tile.T).to(tl.float32) * tau
        # P = exp(S - L) -> fp32
        P_mat = tl.exp(S_mat - L_row[:, None])
        # dP = dO @ V^T -> fp32
        dP_mat = tl.dot(dO_tile, V_tile.T).to(tl.float32)
        # dS = P * (dP - D) -> fp32
        dS_raw = P_mat * (dP_mat - D_row[:, None])
        # dQ += (dS * tau) @ K -> accumulate in fp32
        dQ_acc += tl.dot((dS_raw * tau).to(tl.bfloat16), K_tile.to(tl.bfloat16))

    dq_out = dQ_acc.to(tl.bfloat16)
    dq_base = pid_b * stride_dqb + pid_h * stride_dqh
    tl.store(
        dQ_ptr + dq_base + m_off[:, None] * stride_dqs + d_off[None, :] * stride_dqd,
        dq_out,
        mask=m_mask[:, None],
    )


@triton.jit
def _dKV_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, O_ptr, L_ptr, dK_ptr, dV_ptr,
    B, H, S,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lb, stride_lh, stride_ls,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    tau,
    TILE_M: tl.constexpr,
    TILE_K: tl.constexpr,
    HEAD_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_n = tl.program_id(2)

    n_off = pid_n * TILE_K + tl.arange(0, TILE_K)
    n_mask = n_off < S
    d_off = tl.arange(0, HEAD_D)

    k_base = pid_b * stride_kb + pid_h * stride_kh
    v_base = pid_b * stride_vb + pid_h * stride_vh
    K_tile = tl.load(
        K_ptr + k_base + n_off[:, None] * stride_ks + d_off[None, :] * stride_kd,
        mask=n_mask[:, None], other=0.0,
    ).to(tl.bfloat16)
    V_tile = tl.load(
        V_ptr + v_base + n_off[:, None] * stride_vs + d_off[None, :] * stride_vd,
        mask=n_mask[:, None], other=0.0,
    ).to(tl.bfloat16)

    dK_acc = tl.zeros((TILE_K, HEAD_D), dtype=tl.float32)
    dV_acc = tl.zeros((TILE_K, HEAD_D), dtype=tl.float32)

    num_q_tiles = tl.cdiv(S, TILE_M)
    for q_idx in range(num_q_tiles):
        m_off = q_idx * TILE_M + tl.arange(0, TILE_M)
        m_mask = m_off < S

        q_base = pid_b * stride_qb + pid_h * stride_qh
        do_base = pid_b * stride_dob + pid_h * stride_doh
        o_base = pid_b * stride_ob + pid_h * stride_oh
        Q_tile = tl.load(
            Q_ptr + q_base + m_off[:, None] * stride_qs + d_off[None, :] * stride_qd,
            mask=m_mask[:, None], other=0.0,
        ).to(tl.bfloat16)
        dO_tile = tl.load(
            dO_ptr + do_base + m_off[:, None] * stride_dos + d_off[None, :] * stride_dod,
            mask=m_mask[:, None], other=0.0,
        ).to(tl.bfloat16)
        O_tile = tl.load(
            O_ptr + o_base + m_off[:, None] * stride_os + d_off[None, :] * stride_od,
            mask=m_mask[:, None], other=0.0,
        ).to(tl.bfloat16)

        D_row = tl.sum((dO_tile * O_tile).to(tl.float32), axis=1)

        l_base = pid_b * stride_lb + pid_h * stride_lh
        L_row = tl.load(L_ptr + l_base + m_off, mask=m_mask, other=0.0).to(tl.float32)

        S_mat = tl.dot(Q_tile, K_tile.T).to(tl.float32) * tau
        P_mat = tl.exp(S_mat - L_row[:, None])
        dP_mat = tl.dot(dO_tile, V_tile.T).to(tl.float32)
        dS_raw = P_mat * (dP_mat - D_row[:, None])

        # dK += dS^T @ Q * tau
        dK_acc += tl.dot((dS_raw.T * tau).to(tl.bfloat16), Q_tile.to(tl.bfloat16))
        # dV += P^T @ dO
        dV_acc += tl.dot(P_mat.T.to(tl.bfloat16), dO_tile.to(tl.bfloat16))

    dk_base = pid_b * stride_dkb + pid_h * stride_dkh
    dv_base = pid_b * stride_dvb + pid_h * stride_dvh
    tl.store(
        dK_ptr + dk_base + n_off[:, None] * stride_dks + d_off[None, :] * stride_dkd,
        dK_acc.to(tl.bfloat16),
        mask=n_mask[:, None],
    )
    tl.store(
        dV_ptr + dv_base + n_off[:, None] * stride_dvs + d_off[None, :] * stride_dvd,
        dV_acc.to(tl.bfloat16),
        mask=n_mask[:, None],
    )


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Multi-head attention backward: compute dQ, dK, dV into preallocated outputs."""
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    HEAD_D = d
    TILE_M = 64
    TILE_K = 64
    tau = 1.0 / (d ** 0.5)

    sq = Q.stride()
    sk = K.stride()
    sv = V.stride()
    sdo = dO.stride()
    so = O.stride()
    sl = L.stride()
    sdq = dQ.stride()
    sdk = dK.stride()
    sdv = dV.stride()

    if S == 0 or B == 0 or H == 0:
        return

    grid_dQ = (B, H, triton.cdiv(S, TILE_M))
    _dQ_kernel[grid_dQ](
        Q, K, V, dO, O, L, dQ,
        B, H, S,
        sq[0], sq[1], sq[2], sq[3],
        sk[0], sk[1], sk[2], sk[3],
        sv[0], sv[1], sv[2], sv[3],
        sdo[0], sdo[1], sdo[2], sdo[3],
        so[0], so[1], so[2], so[3],
        sl[0], sl[1], sl[2],
        sdq[0], sdq[1], sdq[2], sdq[3],
        tau,
        TILE_M=TILE_M,
        TILE_K=TILE_K,
        HEAD_D=HEAD_D,
        num_warps=4,
        num_stages=3,
    )

    grid_dKV = (B, H, triton.cdiv(S, TILE_K))
    _dKV_kernel[grid_dKV](
        Q, K, V, dO, O, L, dK, dV,
        B, H, S,
        sq[0], sq[1], sq[2], sq[3],
        sk[0], sk[1], sk[2], sk[3],
        sv[0], sv[1], sv[2], sv[3],
        sdo[0], sdo[1], sdo[2], sdo[3],
        so[0], so[1], so[2], so[3],
        sl[0], sl[1], sl[2],
        sdk[0], sdk[1], sdk[2], sdk[3],
        sdv[0], sdv[1], sdv[2], sdv[3],
        tau,
        TILE_M=TILE_M,
        TILE_K=TILE_K,
        HEAD_D=HEAD_D,
        num_warps=4,
        num_stages=3,
    )