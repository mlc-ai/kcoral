import torch
import triton
import triton.language as tl


@triton.jit
def _dQ_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, O_ptr, L_ptr, dQ_ptr,
    BH, S,
    stride_qbh, stride_qs, stride_qd,
    stride_kbh, stride_ks, stride_kd,
    stride_vbh, stride_vs, stride_vd,
    stride_dobh, stride_dos, stride_dod,
    stride_obh, stride_os, stride_od,
    stride_lbh, stride_ls,
    stride_dqbh, stride_dqs, stride_dqd,
    tau,
    TILE_M: tl.constexpr,
    TILE_K: tl.constexpr,
    HEAD_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    m_off = pid_m * TILE_M + tl.arange(0, TILE_M)
    m_mask = m_off < S
    d_off = tl.arange(0, HEAD_D)

    # Load Q once (TILE_M x HEAD_D)
    Q_base = pid_bh * stride_qbh
    Q_tile = tl.load(
        Q_ptr + Q_base + m_off[:, None] * stride_qs + d_off[None, :] * stride_qd,
        mask=m_mask[:, None], other=0.0,
    )

    # Load dO and O to compute D row locally
    do_base = pid_bh * stride_dobh
    o_base = pid_bh * stride_obh
    dO_tile = tl.load(
        dO_ptr + do_base + m_off[:, None] * stride_dos + d_off[None, :] * stride_dod,
        mask=m_mask[:, None], other=0.0,
    )
    O_tile = tl.load(
        O_ptr + o_base + m_off[:, None] * stride_os + d_off[None, :] * stride_od,
        mask=m_mask[:, None], other=0.0,
    )
    D_row = tl.sum(dO_tile * O_tile, axis=1)

    # Load L row for softmax correction
    l_base = pid_bh * stride_lbh
    L_row = tl.load(L_ptr + l_base + m_off, mask=m_mask, other=0.0)

    # Initialize accumulator
    dQ_acc = tl.zeros((TILE_M, HEAD_D), dtype=tl.float32)

    # Loop over K/V blocks
    num_k_tiles = tl.cdiv(S, TILE_K)
    for k_idx in range(num_k_tiles):
        n_off = k_idx * TILE_K + tl.arange(0, TILE_K)
        n_mask = n_off < S

        k_base = pid_bh * stride_kbh
        v_base = pid_bh * stride_vbh
        K_tile = tl.load(
            K_ptr + k_base + n_off[:, None] * stride_ks + d_off[None, :] * stride_kd,
            mask=n_mask[:, None], other=0.0,
        )
        V_tile = tl.load(
            V_ptr + v_base + n_off[:, None] * stride_vs + d_off[None, :] * stride_vd,
            mask=n_mask[:, None], other=0.0,
        )

        S_mat = tl.dot(Q_tile, K_tile.T) * tau
        P_mat = tl.exp(S_mat - L_row[:, None])
        dP_mat = tl.dot(dO_tile, V_tile.T)
        dS_raw = P_mat * (dP_mat - D_row[:, None])
        dQ_acc += tl.dot(dS_raw, K_tile) * tau

    dq_out = dQ_acc.to(tl.bfloat16)
    tl.store(
        dQ_ptr + pid_bh * stride_dqbh + m_off[:, None] * stride_dqs + d_off[None, :] * stride_dqd,
        dq_out,
        mask=m_mask[:, None],
    )


@triton.jit
def _dKV_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, O_ptr, L_ptr, dK_ptr, dV_ptr,
    BH, S,
    stride_qbh, stride_qs, stride_qd,
    stride_kbh, stride_ks, stride_kd,
    stride_vbh, stride_vs, stride_vd,
    stride_dobh, stride_dos, stride_dod,
    stride_obh, stride_os, stride_od,
    stride_lbh, stride_ls,
    stride_dkbh, stride_dks, stride_dkd,
    stride_dvbh, stride_dvs, stride_dvd,
    tau,
    TILE_M: tl.constexpr,
    TILE_K: tl.constexpr,
    HEAD_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_n = tl.program_id(1)

    n_off = pid_n * TILE_K + tl.arange(0, TILE_K)
    n_mask = n_off < S
    d_off = tl.arange(0, HEAD_D)

    # Load K and V once for this KV block
    k_base = pid_bh * stride_kbh
    v_base = pid_bh * stride_vbh
    K_tile = tl.load(
        K_ptr + k_base + n_off[:, None] * stride_ks + d_off[None, :] * stride_kd,
        mask=n_mask[:, None], other=0.0,
    )
    V_tile = tl.load(
        V_ptr + v_base + n_off[:, None] * stride_vs + d_off[None, :] * stride_vd,
        mask=n_mask[:, None], other=0.0,
    )

    dK_acc = tl.zeros((TILE_K, HEAD_D), dtype=tl.float32)
    dV_acc = tl.zeros((TILE_K, HEAD_D), dtype=tl.float32)

    # Loop over Q blocks
    num_q_tiles = tl.cdiv(S, TILE_M)
    for q_idx in range(num_q_tiles):
        m_off = q_idx * TILE_M + tl.arange(0, TILE_M)
        m_mask = m_off < S

        q_base = pid_bh * stride_qbh
        do_base = pid_bh * stride_dobh
        o_base = pid_bh * stride_obh
        Q_tile = tl.load(
            Q_ptr + q_base + m_off[:, None] * stride_qs + d_off[None, :] * stride_qd,
            mask=m_mask[:, None], other=0.0,
        )
        dO_tile = tl.load(
            dO_ptr + do_base + m_off[:, None] * stride_dos + d_off[None, :] * stride_dod,
            mask=m_mask[:, None], other=0.0,
        )
        O_tile = tl.load(
            O_ptr + o_base + m_off[:, None] * stride_os + d_off[None, :] * stride_od,
            mask=m_mask[:, None], other=0.0,
        )

        D_row = tl.sum(dO_tile * O_tile, axis=1)

        l_base = pid_bh * stride_lbh
        L_row = tl.load(L_ptr + l_base + m_off, mask=m_mask, other=0.0)

        S_mat = tl.dot(Q_tile, K_tile.T) * tau
        P_mat = tl.exp(S_mat - L_row[:, None])
        dP_mat = tl.dot(dO_tile, V_tile.T)
        dS_raw = P_mat * (dP_mat - D_row[:, None])

        dK_acc += tl.dot(dS_raw.T, Q_tile) * tau
        dV_acc += tl.dot(P_mat.T, dO_tile)

    tl.store(
        dK_ptr + pid_bh * stride_dkbh + n_off[:, None] * stride_dks + d_off[None, :] * stride_dkd,
        dK_acc.to(tl.bfloat16),
        mask=n_mask[:, None],
    )
    tl.store(
        dV_ptr + pid_bh * stride_dvbh + n_off[:, None] * stride_dvs + d_off[None, :] * stride_dvd,
        dV_acc.to(tl.bfloat16),
        mask=n_mask[:, None],
    )


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Multi-head attention backward: compute dQ, dK, dV into preallocated outputs."""
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    BH = B * H
    TARGET_SHAPE = [BH, S, d]
    L_TARGET_SHAPE = [BH, S]

    # Make contiguous copies only if needed (avoid unconditional allocation)
    if Q.stride() != (S * d, d, 1):
        Q_c = torch.empty(TARGET_SHAPE, dtype=Q.dtype, device=Q.device)
        Q_c.copy_(Q.reshape(TARGET_SHAPE))
    else:
        Q_c = Q.reshape(TARGET_SHAPE)

    if K.stride() != (S * d, d, 1):
        K_c = torch.empty(TARGET_SHAPE, dtype=K.dtype, device=K.device)
        K_c.copy_(K.reshape(TARGET_SHAPE))
    else:
        K_c = K.reshape(TARGET_SHAPE)

    if V.stride() != (S * d, d, 1):
        V_c = torch.empty(TARGET_SHAPE, dtype=V.dtype, device=V.device)
        V_c.copy_(V.reshape(TARGET_SHAPE))
    else:
        V_c = V.reshape(TARGET_SHAPE)

    if O.stride() != (S * d, d, 1):
        O_c = torch.empty(TARGET_SHAPE, dtype=O.dtype, device=O.device)
        O_c.copy_(O.reshape(TARGET_SHAPE))
    else:
        O_c = O.reshape(TARGET_SHAPE)

    if dO.stride() != (S * d, d, 1):
        dO_c = torch.empty(TARGET_SHAPE, dtype=dO.dtype, device=dO.device)
        dO_c.copy_(dO.reshape(TARGET_SHAPE))
    else:
        dO_c = dO.reshape(TARGET_SHAPE)

    if L.stride() != (S, 1):
        L_c = torch.empty(L_TARGET_SHAPE, dtype=L.dtype, device=L.device)
        L_c.copy_(L.reshape(L_TARGET_SHAPE))
    else:
        L_c = L.reshape(L_TARGET_SHAPE)

    if dQ.stride() != (S * d, d, 1):
        dQ_c = torch.empty(TARGET_SHAPE, dtype=dQ.dtype, device=dQ.device)
    else:
        dQ_c = dQ.reshape(TARGET_SHAPE)

    if dK.stride() != (S * d, d, 1):
        dK_c = torch.empty(TARGET_SHAPE, dtype=dK.dtype, device=dK.device)
    else:
        dK_c = dK.reshape(TARGET_SHAPE)

    if dV.stride() != (S * d, d, 1):
        dV_c = torch.empty(TARGET_SHAPE, dtype=dV.dtype, device=dV.device)
    else:
        dV_c = dV.reshape(TARGET_SHAPE)

    if S == 0 or BH == 0:
        return

    TILE_M = 64
    TILE_K = 64
    HEAD_D = d
    tau = 1.0 / (d ** 0.5)

    sq = Q_c.stride()
    sk = K_c.stride()
    sv = V_c.stride()
    sdo = dO_c.stride()
    so = O_c.stride()
    sl = L_c.stride()
    sdq = dQ_c.stride()
    sdk = dK_c.stride()
    sdv = dV_c.stride()

    grid_dQ = (BH, triton.cdiv(S, TILE_M))
    _dQ_kernel[grid_dQ](
        Q_c, K_c, V_c, dO_c, O_c, L_c, dQ_c,
        BH, S,
        sq[0], sq[1], sq[2],
        sk[0], sk[1], sk[2],
        sv[0], sv[1], sv[2],
        sdo[0], sdo[1], sdo[2],
        so[0], so[1], so[2],
        sl[0], sl[1],
        sdq[0], sdq[1], sdq[2],
        tau,
        TILE_M=TILE_M,
        TILE_K=TILE_K,
        HEAD_D=HEAD_D,
        num_warps=4,
        num_stages=3,
    )

    grid_dKV = (BH, triton.cdiv(S, TILE_K))
    _dKV_kernel[grid_dKV](
        Q_c, K_c, V_c, dO_c, O_c, L_c, dK_c, dV_c,
        BH, S,
        sq[0], sq[1], sq[2],
        sk[0], sk[1], sk[2],
        sv[0], sv[1], sv[2],
        sdo[0], sdo[1], sdo[2],
        so[0], so[1], so[2],
        sl[0], sl[1],
        sdk[0], sdk[1], sdk[2],
        sdv[0], sdv[1], sdv[2],
        tau,
        TILE_M=TILE_M,
        TILE_K=TILE_K,
        HEAD_D=HEAD_D,
        num_warps=4,
        num_stages=3,
    )

    # Copy back from temporary buffers if we allocated them
    if dQ.stride() != (S * d, d, 1):
        dQ.reshape(TARGET_SHAPE).copy_(dQ_c)
    if dK.stride() != (S * d, d, 1):
        dK.reshape(TARGET_SHAPE).copy_(dK_c)
    if dV.stride() != (S * d, d, 1):
        dV.reshape(TARGET_SHAPE).copy_(dV_c)