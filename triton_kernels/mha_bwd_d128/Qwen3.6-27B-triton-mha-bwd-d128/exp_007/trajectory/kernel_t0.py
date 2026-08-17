import torch
import triton
import triton.language as tl

TILE_SIZE = 64
NUM_WARPS = 4
NUM_STAGES = 3


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Multi-head attention backward pass."""
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    BH = B * H

    # Reshape to [BH, S, d] for uniform handling across all heads
    Q_c = Q.reshape([BH, S, d]).contiguous()
    K_c = K.reshape([BH, S, d]).contiguous()
    V_c = V.reshape([BH, S, d]).contiguous()
    O_c = O.reshape([BH, S, d]).contiguous()
    dO_c = dO.reshape([BH, S, d]).contiguous()

    # L has shape [BH, S]
    L_c = L.reshape([BH, S]).contiguous()

    # Output views for kernel writing
    dQ_c = dQ.reshape([BH, S, d]).contiguous()
    dK_c = dK.reshape([BH, S, d]).contiguous()
    dV_c = dV.reshape([BH, S, d]).contiguous()

    if S == 0:
        return

    # Precompute D = rowsum(dO * O) with Triton kernel
    D = torch.empty([BH, S], device=dO.device, dtype=torch.float32)
    grid_D = (BH, triton.cdiv(S, TILE_SIZE), 1)
    _preprocess_D_kernel[grid_D](
        dO_c, O_c, D,
        BH, S, d,
        dO_c.stride(0), dO_c.stride(1), dO_c.stride(2),
        O_c.stride(0), O_c.stride(1), O_c.stride(2),
        D.stride(0), D.stride(1),
        TILE_SIZE=TILE_SIZE,
        num_warps=NUM_WARPS,
    )

    tau = 1.0 / (d ** 0.5)

    # Get strides
    sq = Q_c.stride()
    sk = K_c.stride()
    sv = V_c.stride()
    sdO = dO_c.stride()
    sD = D.stride()
    sL = L_c.stride()
    sdQ = dQ_c.stride()
    sdK = dK_c.stride()
    sdV = dV_c.stride()

    # Launch dQ kernel
    grid_dQ = (BH, triton.cdiv(S, TILE_SIZE))
    _dQ_kernel[grid_dQ](
        Q_c, K_c, V_c, dO_c, D, L_c, dQ_c,
        BH, S, d,
        sq[0], sq[1], sq[2],
        sk[0], sk[1], sk[2],
        sv[0], sv[1], sv[2],
        sdO[0], sdO[1], sdO[2],
        sD[0], sD[1],
        sL[0], sL[1],
        sdQ[0], sdQ[1], sdQ[2],
        tau,
        TILE_SIZE=TILE_SIZE,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )

    # Launch dK/dV kernel
    grid_dKV = (BH, triton.cdiv(S, TILE_SIZE))
    _dKV_kernel[grid_dKV](
        Q_c, K_c, V_c, dO_c, D, L_c, dK_c, dV_c,
        BH, S, d,
        sq[0], sq[1], sq[2],
        sk[0], sk[1], sk[2],
        sv[0], sv[1], sv[2],
        sdO[0], sdO[1], sdO[2],
        sD[0], sD[1],
        sL[0], sL[1],
        sdK[0], sdK[1], sdK[2],
        sdV[0], sdV[1], sdV[2],
        tau,
        TILE_SIZE=TILE_SIZE,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )


@triton.jit
def _preprocess_D_kernel(
    dO_ptr, O_ptr, D_ptr,
    BH, S, d,
    stride_dobh, stride_dos, stride_dod,
    stride_obh, stride_os, stride_od,
    stride_dbh, stride_ds,
    TILE_S: tl.constexpr,
):
    """Compute D[bh, s] = sum_d(dO[bh, s, d] * O[bh, s, d])."""
    pid_bh = tl.program_id(0)
    pid_s = tl.program_id(1)

    s_off = pid_s * TILE_S + tl.arange(0, TILE_S)
    d_off = tl.arange(0, d)
    s_mask = s_off < S

    bh_off = pid_bh * stride_dobh
    do_tile = tl.load(
        dO_ptr + bh_off + s_off[:, None] * stride_dos + d_off[None, :] * stride_dod,
        mask=s_mask[:, None], other=0.0,
    )
    o_tile = tl.load(
        O_ptr + pid_bh * stride_obh + s_off[:, None] * stride_os + d_off[None, :] * stride_od,
        mask=s_mask[:, None], other=0.0,
    )

    prod = do_tile * o_tile
    D_val = tl.sum(prod, axis=1)

    tl.store(D_ptr + pid_bh * stride_dbh + s_off, D_val, mask=s_mask)


@triton.jit
def _dQ_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, D_ptr, L_ptr, dQ_ptr,
    BH, S, d,
    stride_qbh, stride_qs, stride_qd,
    stride_kbh, stride_ks, stride_kd,
    stride_vbh, stride_vs, stride_vd,
    stride_dobh, stride_dos, stride_dod,
    stride_dbh, stride_ds,
    stride_lbh, stride_ls,
    stride_dqbh, stride_dqs, stride_dqd,
    tau,
    TILE_SIZE: tl.constexpr,
):
    """Compute dQ for a single (batch_head, Q_tile)."""
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    m_off = pid_m * TILE_SIZE + tl.arange(0, TILE_SIZE)
    m_mask = m_off < S
    d_off = tl.arange(0, d)

    dQ_acc = tl.zeros((TILE_SIZE, d), dtype=tl.float32)

    q_base = pid_bh * stride_qbh
    k_base = pid_bh * stride_kbh
    v_base = pid_bh * stride_vbh
    do_base = pid_bh * stride_dobh
    l_base = pid_bh * stride_lbh
    D_base = pid_bh * stride_dbh

    # Load Q once (outside loop)
    Q_tile = tl.load(
        Q_ptr + q_base + m_off[:, None] * stride_qs + d_off[None, :] * stride_qd,
        mask=m_mask[:, None], other=0.0,
    )

    # Load dO once (outside loop)
    dO_tile = tl.load(
        dO_ptr + do_base + m_off[:, None] * stride_dos + d_off[None, :] * stride_dod,
        mask=m_mask[:, None], other=0.0,
    )

    # Load L and D once (per Q tile)
    L_row = tl.load(L_ptr + l_base + m_off, mask=m_mask, other=0.0)
    D_row = tl.load(D_ptr + D_base + m_off, mask=m_mask, other=0.0)

    num_k_tiles = tl.cdiv(S, TILE_SIZE)
    for k_idx in range(num_k_tiles):
        n_off = k_idx * TILE_SIZE + tl.arange(0, TILE_SIZE)
        n_mask = n_off < S

        K_tile = tl.load(
            K_ptr + k_base + n_off[:, None] * stride_ks + d_off[None, :] * stride_kd,
            mask=n_mask[:, None], other=0.0,
        )
        V_tile = tl.load(
            V_ptr + v_base + n_off[:, None] * stride_vs + d_off[None, :] * stride_vd,
            mask=n_mask[:, None], other=0.0,
        )

        # S = Q @ K^T * tau
        S_mat = tl.dot(Q_tile, K_tile.T) * tau
        # P = exp(S - L)
        P_mat = tl.exp(S_mat - L_row[:, None])
        # dP = dO @ V^T
        dP_mat = tl.dot(dO_tile, V_tile.T)
        # dS = P * (dP - D) * tau is folded into dQ accumulation
        # dS_unscaled = P * (dP - D)
        dS_raw = P_mat * (dP_mat - D_row[:, None])
        # dQ += dS @ K = (dS_raw * tau) @ K
        dQ_acc += tl.dot(dS_raw, K_tile) * tau

    # Store dQ
    tl.store(
        dQ_ptr + pid_bh * stride_dqbh + m_off[:, None] * stride_dqs + d_off[None, :] * stride_dqd,
        dQ_acc.to(tl.bfloat16),
        mask=m_mask[:, None],
    )


@triton.jit
def _dKV_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, D_ptr, L_ptr, dK_ptr, dV_ptr,
    BH, S, d,
    stride_qbh, stride_qs, stride_qd,
    stride_kbh, stride_ks, stride_kd,
    stride_vbh, stride_vs, stride_vd,
    stride_dobh, stride_dos, stride_dod,
    stride_dbh, stride_ds,
    stride_lbh, stride_ls,
    stride_dkbh, stride_dks, stride_dkd,
    stride_dvbh, stride_dvs, stride_dvd,
    tau,
    TILE_SIZE: tl.constexpr,
):
    """Compute dK and dV for a single (batch_head, KV_tile)."""
    pid_bh = tl.program_id(0)
    pid_n = tl.program_id(1)

    n_off = pid_n * TILE_SIZE + tl.arange(0, TILE_SIZE)
    n_mask = n_off < S
    d_off = tl.arange(0, d)

    dK_acc = tl.zeros((TILE_SIZE, d), dtype=tl.float32)
    dV_acc = tl.zeros((TILE_SIZE, d), dtype=tl.float32)

    k_base = pid_bh * stride_kbh
    v_base = pid_bh * stride_vbh

    # Load K and V once (outside loop)
    K_tile = tl.load(
        K_ptr + k_base + n_off[:, None] * stride_ks + d_off[None, :] * stride_kd,
        mask=n_mask[:, None], other=0.0,
    )
    V_tile = tl.load(
        V_ptr + v_base + n_off[:, None] * stride_vs + d_off[None, :] * stride_vd,
        mask=n_mask[:, None], other=0.0,
    )

    num_q_tiles = tl.cdiv(S, TILE_SIZE)
    for q_idx in range(num_q_tiles):
        m_off = q_idx * TILE_SIZE + tl.arange(0, TILE_SIZE)
        m_mask = m_off < S

        Q_tile = tl.load(
            Q_ptr + pid_bh * stride_qbh + m_off[:, None] * stride_qs + d_off[None, :] * stride_qd,
            mask=m_mask[:, None], other=0.0,
        )
        dO_tile = tl.load(
            dO_ptr + pid_bh * stride_dobh + m_off[:, None] * stride_dos + d_off[None, :] * stride_dod,
            mask=m_mask[:, None], other=0.0,
        )
        L_col = tl.load(D_ptr + pid_bh * stride_dbh + m_off, mask=m_mask, other=0.0)
        D_col = tl.load(L_ptr + pid_bh * stride_lbh + m_off, mask=m_mask, other=0.0)

        # Re-load L and D with correct names
        L_col = tl.load(L_ptr + pid_bh * stride_lbh + m_off, mask=m_mask, other=0.0)
        D_col = tl.load(D_ptr + pid_bh * stride_dbh + m_off, mask=m_mask, other=0.0)

        # S = Q @ K^T * tau
        S_mat = tl.dot(Q_tile, K_tile.T) * tau
        # P = exp(S - L)
        P_mat = tl.exp(S_mat - L_col[:, None])
        # dP = dO @ V^T
        dP_mat = tl.dot(dO_tile, V_tile.T)
        # dS_raw = P * (dP - D)
        dS_raw = P_mat * (dP_mat - D_col[:, None])

        # dK += dS^T @ Q (with tau factor: dS_unscaled * tau)
        dK_acc += tl.dot(dS_raw.T, Q_tile) * tau
        # dV += P^T @ dO
        dV_acc += tl.dot(P_mat.T, dO_tile)

    # Store dK
    tl.store(
        dK_ptr + pid_bh * stride_dkbh + n_off[:, None] * stride_dks + d_off[None, :] * stride_dkd,
        dK_acc.to(tl.bfloat16),
        mask=n_mask[:, None],
    )
    # Store dV
    tl.store(
        dV_ptr + pid_bh * stride_dvbh + n_off[:, None] * stride_dvs + d_off[None, :] * stride_dvd,
        dV_acc.to(tl.bfloat16),
        mask=n_mask[:, None],
    )