import torch
import triton
import triton.language as tl


@triton.jit
def _dQ_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, O_ptr, L_ptr, dQ_ptr,
    BH, S, d,
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
):
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    m_off = pid_m * TILE_M + tl.arange(0, TILE_M)
    m_mask = m_off < S
    d_off = tl.arange(0, d)

    # Load Q once (TILE_M x d)
    Q_base = pid_bh * stride_qbh
    Q_tile = tl.load(
        Q_ptr + Q_base + m_off[:, None] * stride_qs + d_off[None, :] * stride_qd,
        mask=m_mask[:, None], other=0.0,
    )

    # Load dO and O to compute D row
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
    # D_row[b] = sum_d(dO[b,d] * O[b,d]) -> [TILE_M]
    D_row = tl.sum(dO_tile * O_tile, axis=1)

    # Load L row for softmax
    l_base = pid_bh * stride_lbh
    L_row = tl.load(L_ptr + l_base + m_off, mask=m_mask, other=0.0)

    # Initialize accumulator
    dQ_acc = tl.zeros((TILE_M, d), dtype=tl.float32)

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

        # S = Q @ K^T * tau  -> [TILE_M, TILE_K]
        S_mat = tl.dot(Q_tile, K_tile.T) * tau
        # P = exp(S - L)     -> [TILE_M, TILE_K]
        P_mat = tl.exp(S_mat - L_row[:, None])
        # dP = dO @ V^T      -> [TILE_M, TILE_K]
        dP_mat = tl.dot(dO_tile, V_tile.T)
        # dS = P * (dP - D)  -> [TILE_M, TILE_K]
        dS_raw = P_mat * (dP_mat - D_row[:, None])
        # dQ += (dS * tau) @ K  -> [TILE_M, d]
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
    BH, S, d,
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
):
    pid_bh = tl.program_id(0)
    pid_n = tl.program_id(1)

    n_off = pid_n * TILE_K + tl.arange(0, TILE_K)
    n_mask = n_off < S
    d_off = tl.arange(0, d)

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

    dK_acc = tl.zeros((TILE_K, d), dtype=tl.float32)
    dV_acc = tl.zeros((TILE_K, d), dtype=tl.float32)

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

        # D_row from this Q block
        D_row = tl.sum(dO_tile * O_tile, axis=1)

        # L row
        l_base = pid_bh * stride_lbh
        L_row = tl.load(L_ptr + l_base + m_off, mask=m_mask, other=0.0)

        # S = Q @ K^T * tau -> [TILE_M, TILE_K]
        S_mat = tl.dot(Q_tile, K_tile.T) * tau
        # P = exp(S - L) -> [TILE_M, TILE_K]
        P_mat = tl.exp(S_mat - L_row[:, None])
        # dP = dO @ V^T -> [TILE_M, TILE_K]
        dP_mat = tl.dot(dO_tile, V_tile.T)
        # dS = P * (dP - D) -> [TILE_M, TILE_K]
        dS_raw = P_mat * (dP_mat - D_row[:, None])

        # dK += dS^T @ Q * tau -> [TILE_K, d]
        dK_acc += tl.dot(dS_raw.T, Q_tile) * tau
        # dV += P^T @ dO -> [TILE_K, d]
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

    # Flatten (B, H) -> single dimension for uniform tiling
    Q_c = Q.reshape([BH, S, d]).contiguous()
    K_c = K.reshape([BH, S, d]).contiguous()
    V_c = V.reshape([BH, S, d]).contiguous()
    O_c = O.reshape([BH, S, d]).contiguous()
    dO_c = dO.reshape([BH, S, d]).contiguous()
    L_c = L.reshape([BH, S]).contiguous()

    dQ_c = dQ.reshape([BH, S, d]).contiguous()
    dK_c = dK.reshape([BH, S, d]).contiguous()
    dV_c = dV.reshape([BH, S, d]).contiguous()

    if S == 0 or BH == 0:
        return

    TILE_M = 64
    TILE_K = 64
    tau = 1.0 / (d ** 0.5)

    # Strides on contiguous [BH, S, d] tensors
    sq_bh, sq_s, sq_d = Q_c.stride()
    sk_bh, sk_s, sk_d = K_c.stride()
    sv_bh, sv_s, sv_d = V_c.stride()
    sdO_bh, sdO_s, sdO_d = dO_c.stride()
    sO_bh, sO_s, sO_d = O_c.stride()
    sL_bh, sL_s = L_c.stride()
    sdQ_bh, sdQ_s, sdQ_d = dQ_c.stride()
    sdK_bh, sdK_s, sdK_d = dK_c.stride()
    sdV_bh, sdV_s, sdV_d = dV_c.stride()

    # Launch dQ kernel: grid = (BH, cdiv(S, TILE_M))
    grid_dQ = (BH, triton.cdiv(S, TILE_M))
    _dQ_kernel[grid_dQ](
        Q_c, K_c, V_c, dO_c, O_c, L_c, dQ_c,
        BH, S, d,
        sq_bh, sq_s, sq_d,
        sk_bh, sk_s, sk_d,
        sv_bh, sv_s, sv_d,
        sdO_bh, sdO_s, sdO_d,
        sO_bh, sO_s, sO_d,
        sL_bh, sL_s,
        sdQ_bh, sdQ_s, sdQ_d,
        tau,
        TILE_M=TILE_M,
        TILE_K=TILE_K,
        num_warps=4,
        num_stages=3,
    )

    # Launch dKV kernel: grid = (BH, cdiv(S, TILE_K))
    grid_dKV = (BH, triton.cdiv(S, TILE_K))
    _dKV_kernel[grid_dKV](
        Q_c, K_c, V_c, dO_c, O_c, L_c, dK_c, dV_c,
        BH, S, d,
        sq_bh, sq_s, sq_d,
        sk_bh, sk_s, sk_d,
        sv_bh, sv_s, sv_d,
        sdO_bh, sdO_s, sdO_d,
        sO_bh, sO_s, sO_d,
        sL_bh, sL_s,
        sdK_bh, sdK_s, sdK_d,
        sdV_bh, sdV_s, sdV_d,
        tau,
        TILE_M=TILE_M,
        TILE_K=TILE_K,
        num_warps=4,
        num_stages=3,
    )