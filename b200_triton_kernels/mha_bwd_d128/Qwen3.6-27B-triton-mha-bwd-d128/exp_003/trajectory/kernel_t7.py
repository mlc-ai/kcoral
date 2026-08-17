import torch
import triton
import triton.language as tl
import math


@triton.jit
def _dQ_full(
    Q, K, V, dO, L, dQ, dK, dV,
    stride_QB, stride_QH, stride_QS, stride_QD,
    stride_KB, stride_KH, stride_KS, stride_KD,
    stride_VB, stride_VH, stride_VS, stride_VD,
    stride_OB, stride_OH, stride_Os, stride_OD,
    stride_LB, stride_LH, stride_LS,
    stride_dQB, stride_dQH, stride_dQS, stride_dQD,
    stride_dKB, stride_dKH, stride_dKS, stride_dKD,
    stride_dVB, stride_dVH, stride_dVS, stride_dVD,
    B, H, S, D, inv_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """Compute dQ via decomposition. Stash C[b,h,q] into dV[b,h,:,0]."""
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    n_sm = tl.cdiv(S, BLOCK_M)
    total_bh = B * H

    if pid_m >= n_sm or pid_bh >= total_bh:
        return

    b = pid_bh // H
    h = pid_bh % H

    base_q = Q + b * stride_QB + h * stride_QH
    base_k = K + b * stride_KB + h * stride_KH
    base_v = V + b * stride_VB + h * stride_VH
    base_do = dO + b * stride_OB + h * stride_OH
    base_l = L + b * stride_LB + h * stride_LH
    base_dq = dQ + b * stride_dQB + h * stride_dQH
    base_dv = dV + b * stride_dVB + h * stride_dVH

    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_d = tl.arange(0, BLOCK_D)
    m_mask = off_m < S
    d_mask = off_d < D
    md_mask = m_mask[:, None] & d_mask[None, :]

    Q_tile = tl.load(base_q + off_m[:, None] * stride_QS + off_d[None, :] * stride_QD,
                     mask=md_mask, other=0.0)
    dO_tile = tl.load(base_do + off_m[:, None] * stride_Os + off_d[None, :] * stride_OD,
                      mask=md_mask, other=0.0)
    L_vals = tl.load(base_l + off_m * stride_LS, mask=m_mask, other=0.0)

    acc_A = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    acc_E = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    acc_C = tl.zeros((BLOCK_M,), dtype=tl.float32)

    n_sn = tl.cdiv(S, BLOCK_N)
    for sn in range(n_sn):
        off_n = sn * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = off_n < S
        nd_mask = n_mask[:, None] & d_mask[None, :]

        K_tile = tl.load(base_k + off_n[:, None] * stride_KS + off_d[None, :] * stride_KD,
                         mask=nd_mask, other=0.0)
        V_tile = tl.load(base_v + off_n[:, None] * stride_VS + off_d[None, :] * stride_VD,
                         mask=nd_mask, other=0.0)

        S_mat = tl.dot(Q_tile, K_tile.T) * inv_scale
        P_mat = tl.exp(S_mat - L_vals[:, None])

        dOV = tl.dot(dO_tile.to(tl.float32), V_tile.to(tl.float32).T)
        weight = P_mat * dOV

        acc_A = tl.dot(weight, K_tile.to(tl.float32), acc_A)
        acc_E = tl.dot(P_mat, K_tile.to(tl.float32), acc_E)
        acc_C += tl.sum(weight, axis=1)

    dQ_val = (acc_A - acc_C[:, None] * acc_E) * inv_scale
    tl.store(base_dq + off_m[:, None] * stride_dQS + off_d[None, :] * stride_dQD,
             dQ_val.to(tl.bfloat16), mask=md_mask)

    tl.store(base_dv + off_m * stride_dVS, acc_C.to(tl.bfloat16), mask=m_mask)


@triton.jit
def _dK_full(
    Q, K, V, dO, L, dQ, dK, dV,
    stride_QB, stride_QH, stride_QS, stride_QD,
    stride_KB, stride_KH, stride_KS, stride_KD,
    stride_VB, stride_VH, stride_VS, stride_VD,
    stride_OB, stride_OH, stride_Os, stride_OD,
    stride_LB, stride_LH, stride_LS,
    stride_dQB, stride_dQH, stride_dQS, stride_dQD,
    stride_dKB, stride_dKH, stride_dKS, stride_dKD,
    stride_dVB, stride_dVH, stride_dVS, stride_dVD,
    B, H, S, D, inv_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """Compute dK, reading C from dV[:,:,:0]."""
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    n_sm = tl.cdiv(S, BLOCK_M)
    total_bh = B * H

    if pid_m >= n_sm or pid_bh >= total_bh:
        return

    b = pid_bh // H
    h = pid_bh % H

    base_q = Q + b * stride_QB + h * stride_QH
    base_k = K + b * stride_KB + h * stride_KH
    base_v = V + b * stride_VB + h * stride_VH
    base_do = dO + b * stride_OB + h * stride_OH
    base_l = L + b * stride_LB + h * stride_LH
    base_dk = dK + b * stride_dKB + h * stride_dKH
    base_dv = dV + b * stride_dVB + h * stride_dVH

    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_d = tl.arange(0, BLOCK_D)
    m_mask = off_m < S
    d_mask = off_d < D
    md_mask = m_mask[:, None] & d_mask[None, :]

    K_tile = tl.load(base_k + off_m[:, None] * stride_KS + off_d[None, :] * stride_KD,
                     mask=md_mask, other=0.0)
    V_tile = tl.load(base_v + off_m[:, None] * stride_VS + off_d[None, :] * stride_VD,
                     mask=md_mask, other=0.0)

    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    n_sn = tl.cdiv(S, BLOCK_N)
    for sn in range(n_sn):
        off_n = sn * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = off_n < S
        nd_mask = n_mask[:, None] & d_mask[None, :]

        Q_tile = tl.load(base_q + off_n[:, None] * stride_QS + off_d[None, :] * stride_QD,
                         mask=nd_mask, other=0.0)
        dO_tile = tl.load(base_do + off_n[:, None] * stride_Os + off_d[None, :] * stride_OD,
                          mask=nd_mask, other=0.0)
        L_vals = tl.load(base_l + off_n * stride_LS, mask=n_mask, other=0.0)
        C_vals = tl.load(base_dv + off_n * stride_dVS, mask=n_mask, other=0.0)

        S_mat = tl.dot(Q_tile, K_tile.T) * inv_scale
        P_mat = tl.exp(S_mat - L_vals[:, None])
        dOV = tl.dot(dO_tile.to(tl.float32), V_tile.to(tl.float32).T)

        d_logits = P_mat * (dOV - C_vals[:, None])
        acc = tl.dot(d_logits.T, Q_tile.to(tl.float32), acc)

    acc *= inv_scale
    tl.store(base_dk + off_m[:, None] * stride_dKS + off_d[None, :] * stride_dKD,
             acc.to(tl.bfloat16), mask=md_mask)


@triton.jit
def _dV_full(
    Q, K, dO, L, dV,
    stride_QB, stride_QH, stride_QS, stride_QD,
    stride_KB, stride_KH, stride_KS, stride_KD,
    stride_OB, stride_OH, stride_Os, stride_OD,
    stride_LB, stride_LH, stride_LS,
    stride_dVB, stride_dVH, stride_dVS, stride_dVD,
    B, H, S, D, inv_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """Compute dV, overwriting C stash."""
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    n_sm = tl.cdiv(S, BLOCK_M)
    total_bh = B * H

    if pid_m >= n_sm or pid_bh >= total_bh:
        return

    b = pid_bh // H
    h = pid_bh % H

    base_q = Q + b * stride_QB + h * stride_QH
    base_k = K + b * stride_KB + h * stride_KH
    base_do = dO + b * stride_OB + h * stride_OH
    base_l = L + b * stride_LB + h * stride_LH
    base_dv = dV + b * stride_dVB + h * stride_dVH

    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_d = tl.arange(0, BLOCK_D)
    m_mask = off_m < S
    d_mask = off_d < D
    md_mask = m_mask[:, None] & d_mask[None, :]

    K_tile = tl.load(base_k + off_m[:, None] * stride_KS + off_d[None, :] * stride_KD,
                     mask=md_mask, other=0.0)

    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    n_sn = tl.cdiv(S, BLOCK_N)
    for sn in range(n_sn):
        off_n = sn * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = off_n < S
        nd_mask = n_mask[:, None] & d_mask[None, :]

        Q_tile = tl.load(base_q + off_n[:, None] * stride_QS + off_d[None, :] * stride_QD,
                         mask=nd_mask, other=0.0)
        dO_tile = tl.load(base_do + off_n[:, None] * stride_Os + off_d[None, :] * stride_OD,
                          mask=nd_mask, other=0.0)
        L_vals = tl.load(base_l + off_n * stride_LS, mask=n_mask, other=0.0)

        S_mat = tl.dot(Q_tile, K_tile.T) * inv_scale
        P_mat = tl.exp(S_mat - L_vals[:, None])

        acc = tl.dot(P_mat.T, dO_tile.to(tl.float32), acc)

    tl.store(base_dv + off_m[:, None] * stride_dVS + off_d[None, :] * stride_dVD,
             acc.to(tl.bfloat16), mask=md_mask)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Multi-head attention backward: compute dQ, dK, dV."""
    torch.cuda.set_device(Q.device)

    B, H, S, D = Q.shape
    inv_scale = 1.0 / math.sqrt(D)

    if L.dim() == 3:
        L = L.unsqueeze(-1)

    sq = list(Q.stride())
    sk = list(K.stride())
    sv = list(V.stride())
    sdo = list(dO.stride())
    sl = list(L.stride())
    sdq = list(dQ.stride())
    sdk = list(dK.stride())
    sdv = list(dV.stride())

    BLOCK_M = 32
    BLOCK_N = 64
    BLOCK_D = 128

    n_sm = triton.cdiv(S, BLOCK_M)
    total_bh = B * H

    grid = (n_sm, total_bh)

    def _launch_dQ():
        _dQ_full[grid](
            Q, K, V, dO, L, dQ, dK, dV,
            sq[0], sq[1], sq[2], sq[3],
            sk[0], sk[1], sk[2], sk[3],
            sv[0], sv[1], sv[2], sv[3],
            sdo[0], sdo[1], sdo[2], sdo[3],
            sl[0], sl[1], sl[2],
            sdq[0], sdq[1], sdq[2], sdq[3],
            sdk[0], sdk[1], sdk[2], sdk[3],
            sdv[0], sdv[1], sdv[2], sdv[3],
            B, H, S, D, inv_scale,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
            num_warps=4, num_stages=3)

    def _launch_dK():
        _dK_full[grid](
            Q, K, V, dO, L, dQ, dK, dV,
            sq[0], sq[1], sq[2], sq[3],
            sk[0], sk[1], sk[2], sk[3],
            sv[0], sv[1], sv[2], sv[3],
            sdo[0], sdo[1], sdo[2], sdo[3],
            sl[0], sl[1], sl[2],
            sdq[0], sdq[1], sdq[2], sdq[3],
            sdk[0], sdk[1], sdk[2], sdk[3],
            sdv[0], sdv[1], sdv[2], sdv[3],
            B, H, S, D, inv_scale,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
            num_warps=4, num_stages=3)

    def _launch_dV():
        _dV_full[grid](
            Q, K, dO, L, dV,
            sq[0], sq[1], sq[2], sq[3],
            sk[0], sk[1], sk[2], sk[3],
            sdo[0], sdo[1], sdo[2], sdo[3],
            sl[0], sl[1], sl[2],
            sdv[0], sdv[1], sdv[2], sdv[3],
            B, H, S, D, inv_scale,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
            num_warps=4, num_stages=3)

    # Phase 1: dQ + stash C
    _launch_dQ()

    # Phase 2: dK (reads C from dV)
    _launch_dK()

    # Phase 3: dV (overwrites C stash)
    _launch_dV()