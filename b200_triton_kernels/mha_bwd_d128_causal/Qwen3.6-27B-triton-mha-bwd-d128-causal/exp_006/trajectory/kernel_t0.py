import torch
import triton
import triton.language as tl
import math


@triton.jit
def _dq_kernel(
    Q, K, V, dO, O, L, dQ_out,
    SB_Q, SH_Q, SS_Q, SD_Q,
    SB_K, SH_K, SS_K, SD_K,
    SB_V, SH_V, SS_V, SD_V,
    SB_dO, SH_dO, SS_dO, SD_dO,
    SB_O, SH_O, SS_O, SD_O,
    SB_L, SH_L, SS_L,
    SB_dQ, SH_dQ, SS_dQ, SD_dQ,
    S, D, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)

    im = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    id_ = tl.arange(0, BLOCK_D)
    mm = im < S
    dm = id_ < D

    bq = pid_b * SB_Q + pid_h * SH_Q
    QT = tl.load(Q + bq + im[:, None] * SS_Q + id_[None, :] * SD_Q,
                 mask=mm[:, None] & dm[None, :], other=0.0).to(tl.float32)

    bq_dO = pid_b * SB_dO + pid_h * SH_dO
    dOT = tl.load(dO + bq_dO + im[:, None] * SS_dO + id_[None, :] * SD_dO,
                  mask=mm[:, None] & dm[None, :], other=0.0).to(tl.float32)

    bq_O = pid_b * SB_O + pid_h * SH_O
    OT = tl.load(O + bq_O + im[:, None] * SS_O + id_[None, :] * SD_O,
                 mask=mm[:, None] & dm[None, :], other=0.0).to(tl.float32)

    bq_L = pid_b * SB_L + pid_h * SH_L
    LT = tl.load(L + bq_L + im * SS_L, mask=mm, other=0.0)

    DT = tl.sum(dOT * OT, axis=1)

    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    nk = tl.cdiv(S, BLOCK_N)
    for pn in range(nk):
        in_ = pn * BLOCK_N + tl.arange(0, BLOCK_N)
        nm = in_ < S

        bk = pid_b * SB_K + pid_h * SH_K
        KT = tl.load(K + bk + in_[:, None] * SS_K + id_[None, :] * SD_K,
                     mask=nm[:, None] & dm[None, :], other=0.0).to(tl.float32)

        bv = pid_b * SB_V + pid_h * SH_V
        VT = tl.load(V + bv + in_[:, None] * SS_V + id_[None, :] * SD_V,
                     mask=nm[:, None] & dm[None, :], other=0.0).to(tl.float32)

        sc = tl.dot(QT, KT.T) * scale

        cm = (in_[None, :] <= im[:, None])
        am = cm & nm[None, :] & mm[:, None]

        P = tl.where(am, tl.exp(sc - LT[:, None]), 0.0)
        dP = tl.dot(dOT, VT.T)
        dS = P * (dP - DT[:, None])

        acc = tl.dot(dS * scale, KT, acc=acc)

    bdQ = pid_b * SB_dQ + pid_h * SH_dQ
    tl.store(dQ_out + bdQ + im[:, None] * SS_dQ + id_[None, :] * SD_dQ,
             acc.to(tl.bfloat16), mask=mm[:, None] & dm[None, :])


@triton.jit
def _dkv_kernel(
    Q, K, V, dO, O, L, dK_out, dV_out,
    SB_Q, SH_Q, SS_Q, SD_Q,
    SB_K, SH_K, SS_K, SD_K,
    SB_V, SH_V, SS_V, SD_V,
    SB_dO, SH_dO, SS_dO, SD_dO,
    SB_O, SH_O, SS_O, SD_O,
    SB_L, SH_L, SS_L,
    SB_dK, SH_dK, SS_dK, SD_dK,
    SB_dV, SH_dV, SS_dV, SD_dV,
    S, D, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)

    in_ = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    id_ = tl.arange(0, BLOCK_D)
    nm = in_ < S
    dm = id_ < D

    bk = pid_b * SB_K + pid_h * SH_K
    KT = tl.load(K + bk + in_[:, None] * SS_K + id_[None, :] * SD_K,
                 mask=nm[:, None] & dm[None, :], other=0.0).to(tl.float32)

    bv = pid_b * SB_V + pid_h * SH_V
    VT = tl.load(V + bv + in_[:, None] * SS_V + id_[None, :] * SD_V,
                 mask=nm[:, None] & dm[None, :], other=0.0).to(tl.float32)

    accK = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)
    accV = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)

    nq = tl.cdiv(S, BLOCK_M)
    for pm in range(nq):
        im = pm * BLOCK_M + tl.arange(0, BLOCK_M)
        mm = im < S

        bq = pid_b * SB_Q + pid_h * SH_Q
        QT = tl.load(Q + bq + im[:, None] * SS_Q + id_[None, :] * SD_Q,
                     mask=mm[:, None] & dm[None, :], other=0.0).to(tl.float32)

        bq_dO = pid_b * SB_dO + pid_h * SH_dO
        dOT = tl.load(dO + bq_dO + im[:, None] * SS_dO + id_[None, :] * SD_dO,
                      mask=mm[:, None] & dm[None, :], other=0.0).to(tl.float32)

        bq_O = pid_b * SB_O + pid_h * SH_O
        OT = tl.load(O + bq_O + im[:, None] * SS_O + id_[None, :] * SD_O,
                     mask=mm[:, None] & dm[None, :], other=0.0).to(tl.float32)

        bq_L = pid_b * SB_L + pid_h * SH_L
        LT = tl.load(L + bq_L + im * SS_L, mask=mm, other=0.0)

        DT = tl.sum(dOT * OT, axis=1)

        sc = tl.dot(QT, KT.T) * scale

        cm = (in_[None, :] <= im[:, None])
        am = cm & nm[None, :] & mm[:, None]

        P = tl.where(am, tl.exp(sc - LT[:, None]), 0.0)
        dP = tl.dot(dOT, VT.T)
        dS = P * (dP - DT[:, None])

        accK = tl.dot(dS.T * scale, QT, acc=accK)
        accV = tl.dot(P.T, dOT, acc=accV)

    bdK = pid_b * SB_dK + pid_h * SH_dK
    tl.store(dK_out + bdK + in_[:, None] * SS_dK + id_[None, :] * SD_dK,
             accK.to(tl.bfloat16), mask=nm[:, None] & dm[None, :])

    bdV = pid_b * SB_dV + pid_h * SH_dV
    tl.store(dV_out + bdV + in_[:, None] * SS_dV + id_[None, :] * SD_dV,
             accV.to(tl.bfloat16), mask=nm[:, None] & dm[None, :])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Causal MHA backward: compute dQ, dK, dV from Q, K, V, O, dO, L."""
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    scale = 1.0 / math.sqrt(d)

    BM = 64
    BN = 64
    BD = d

    g_dq = (triton.cdiv(S, BM), H, B)
    g_dkv = (triton.cdiv(S, BN), H, B)

    _dq_kernel[g_dq](
        Q, K, V, dO, O, L, dQ,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        S, d, scale,
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_D=BD,
        num_warps=4, num_stages=3,
    )

    _dkv_kernel[g_dkv](
        Q, K, V, dO, O, L, dK, dV,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        S, d, scale,
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_D=BD,
        num_warps=4, num_stages=3,
    )