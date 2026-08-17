import torch
import triton
import triton.language as tl
import math


@triton.jit
def _phase1_kernel(
    Q_base, K_base, dO_base, V_base, L_base,
    dQ_base, dV_base,
    stride_QS, stride_QD, stride_KS, stride_KD,
    stride_Os, stride_OD, stride_VS, stride_VD,
    stride_LS, stride_dQS, stride_dQD,
    stride_dVS,
    S, D, inv_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """Compute dQ for one (b,h). Stash C[b,h,:] into dV[b,h,:,0]."""
    pid_m = tl.program_id(0)
    
    n_sm = tl.cdiv(S, BLOCK_M)
    if pid_m >= n_sm:
        return
    
    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_d = tl.arange(0, BLOCK_D)
    m_mask = off_m < S
    d_mask = off_d < D
    md_mask = m_mask[:, None] & d_mask[None, :]

    Q_tile = tl.load(Q_base + off_m[:, None] * stride_QS + off_d[None, :] * stride_QD,
                     mask=md_mask, other=0.0)
    dO_tile = tl.load(dO_base + off_m[:, None] * stride_Os + off_d[None, :] * stride_OD,
                      mask=md_mask, other=0.0)
    L_vals = tl.load(L_base + off_m * stride_LS, mask=m_mask, other=0.0)

    acc_A = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    acc_E = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    acc_C = tl.zeros((BLOCK_M,), dtype=tl.float32)

    n_sn = tl.cdiv(S, BLOCK_N)
    for sn in range(n_sn):
        off_n = sn * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = off_n < S
        nd_mask = n_mask[:, None] & d_mask[None, :]

        K_tile = tl.load(K_base + off_n[:, None] * stride_KS + off_d[None, :] * stride_KD,
                         mask=nd_mask, other=0.0)
        V_tile = tl.load(V_base + off_n[:, None] * stride_VS + off_d[None, :] * stride_VD,
                         mask=nd_mask, other=0.0)

        S_mat = tl.dot(Q_tile, K_tile.T) * inv_scale
        P_mat = tl.exp(S_mat - L_vals[:, None])

        dOV = tl.dot(dO_tile.to(tl.float32), V_tile.to(tl.float32).T)
        weight = P_mat * dOV

        acc_A = tl.dot(weight, K_tile.to(tl.float32), acc_A)
        acc_E = tl.dot(P_mat, K_tile.to(tl.float32), acc_E)
        acc_C += tl.sum(weight, axis=1)

    dQ_val = (acc_A - acc_C[:, None] * acc_E) * inv_scale
    tl.store(dQ_base + off_m[:, None] * stride_dQS + off_d[None, :] * stride_dQD,
             dQ_val.to(tl.bfloat16), mask=md_mask)

    tl.store(dV_base + off_m * stride_dVS, acc_C.to(tl.bfloat16), mask=m_mask)


@triton.jit
def _phase2_kernel(
    Q_base, K_base, dO_base, V_base, L_base,
    dK_base, dV_base,
    stride_QS, stride_QD, stride_KS, stride_KD,
    stride_Os, stride_OD, stride_VS, stride_VD,
    stride_LS, stride_dKS, stride_dKD,
    stride_dVS,
    S, D, inv_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """Compute dK for one (b,h). Reads C from dV[:,:,:0]."""
    pid_m = tl.program_id(0)
    
    n_sm = tl.cdiv(S, BLOCK_M)
    if pid_m >= n_sm:
        return
    
    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_d = tl.arange(0, BLOCK_D)
    m_mask = off_m < S
    d_mask = off_d < D
    md_mask = m_mask[:, None] & d_mask[None, :]

    K_tile = tl.load(K_base + off_m[:, None] * stride_KS + off_d[None, :] * stride_KD,
                     mask=md_mask, other=0.0)
    V_tile = tl.load(V_base + off_m[:, None] * stride_VS + off_d[None, :] * stride_VD,
                     mask=md_mask, other=0.0)

    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    n_sn = tl.cdiv(S, BLOCK_N)
    for sn in range(n_sn):
        off_n = sn * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = off_n < S
        nd_mask = n_mask[:, None] & d_mask[None, :]

        Q_tile = tl.load(Q_base + off_n[:, None] * stride_QS + off_d[None, :] * stride_QD,
                         mask=nd_mask, other=0.0)
        dO_tile = tl.load(dO_base + off_n[:, None] * stride_Os + off_d[None, :] * stride_OD,
                          mask=nd_mask, other=0.0)
        L_vals = tl.load(L_base + off_n * stride_LS, mask=n_mask, other=0.0)
        C_vals = tl.load(dV_base + off_n * stride_dVS, mask=n_mask, other=0.0)

        S_mat = tl.dot(Q_tile, K_tile.T) * inv_scale
        P_mat = tl.exp(S_mat - L_vals[:, None])
        dOV = tl.dot(dO_tile.to(tl.float32), V_tile.to(tl.float32).T)

        d_logits = P_mat * (dOV - C_vals[:, None])
        acc = tl.dot(d_logits.T, Q_tile.to(tl.float32), acc)

    acc *= inv_scale
    tl.store(dK_base + off_m[:, None] * stride_dKS + off_d[None, :] * stride_dKD,
             acc.to(tl.bfloat16), mask=md_mask)


@triton.jit
def _phase3_kernel(
    Q_base, K_base, dO_base, L_base, dV_base,
    stride_QS, stride_QD, stride_KS, stride_KD,
    stride_Os, stride_OD, stride_LS,
    stride_dVS, stride_dVD,
    S, D, inv_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """Compute dV for one (b,h). Overwrites C stash."""
    pid_m = tl.program_id(0)
    
    n_sm = tl.cdiv(S, BLOCK_M)
    if pid_m >= n_sm:
        return
    
    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_d = tl.arange(0, BLOCK_D)
    m_mask = off_m < S
    d_mask = off_d < D
    md_mask = m_mask[:, None] & d_mask[None, :]

    K_tile = tl.load(K_base + off_m[:, None] * stride_KS + off_d[None, :] * stride_KD,
                     mask=md_mask, other=0.0)

    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    n_sn = tl.cdiv(S, BLOCK_N)
    for sn in range(n_sn):
        off_n = sn * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = off_n < S
        nd_mask = n_mask[:, None] & d_mask[None, :]

        Q_tile = tl.load(Q_base + off_n[:, None] * stride_QS + off_d[None, :] * stride_QD,
                         mask=nd_mask, other=0.0)
        dO_tile = tl.load(dO_base + off_n[:, None] * stride_Os + off_d[None, :] * stride_OD,
                          mask=nd_mask, other=0.0)
        L_vals = tl.load(L_base + off_n * stride_LS, mask=n_mask, other=0.0)

        S_mat = tl.dot(Q_tile, K_tile.T) * inv_scale
        P_mat = tl.exp(S_mat - L_vals[:, None])

        acc = tl.dot(P_mat.T, dO_tile.to(tl.float32), acc)

    tl.store(dV_base + off_m[:, None] * stride_dVS + off_d[None, :] * stride_dVD,
             acc.to(tl.bfloat16), mask=md_mask)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Multi-head attention backward: compute dQ, dK, dV."""
    torch.cuda.set_device(Q.device)

    B, H, S, D = Q.shape
    inv_scale = 1.0 / math.sqrt(D)

    if L.dim() == 3:
        L = L.unsqueeze(-1)

    sQ = Q.stride()
    sK = K.stride()
    sV = V.stride()
    sdO = dO.stride()
    sL = L.stride()
    sdQ = dQ.stride()
    sdK = dK.stride()
    sdV = dV.stride()

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = 128

    n_sm = triton.cdiv(S, BLOCK_M)

    kw = dict(BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
              num_warps=4, num_stages=3)

    for b in range(B):
        for h in range(H):
            Q_base = Q + b * sQ[0] + h * sQ[1]
            K_base = K + b * sK[0] + h * sK[1]
            V_base = V + b * sV[0] + h * sV[1]
            dO_base = dO + b * sdO[0] + h * sdO[1]
            L_base = L + b * sL[0] + h * sL[1]
            dQ_base = dQ + b * sdQ[0] + h * sdQ[1]
            dK_base = dK + b * sdK[0] + h * sdK[1]
            dV_base = dV + b * sdV[0] + h * sdV[1]

            grid = (n_sm,)

            _phase1_kernel[grid](
                Q_base, K_base, dO_base, V_base, L_base,
                dQ_base, dV_base,
                sQ[2], sQ[3], sK[2], sK[3],
                sdO[2], sdO[3], sV[2], sV[3],
                sL[2], sdQ[2], sdQ[3],
                sdV[2],
                S, D, inv_scale, **kw,
            )

            _phase2_kernel[grid](
                Q_base, K_base, dO_base, V_base, L_base,
                dK_base, dV_base,
                sQ[2], sQ[3], sK[2], sK[3],
                sdO[2], sdO[3], sV[2], sV[3],
                sL[2], sdK[2], sdK[3],
                sdV[2],
                S, D, inv_scale, **kw,
            )

            _phase3_kernel[grid](
                Q_base, K_base, dO_base, L_base, dV_base,
                sQ[2], sQ[3], sK[2], sK[3],
                sdO[2], sdO[3], sL[2],
                sdV[2], sdV[3],
                S, D, inv_scale, **kw,
            )