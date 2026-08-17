import torch
import triton
import triton.language as tl
import math


@triton.jit
def _dV_kernel(
    Q, K, dO, L, dV,
    stride_QB, stride_QH, stride_QS, stride_QD,
    stride_KB, stride_KH, stride_KS, stride_KD,
    stride_dOB, stride_dOH, stride_dOS, stride_dOD,
    stride_LB, stride_LH, stride_LS,
    stride_dVB, stride_dVH, stride_dVS, stride_dVD,
    B, H, S, D, inv_scale,
    BLOCK_SM: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """dV[k,d] = sum_q P[q,k] * dO[q,d], iterating q tiles, fixing k."""
    pid_k = tl.program_id(0)
    pid_bh = tl.program_id(1)

    n_pk = tl.cdiv(S, BLOCK_SM)
    n_bh = B * H

    if pid_k >= n_pk or pid_bh >= n_bh:
        return

    b = pid_bh // H
    h = pid_bh % H

    off_k = pid_k * BLOCK_SM + tl.arange(0, BLOCK_SM)
    off_d = tl.arange(0, BLOCK_D)

    k_mask_2d = (off_k[:, None] < S) & (off_d[None, :] < D)

    K_base = K + b * stride_KB + h * stride_KH
    K_tile = tl.load(K_base + off_k[:, None] * stride_KS + off_d[None, :] * stride_KD,
                     mask=k_mask_2d, other=0.0)

    acc = tl.zeros((BLOCK_SM, BLOCK_D), dtype=tl.float32)

    n_pq = tl.cdiv(S, BLOCK_SM)
    for pq in range(n_pq):
        off_q = pq * BLOCK_SM + tl.arange(0, BLOCK_SM)
        q_mask_2d = (off_q[:, None] < S) & (off_d[None, :] < D)

        Q_base = Q + b * stride_QB + h * stride_QH
        Q_tile = tl.load(Q_base + off_q[:, None] * stride_QS + off_d[None, :] * stride_QD,
                         mask=q_mask_2d, other=0.0)

        S_mat = tl.dot(Q_tile, K_tile.T) * inv_scale

        L_base = L + b * stride_LB + h * stride_LH
        L_vals = tl.load(L_base + off_q * stride_LS, mask=(off_q < S), other=0.0)
        P_mat = tl.exp(S_mat - L_vals[:, None])

        dO_base = dO + b * stride_dOB + h * stride_dOH
        dO_tile = tl.load(dO_base + off_q[:, None] * stride_dOS + off_d[None, :] * stride_dOD,
                          mask=q_mask_2d, other=0.0)

        acc = tl.dot(P_mat.T, dO_tile.to(tl.float32), acc)

    dV_base = dV + b * stride_dVB + h * stride_dVH
    tl.store(dV_base + off_k[:, None] * stride_dVS + off_d[None, :] * stride_dVD,
             acc.to(tl.bfloat16), mask=k_mask_2d)


@triton.jit
def _dQ_C_kernel(
    Q, K, V, dO, L, dQ, dK, dV,
    stride_QB, stride_QH, stride_QS, stride_QD,
    stride_KB, stride_KH, stride_KS, stride_KD,
    stride_VB, stride_VH, stride_VS, stride_VD,
    stride_dOB, stride_dOH, stride_dOS, stride_dOD,
    stride_LB, stride_LH, stride_LS,
    stride_dQB, stride_dQH, stride_dQS, stride_dQD,
    stride_dKB, stride_dKH, stride_dKS, stride_dKD,
    stride_dVB, stride_dVH, stride_dVS, stride_dVD,
    B, H, S, D, inv_scale,
    BLOCK_SM: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """Compute dQ via decomposition and stash C[b,h,q] into dV[b,h,q,0].
    
    dQ[q,d] = (A[q,d] - C[q]*E[q,d]) * inv_scale
      A[q,d] = sum_k P[q,k] * dOV[q,k] * K[k,d]
      E[q,d] = sum_k P[q,k] * K[k,d]
      C[q]   = sum_k P[q,k] * dOV[q,k]
    """
    pid_q = tl.program_id(0)
    pid_bh = tl.program_id(1)

    n_pq = tl.cdiv(S, BLOCK_SM)
    n_bh = B * H

    if pid_q >= n_pq or pid_bh >= n_bh:
        return

    b = pid_bh // H
    h = pid_bh % H

    off_q = pid_q * BLOCK_SM + tl.arange(0, BLOCK_SM)
    off_d = tl.arange(0, BLOCK_D)
    q_mask_2d = (off_q[:, None] < S) & (off_d[None, :] < D)

    Q_base = Q + b * stride_QB + h * stride_QH
    Q_tile = tl.load(Q_base + off_q[:, None] * stride_QS + off_d[None, :] * stride_QD,
                     mask=q_mask_2d, other=0.0)

    L_base = L + b * stride_LB + h * stride_LH
    L_vals = tl.load(L_base + off_q * stride_LS, mask=(off_q < S), other=0.0)

    dO_base = dO + b * stride_dOB + h * stride_dOH
    dO_tile = tl.load(dO_base + off_q[:, None] * stride_dOS + off_d[None, :] * stride_dOD,
                      mask=q_mask_2d, other=0.0)

    acc_A = tl.zeros((BLOCK_SM, BLOCK_D), dtype=tl.float32)
    acc_E = tl.zeros((BLOCK_SM, BLOCK_D), dtype=tl.float32)
    acc_C = tl.zeros((BLOCK_SM,), dtype=tl.float32)

    n_pk = tl.cdiv(S, BLOCK_SM)
    for pk in range(n_pk):
        off_k = pk * BLOCK_SM + tl.arange(0, BLOCK_SM)
        k_mask_2d = (off_k[:, None] < S) & (off_d[None, :] < D)

        K_base = K + b * stride_KB + h * stride_KH
        K_tile = tl.load(K_base + off_k[:, None] * stride_KS + off_d[None, :] * stride_KD,
                         mask=k_mask_2d, other=0.0)

        S_mat = tl.dot(Q_tile, K_tile.T) * inv_scale
        P_mat = tl.exp(S_mat - L_vals[:, None])

        V_base = V + b * stride_VB + h * stride_VH
        V_tile = tl.load(V_base + off_k[:, None] * stride_VS + off_d[None, :] * stride_VD,
                         mask=k_mask_2d, other=0.0)

        dOV = tl.dot(dO_tile.to(tl.float32), V_tile.to(tl.float32).T)
        weight = P_mat * dOV

        acc_A = tl.dot(weight, K_tile.to(tl.float32), acc_A)
        acc_E = tl.dot(P_mat, K_tile.to(tl.float32), acc_E)
        acc_C = acc_C + tl.sum(weight, axis=1)

    dQ_val = (acc_A - acc_C[:, None] * acc_E) * inv_scale

    dQ_base = dQ + b * stride_dQB + h * stride_dQH
    tl.store(dQ_base + off_q[:, None] * stride_dQS + off_d[None, :] * stride_dQD,
             dQ_val.to(tl.bfloat16), mask=q_mask_2d)

    # Stash C into dV[:,:,:0] for dK kernel
    dV_base = dV + b * stride_dVB + h * stride_dVH
    tl.store(dV_base + off_q * stride_dVS, acc_C.to(tl.bfloat16), mask=(off_q < S))


@triton.jit
def _dK_kernel(
    Q, K, V, dO, L, dQ, dK, dV,
    stride_QB, stride_QH, stride_QS, stride_QD,
    stride_KB, stride_KH, stride_KS, stride_KD,
    stride_VB, stride_VH, stride_VS, stride_VD,
    stride_dOB, stride_dOH, stride_dOS, stride_dOD,
    stride_LB, stride_LH, stride_LS,
    stride_dQB, stride_dQH, stride_dQS, stride_dQD,
    stride_dKB, stride_dKH, stride_dKS, stride_dKD,
    stride_dVB, stride_dVH, stride_dVS, stride_dVD,
    B, H, S, D, inv_scale,
    BLOCK_SM: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """dK[k,d] = sum_q P[q,k]*(dOV[q,k]-C[q])*Q[q,d] * inv_scale
    
    Reads C[b,h,q] from dV[b,h,q,0] written by _dQ_C_kernel.
    """
    pid_k = tl.program_id(0)
    pid_bh = tl.program_id(1)

    n_pk = tl.cdiv(S, BLOCK_SM)
    n_bh = B * H

    if pid_k >= n_pk or pid_bh >= n_bh:
        return

    b = pid_bh // H
    h = pid_bh % H

    off_k = pid_k * BLOCK_SM + tl.arange(0, BLOCK_SM)
    off_d = tl.arange(0, BLOCK_D)
    k_mask_2d = (off_k[:, None] < S) & (off_d[None, :] < D)

    K_base = K + b * stride_KB + h * stride_KH
    K_tile = tl.load(K_base + off_k[:, None] * stride_KS + off_d[None, :] * stride_KD,
                     mask=k_mask_2d, other=0.0)

    V_base = V + b * stride_VB + h * stride_VH
    V_tile = tl.load(V_base + off_k[:, None] * stride_VS + off_d[None, :] * stride_VD,
                     mask=k_mask_2d, other=0.0)

    acc = tl.zeros((BLOCK_SM, BLOCK_D), dtype=tl.float32)

    n_pq = tl.cdiv(S, BLOCK_SM)
    for pq in range(n_pq):
        off_q = pq * BLOCK_SM + tl.arange(0, BLOCK_SM)
        q_mask_2d = (off_q[:, None] < S) & (off_d[None, :] < D)

        Q_base = Q + b * stride_QB + h * stride_QH
        Q_tile = tl.load(Q_base + off_q[:, None] * stride_QS + off_d[None, :] * stride_QD,
                         mask=q_mask_2d, other=0.0)

        dO_base = dO + b * stride_dOB + h * stride_dOH
        dO_tile = tl.load(dO_base + off_q[:, None] * stride_dOS + off_d[None, :] * stride_dOD,
                          mask=q_mask_2d, other=0.0)

        L_base = L + b * stride_LB + h * stride_LH
        L_vals = tl.load(L_base + off_q * stride_LS, mask=(off_q < S), other=0.0)

        # Load C stashed in dV[:,:,:0]
        dV_base = dV + b * stride_dVB + h * stride_dVH
        C_vals = tl.load(dV_base + off_q * stride_dVS, mask=(off_q < S), other=0.0)

        S_mat = tl.dot(Q_tile, K_tile.T) * inv_scale
        P_mat = tl.exp(S_mat - L_vals[:, None])

        dOV = tl.dot(dO_tile.to(tl.float32), V_tile.to(tl.float32).T)

        d_logits = P_mat * (dOV - C_vals[:, None])

        acc = tl.dot(d_logits.T, Q_tile.to(tl.float32), acc)

    acc *= inv_scale

    dK_base = dK + b * stride_dKB + h * stride_dKH
    tl.store(dK_base + off_k[:, None] * stride_dKS + off_d[None, :] * stride_dKD,
             acc.to(tl.bfloat16), mask=k_mask_2d)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Multi-head attention backward: compute dQ, dK, dV.
    
    Three-phase execution:
      1. _dQ_C_kernel → writes dQ, stashes C[b,h,q] into dV[b,h,q,0]
      2. _dK_kernel   → reads C from dV, writes dK
      3. _dV_kernel   → overwrites entire dV with true dV gradients
    """
    torch.cuda.set_device(Q.device)

    B, H, S, D = Q.shape

    inv_scale = 1.0 / math.sqrt(D)

    BLOCK_SM = 32
    BLOCK_D = 128

    n_seq = triton.cdiv(S, BLOCK_SM)
    n_bh = B * H
    grid = (n_seq, n_bh)

    # Ensure L has consistent dimensions for stride access
    if L.dim() == 3:
        L_view = L.unsqueeze(-1)
    else:
        L_view = L

    sQ = list(Q.stride())
    sK = list(K.stride())
    sV = list(V.stride())
    sdO = list(dO.stride())
    sL = list(L_view.stride())
    sdQ = list(dQ.stride())
    sdK = list(dK.stride())
    sdV = list(dV.stride())

    launch_kwargs = dict(BLOCK_SM=BLOCK_SM, BLOCK_D=BLOCK_D, num_warps=4, num_stages=3)

    # Phase 1: compute dQ and stash C
    _dQ_C_kernel[grid](
        Q, K, V, dO, L_view, dQ, dK, dV,
        sQ[0], sQ[1], sQ[2], sQ[3],
        sK[0], sK[1], sK[2], sK[3],
        sV[0], sV[1], sV[2], sV[3],
        sdO[0], sdO[1], sdO[2], sdO[3],
        sL[0], sL[1], sL[2],
        sdQ[0], sdQ[1], sdQ[2], sdQ[3],
        sdK[0], sdK[1], sdK[2], sdK[3],
        sdV[0], sdV[1], sdV[2], sdV[3],
        B, H, S, D, inv_scale,
        **launch_kwargs,
    )

    # Phase 2: compute dK (consumes C from dV)
    _dK_kernel[grid](
        Q, K, V, dO, L_view, dQ, dK, dV,
        sQ[0], sQ[1], sQ[2], sQ[3],
        sK[0], sK[1], sK[2], sK[3],
        sV[0], sV[1], sV[2], sV[3],
        sdO[0], sdO[1], sdO[2], sdO[3],
        sL[0], sL[1], sL[2],
        sdQ[0], sdQ[1], sdQ[2], sdQ[3],
        sdK[0], sdK[1], sdK[2], sdK[3],
        sdV[0], sdV[1], sdV[2], sdV[3],
        B, H, S, D, inv_scale,
        **launch_kwargs,
    )

    # Phase 3: compute dV (overwrites dV entirely, erasing C stash)
    _dV_kernel[grid](
        Q, K, dO, L_view, dV,
        sQ[0], sQ[1], sQ[2], sQ[3],
        sK[0], sK[1], sK[2], sK[3],
        sdO[0], sdO[1], sdO[2], sdO[3],
        sL[0], sL[1], sL[2],
        sdV[0], sdV[1], sdV[2], sdV[3],
        B, H, S, D, inv_scale,
        **launch_kwargs,
    )