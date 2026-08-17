import torch
import triton
import triton.language as tl
import math


@triton.jit
def _dV_kernel(
    Q, K, V, dO, L, dQ, dK, dV,
    stride_QB, stride_QH, stride_QS, stride_QD,
    stride_KB, stride_KH, stride_KS, stride_KD,
    stride_VB, stride_VH, stride_VS, stride_VD,
    stride_dOB, stride_dOH, stride_dOS, stride_dOD,
    stride_LB, stride_LH, stride_LS,
    stride_dQB, stride_dQH, stride_dQS, stride_dQD,
    stride_dKB, stride_dKH, stride_dKS, stride_dKD,
    stride_dVB, stride_dVH, stride_dVS, stride_dVD,
    B, H, S, D,
    inv_scale,
    BLOCK_SM: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """dV[k,d] = sum_q P[q,k] * dO[q,d]  (iterate q, fix k)"""
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

    k_mask = (off_k[:, None] < S) & (off_d[None, :] < D)

    K_ptr = K + b * stride_KB + h * stride_KH
    K_tile = tl.load(K_ptr + off_k[:, None] * stride_KS + off_d[None, :] * stride_KD,
                     mask=k_mask, other=0.0)

    acc = tl.zeros((BLOCK_SM, BLOCK_D), dtype=tl.float32)

    n_pq = tl.cdiv(S, BLOCK_SM)
    for pq in range(n_pq):
        off_q = pq * BLOCK_SM + tl.arange(0, BLOCK_SM)

        Q_ptr = Q + b * stride_QB + h * stride_QH
        Q_tile = tl.load(
            Q_ptr + off_q[:, None] * stride_QS + off_d[None, :] * stride_QD,
            mask=((off_q[:, None] < S) & (off_d[None, :] < D)), other=0.0)

        S_mat = tl.dot(Q_tile, K_tile.T) * inv_scale

        L_ptr = L + b * stride_LB + h * stride_LH
        L_vals = tl.load(L_ptr + off_q * stride_LS, mask=off_q < S, other=0.0)

        P_mat = tl.exp(S_mat - L_vals[:, None])

        dO_ptr = dO + b * stride_dOB + h * stride_dOH
        dO_tile = tl.load(
            dO_ptr + off_q[:, None] * stride_dOS + off_d[None, :] * stride_dOD,
            mask=((off_q[:, None] < S) & (off_d[None, :] < D)), other=0.0)

        acc = tl.dot(P_mat.T, dO_tile, acc)

    dV_ptr = dV + b * stride_dVB + h * stride_dVH
    tl.store(dV_ptr + off_k[:, None] * stride_dVS + off_d[None, :] * stride_dVD,
             acc.to(tl.bfloat16), mask=k_mask)


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
    B, H, S, D,
    inv_scale,
    BLOCK_SM: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """Compute dQ via decomposition and stash C[b,h,q] in dV[b,h,q,0].
    
    dQ[q,d] = (A - C*E) * inv_scale
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

    q_mask = (off_q[:, None] < S) & (off_d[None, :] < D)

    Q_ptr = Q + b * stride_QB + h * stride_QH
    Q_tile = tl.load(
        Q_ptr + off_q[:, None] * stride_QS + off_d[None, :] * stride_QD,
        mask=q_mask, other=0.0)

    L_ptr = L + b * stride_LB + h * stride_LH
    L_vals = tl.load(L_ptr + off_q * stride_LS, mask=off_q < S, other=0.0)

    dO_ptr = dO + b * stride_dOB + h * stride_dOH
    dO_tile = tl.load(
        dO_ptr + off_q[:, None] * stride_dOS + off_d[None, :] * stride_dOD,
        mask=q_mask, other=0.0)

    acc_A = tl.zeros((BLOCK_SM, BLOCK_D), dtype=tl.float32)
    acc_E = tl.zeros((BLOCK_SM, BLOCK_D), dtype=tl.float32)
    acc_C = tl.zeros((BLOCK_SM,), dtype=tl.float32)

    n_pk = tl.cdiv(S, BLOCK_SM)
    for pk in range(n_pk):
        off_k = pk * BLOCK_SM + tl.arange(0, BLOCK_SM)
        k_mask = (off_k[:, None] < S) & (off_d[None, :] < D)

        K_ptr = K + b * stride_KB + h * stride_KH
        K_tile = tl.load(
            K_ptr + off_k[:, None] * stride_KS + off_d[None, :] * stride_KD,
            mask=k_mask, other=0.0)

        S_mat = tl.dot(Q_tile, K_tile.T) * inv_scale
        P_mat = tl.exp(S_mat - L_vals[:, None])

        V_ptr = V + b * stride_VB + h * stride_VH
        V_tile = tl.load(
            V_ptr + off_k[:, None] * stride_VS + off_d[None, :] * stride_VD,
            mask=k_mask, other=0.0)

        dOV = tl.dot(dO_tile, V_tile.T)
        weight = P_mat * dOV

        acc_A = tl.dot(weight, K_tile, acc_A)
        acc_E = tl.dot(P_mat, K_tile, acc_E)
        acc_C = acc_C + tl.sum(weight, axis=1)

    dQ_val = (acc_A - acc_C[:, None] * acc_E) * inv_scale

    dQ_ptr = dQ + b * stride_dQB + h * stride_dQH
    tl.store(dQ_ptr + off_q[:, None] * stride_dQS + off_d[None, :] * stride_dQD,
             dQ_val.to(tl.bfloat16), mask=q_mask)

    # Stash C into dV[:,:,:0] for dK kernel to consume later
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
    B, H, S, D,
    inv_scale,
    BLOCK_SM: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """dK[k,d] = sum_q d_logits[q,k] * Q[q,d] * inv_scale
    
    Reads C[b,h,q] from dV[b,h,q,0] (written by dQ_C_kernel).
    d_logits[q,k] = P[q,k] * (dOV[q,k] - C[q])
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

    k_mask = (off_k[:, None] < S) & (off_d[None, :] < D)

    K_ptr = K + b * stride_KB + h * stride_KH
    K_tile = tl.load(
        K_ptr + off_k[:, None] * stride_KS + off_d[None, :] * stride_KD,
        mask=k_mask, other=0.0)

    V_ptr = V + b * stride_VB + h * stride_VH
    V_tile = tl.load(
        V_ptr + off_k[:, None] * stride_VS + off_d[None, :] * stride_VD,
        mask=k_mask, other=0.0)

    acc = tl.zeros((BLOCK_SM, BLOCK_D), dtype=tl.float32)

    n_pq = tl.cdiv(S, BLOCK_SM)
    for pq in range(n_pq):
        off_q = pq * BLOCK_SM + tl.arange(0, BLOCK_SM)
        q_mask = (off_q[:, None] < S) & (off_d[None, :] < D)

        Q_ptr = Q + b * stride_QB + h * stride_QH
        Q_tile = tl.load(
            Q_ptr + off_q[:, None] * stride_QS + off_d[None, :] * stride_QD,
            mask=q_mask, other=0.0)

        dO_ptr = dO + b * stride_dOB + h * stride_dOH
        dO_tile = tl.load(
            dO_ptr + off_q[:, None] * stride_dOS + off_d[None, :] * stride_dOD,
            mask=q_mask, other=0.0)

        L_ptr = L + b * stride_LB + h * stride_LH
        L_vals = tl.load(L_ptr + off_q * stride_LS, mask=off_q < S, other=0.0)

        # Load C from dV[:,:,:0]
        dV_base = dV + b * stride_dVB + h * stride_dVH
        C_vals = tl.load(dV_base + off_q * stride_dVS, mask=off_q < S, other=0.0)

        S_mat = tl.dot(Q_tile, K_tile.T) * inv_scale
        P_mat = tl.exp(S_mat - L_vals[:, None])

        dOV = tl.dot(dO_tile, V_tile.T)

        d_logits = P_mat * (dOV - C_vals[:, None])

        acc = tl.dot(d_logits.T, Q_tile, acc)

    acc *= inv_scale

    dK_ptr = dK + b * stride_dKB + h * stride_dKH
    tl.store(dK_ptr + off_k[:, None] * stride_dKS + off_d[None, :] * stride_dKD,
             acc.to(tl.bfloat16), mask=k_mask)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Multi-head attention backward: compute dQ, dK, dV.
    
    Execution order:
      1. _dQ_C_kernel  → writes dQ, and stashes C[b,h,q] into dV[b,h,q,0]
      2. _dK_kernel    → reads C from dV, writes dK
      3. _dV_kernel    → overwrites all of dV with the true dV gradients
    """
    torch.cuda.set_device(Q.device)

    B, H, S, D = Q.shape

    inv_scale = 1.0 / math.sqrt(D)

    BLOCK_SM = 32
    BLOCK_D = 128

    n_seq = triton.cdiv(S, BLOCK_SM)
    n_bh = B * H
    grid = (n_seq, n_bh)

    # Ensure L is 4-D for consistent stride indexing
    if L.dim() == 3:
        L = L.unsqueeze(-1)

    s_QB, s_QH, s_QS, s_QD = Q.stride()
    s_KB, s_KH, s_KS, s_KD = K.stride()
    s_VB, s_VH, s_VS, s_VD = V.stride()
    s_dOB, s_dOH, s_dOS, s_dOD = dO.stride()
    s_LB, s_LH, s_LS, _ = L.stride()
    s_dQB, s_dQH, s_dQS, s_dQD = dQ.stride()
    s_dKB, s_dKH, s_dKS, s_dKD = dK.stride()
    s_dVB, s_dVH, s_dVS, s_dVD = dV.stride()

    sc = (
        s_QB, s_QH, s_QS, s_QD,
        s_KB, s_KH, s_KS, s_KD,
        s_VB, s_VH, s_VS, s_VD,
        s_dOB, s_dOH, s_dOS, s_dOD,
        s_LB, s_LH, s_LS,
        s_dQB, s_dQH, s_dQS, s_dQD,
        s_dKB, s_dKH, s_dKS, s_dKD,
        s_dVB, s_dVH, s_dVS, s_dVD,
        B, H, S, D, inv_scale,
    )

    kwargs = dict(BLOCK_SM=BLOCK_SM, BLOCK_D=BLOCK_D, num_warps=4, num_stages=3)

    # Phase 1: dQ + stash C
    _dQ_C_kernel[grid](Q, K, V, dO, L, dQ, dK, dV, *sc, **kwargs)

    # Phase 2: dK (consumes C from dV)
    _dK_kernel[grid](Q, K, V, dO, L, dQ, dK, dV, *sc, **kwargs)

    # Phase 3: dV (overwrites the entire dV tensor, erasing C)
    _dV_kernel[grid](Q, K, V, dO, L, dQ, dK, dV, *sc, **kwargs)