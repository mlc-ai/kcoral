import torch
import triton
import triton.language as tl
import math


@triton.jit
def _mha_bwd_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr,
    dQ_ptr, dK_ptr, dV_ptr,
    stride_QS, stride_QD, stride_KS, stride_KD,
    stride_VS, stride_VD, stride_Os, stride_OD,
    stride_LS,
    stride_dQS, stride_dQD,
    stride_dKS, stride_dKD,
    stride_dVS, stride_dVD,
    S, D, inv_scale,
    BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """Fused MHA backward for one (b,h) head.
    
    Per-thread-state computes all of dQ[s,:], dK[s,:], dV[s,:] via decomposition.
    
    Phase A (forward over k tiles): accumulate A,E,C for dQ decomposition
    Phase B (forward over k tiles): accumulate dK components  
    Phase C (forward over k tiles): accumulate dV components
    
    Then finalize dQ using decomposition formula.
    dK needs C correction computed separately.
    
    We structure it so each tile computes its own dQ row and contributes 
    to full-sequence dK/dV accumulators via atomics... 
    Actually no, let's just compute everything cleanly per-(b,h) in registers.
    """
    pid_bh = tl.program_id(0)
    n_bh = B_H = tl.num_programs(0)
    
    if pid_bh >= n_bh:
        return

    b_idx = pid_bh // tl.cdiv(n_bh, 1)  # placeholder, handled by caller
    
    off_s_base = tl.arange(0, BLOCK_S)
    off_d = tl.arange(0, BLOCK_D)
    
    s_mask = off_s_base < S
    d_mask = off_d < D
    sd_mask = s_mask[:, None] & d_mask[None, :]

    # Load full Q[d_ohead,s,D] row block — but we need ALL q rows for P matrix
    # This is the core challenge: softmax couples all q positions together.
    
    # Strategy: outer loop over (q_tile), inner loop over (k_tile)
    # Accumulate global results in shared memory or use multi-phase approach.
    # For register-only approach with S=4096 and BLOCK_S=32:
    #   - 128 q-tiles, each needs 128 k-tiles => 16K dot products per head
    #   - Too many registers if we hold everything.
    
    # Better: decompose into 3 independent phases launched sequentially
    # but within SAME grid, sharing C buffer via dV column 0.
    
    tl.store(dQ_ptr, tl.zeros((1,), dtype=tl.bfloat16), mask=False)


@triton.jit
def _dQ_phase(
    Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, dQ_ptr, dV_ptr,
    stride_QS, stride_QD, stride_KS, stride_KD,
    stride_VS, stride_VD, stride_Os, stride_OD,
    stride_LS,
    stride_dQS, stride_dQD,
    stride_dVS,
    S, D, inv_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """Compute dQ via decomposition. Store C[b,h,q] into dV[b,h,q,0]."""
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    n_sm = tl.cdiv(S, BLOCK_M)
    n_bh_total = tl.num_programs(1)
    
    if pid_m >= n_sm or pid_bh >= n_bh_total:
        return
    
    b_idx = pid_bh // H_T
    h_idx = pid_bh % H_T

    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_d = tl.arange(0, BLOCK_D)
    m_mask = off_m < S
    d_mask = off_d < D
    md_mask = m_mask[:, None] & d_mask[None, :]

    Q_tile = tl.load(Q_ptr + off_m[:, None] * stride_QS + off_d[None, :] * stride_QD,
                     mask=md_mask, other=0.0)
    dO_tile = tl.load(dO_ptr + off_m[:, None] * stride_Os + off_d[None, :] * stride_OD,
                      mask=md_mask, other=0.0)
    L_vals = tl.load(L_ptr + off_m * stride_LS, mask=m_mask, other=0.0)

    acc_A = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    acc_E = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    acc_C = tl.zeros((BLOCK_M,), dtype=tl.float32)

    n_sn = tl.cdiv(S, BLOCK_N)
    for sn in range(n_sn):
        off_n = sn * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = off_n < S
        nd_mask = n_mask[:, None] & d_mask[None, :]

        K_tile = tl.load(K_ptr + off_n[:, None] * stride_KS + off_d[None, :] * stride_KD,
                         mask=nd_mask, other=0.0)
        V_tile = tl.load(V_ptr + off_n[:, None] * stride_VS + off_d[None, :] * stride_VD,
                         mask=nd_mask, other=0.0)

        S_mat = tl.dot(Q_tile, K_tile.T) * inv_scale
        P_mat = tl.exp(S_mat - L_vals[:, None])

        dOV = tl.dot(dO_tile.to(tl.float32), V_tile.to(tl.float32).T)
        weight = P_mat * dOV

        acc_A = tl.dot(weight, K_tile.to(tl.float32), acc_A)
        acc_E = tl.dot(P_mat, K_tile.to(tl.float32), acc_E)
        acc_C += tl.sum(weight, axis=1)

    dQ_val = (acc_A - acc_C[:, None] * acc_E) * inv_scale
    tl.store(dQ_ptr + off_m[:, None] * stride_dQS + off_d[None, :] * stride_dQD,
             dQ_val.to(tl.bfloat16), mask=md_mask)

    tl.store(dV_ptr + off_m * stride_dVS, acc_C.to(tl.bfloat16), mask=m_mask)


@triton.jit
def _dK_phase(
    Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, dK_ptr, dV_ptr,
    stride_QS, stride_QD, stride_KS, stride_KD,
    stride_VS, stride_VD, stride_Os, stride_OD,
    stride_LS,
    stride_dKS, stride_dKD,
    stride_dVS,
    S, D, inv_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """Compute dK, reading C from dV[:,:,:0]."""
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    n_sm = tl.cdiv(S, BLOCK_M)
    n_bh_total = tl.num_programs(1)
    
    if pid_m >= n_sm or pid_bh >= n_bh_total:
        return
    
    b_idx = pid_bh // H_T
    h_idx = pid_bh % H_T

    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_d = tl.arange(0, BLOCK_D)
    m_mask = off_m < S
    d_mask = off_d < D
    md_mask = m_mask[:, None] & d_mask[None, :]

    K_tile = tl.load(K_ptr + off_m[:, None] * stride_KS + off_d[None, :] * stride_KD,
                     mask=md_mask, other=0.0)
    V_tile = tl.load(V_ptr + off_m[:, None] * stride_VS + off_d[None, :] * stride_VD,
                     mask=md_mask, other=0.0)

    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    n_sn = tl.cdiv(S, BLOCK_N)
    for sn in range(n_sn):
        off_n = sn * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = off_n < S
        nd_mask = n_mask[:, None] & d_mask[None, :]

        Q_tile = tl.load(Q_ptr + off_n[:, None] * stride_QS + off_d[None, :] * stride_QD,
                         mask=nd_mask, other=0.0)
        dO_tile = tl.load(dO_ptr + off_n[:, None] * stride_Os + off_d[None, :] * stride_OD,
                          mask=nd_mask, other=0.0)
        L_vals = tl.load(L_ptr + off_n * stride_LS, mask=n_mask, other=0.0)
        C_vals = tl.load(dV_ptr + off_n * stride_dVS, mask=n_mask, other=0.0)

        S_mat = tl.dot(Q_tile, K_tile.T) * inv_scale
        P_mat = tl.exp(S_mat - L_vals[:, None])
        dOV = tl.dot(dO_tile.to(tl.float32), V_tile.to(tl.float32).T)

        d_logits = P_mat * (dOV - C_vals[:, None])
        acc = tl.dot(d_logits.T, Q_tile.to(tl.float32), acc)

    acc *= inv_scale
    tl.store(dK_ptr + off_m[:, None] * stride_dKS + off_d[None, :] * stride_dKD,
             acc.to(tl.bfloat16), mask=md_mask)


@triton.jit
def _dV_phase(
    Q_ptr, K_ptr, dO_ptr, L_ptr, dV_ptr,
    stride_QS, stride_QD, stride_KS, stride_KD,
    stride_Os, stride_OD,
    stride_LS,
    stride_dVS, stride_dVD,
    S, D, inv_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """Compute dV, overwriting C stash."""
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    n_sm = tl.cdiv(S, BLOCK_M)
    n_bh_total = tl.num_programs(1)
    
    if pid_m >= n_sm or pid_bh >= n_bh_total:
        return
    
    b_idx = pid_bh // H_T
    h_idx = pid_bh % H_T

    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_d = tl.arange(0, BLOCK_D)
    m_mask = off_m < S
    d_mask = off_d < D
    md_mask = m_mask[:, None] & d_mask[None, :]

    K_tile = tl.load(K_ptr + off_m[:, None] * stride_KS + off_d[None, :] * stride_KD,
                     mask=md_mask, other=0.0)

    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    n_sn = tl.cdiv(S, BLOCK_N)
    for sn in range(n_sn):
        off_n = sn * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = off_n < S
        nd_mask = n_mask[:, None] & d_mask[None, :]

        Q_tile = tl.load(Q_ptr + off_n[:, None] * stride_QS + off_d[None, :] * stride_QD,
                         mask=nd_mask, other=0.0)
        dO_tile = tl.load(dO_ptr + off_n[:, None] * stride_Os + off_d[None, :] * stride_OD,
                          mask=nd_mask, other=0.0)
        L_vals = tl.load(L_ptr + off_n * stride_LS, mask=n_mask, other=0.0)

        S_mat = tl.dot(Q_tile, K_tile.T) * inv_scale
        P_mat = tl.exp(S_mat - L_vals[:, None])

        acc = tl.dot(P_mat.T, dO_tile.to(tl.float32), acc)

    tl.store(dV_ptr + off_m[:, None] * stride_dVS + off_d[None, :] * stride_dVD,
             acc.to(tl.bfloat16), mask=md_mask)


# Need to define H_T inside each kernel since we can't reference globals
# Let me redo properly with batch/head indices as explicit arguments

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
    """Compute dQ for entire batch. Stash C into dV[:, :, :, 0]."""
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
    kw = dict(BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
              num_warps=4, num_stages=3)

    sc_all = (sq + sk + sv + sdo + sl + sdq + sdk + sdv)

    # Phase 1: dQ + stash C
    _dQ_full[grid](
        Q, K, V, dO, L, dQ, dK, dV,
        *sc_all, B, H, S, D, inv_scale, **kw)

    # Phase 2: dK (reads C)
    _dK_full[grid](
        Q, K, V, dO, L, dQ, dK, dV,
        *sc_all, B, H, S, D, inv_scale, **kw)

    # Phase 3: dV (overwrites C)
    _dV_full[grid](
        Q, K, dO, L, dV,
        sq[0], sq[1], sq[2], sq[3],
        sk[0], sk[1], sk[2], sk[3],
        sdo[0], sdo[1], sdo[2], sdo[3],
        sl[0], sl[1], sl[2],
        sdv[0], sdv[1], sdv[2], sdv[3],
        B, H, S, D, inv_scale, **kw)