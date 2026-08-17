import math

import torch
import triton
import triton.language as tl


@triton.jit
def _sdpa_bwd_dkv_kernel(
    Q, K, V, O, dO, L,
    dK_out, dV_out,
    stride_qh, stride_qs, stride_qd,
    stride_lhs,
    S, D,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    """Compute dK and dV. Each program owns one K-sequence tile."""
    pid_head = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # Base pointers for this head
    q_base = Q + pid_head * stride_qh
    k_base = K + pid_head * stride_qh
    v_base = V + pid_head * stride_qh
    o_base = O + pid_head * stride_qh
    do_base = dO + pid_head * stride_qh
    l_base = L + pid_head * stride_lhs
    dk_base = dK_out + pid_head * stride_qh
    dv_base = dV_out + pid_head * stride_qh

    # Load K and V tiles once (fixed for this program)
    k_ptrs = k_base + offs_n[:, None] * stride_qs + offs_d[None, :] * stride_qd
    kv_mask = (offs_n[:, None] < S) & (offs_d[None, :] < D)
    K_tile = tl.load(k_ptrs, mask=kv_mask, other=0.0)

    v_ptrs = v_base + offs_n[:, None] * stride_qs + offs_d[None, :] * stride_qd
    V_tile = tl.load(v_ptrs, mask=kv_mask, other=0.0)

    dk_acc = tl.zeros((BLOCK_N, BLOCK_DMODEL), dtype=tl.float32)
    dv_acc = tl.zeros((BLOCK_N, BLOCK_DMODEL), dtype=tl.float32)

    num_m_blocks = tl.cdiv(S, BLOCK_M)

    for pid_m in range(num_m_blocks):
        cur_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

        mq_mask = (cur_m[:, None] < S) & (offs_d[None, :] < D)

        # Load Q tile
        q_ptrs = q_base + cur_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
        Q_tile = tl.load(q_ptrs, mask=mq_mask, other=0.0)

        # Load dO tile
        do_ptrs = do_base + cur_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
        dO_tile = tl.load(do_ptrs, mask=mq_mask, other=0.0)

        # Load O tile (needed for D computation)
        o_ptrs = o_base + cur_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
        O_tile = tl.load(o_ptrs, mask=mq_mask, other=0.0)

        # Load logsumexp for this Q tile
        lm_mask = cur_m < S
        l_ptrs = l_base + cur_m
        L_tile = tl.load(l_ptrs, mask=lm_mask, other=0.0)
        L_tile = L_tile[:, None]

        # D[i] = sum_k (dO[i,k] * O[i,k])
        D_tile = tl.sum(dO_tile * O_tile, axis=1)[:, None]

        # S = Q @ K^T * scale
        S_mat = tl.dot(Q_tile, K_tile.T) * scale

        # P = exp(S - L)
        P_mat = tl.exp(S_mat - L_tile)

        # dP = dO @ V^T
        dP_mat = tl.dot(dO_tile, V_tile.T)

        # dS = P * (dP - D) * scale
        dS_mat = P_mat * (dP_mat - D_tile) * scale

        # Accumulate dK += dS^T @ Q
        dk_acc = tl.dot(dS_mat.T, Q_tile, dk_acc)

        # Accumulate dV += P^T @ dO
        dv_acc = tl.dot(P_mat.T, dO_tile, dv_acc)

    # Store dK
    dk_ptrs = dk_base + offs_n[:, None] * stride_qs + offs_d[None, :] * stride_qd
    tl.store(dk_ptrs, dk_acc.to(tl.bfloat16), mask=kv_mask)

    # Store dV
    dv_ptrs = dv_base + offs_n[:, None] * stride_qs + offs_d[None, :] * stride_qd
    tl.store(dv_ptrs, dv_acc.to(tl.bfloat16), mask=kv_mask)


@triton.jit
def _sdpa_bwd_dq_kernel(
    Q, K, V, O, dO, L,
    dQ_out,
    stride_qh, stride_qs, stride_qd,
    stride_lhs,
    S, D,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    """Compute dQ. Each program owns one Q-sequence tile."""
    pid_head = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    q_base = Q + pid_head * stride_qh
    k_base = K + pid_head * stride_qh
    v_base = V + pid_head * stride_qh
    o_base = O + pid_head * stride_qh
    do_base = dO + pid_head * stride_qh
    l_base = L + pid_head * stride_lhs
    dq_base = dQ_out + pid_head * stride_qh

    mq_mask = (offs_m[:, None] < S) & (offs_d[None, :] < D)

    # Load Q tile once
    q_ptrs = q_base + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    Q_tile = tl.load(q_ptrs, mask=mq_mask, other=0.0)

    # Load dO tile once
    do_ptrs = do_base + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    dO_tile = tl.load(do_ptrs, mask=mq_mask, other=0.0)

    # Load O tile once (needed for D)
    o_ptrs = o_base + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    O_tile = tl.load(o_ptrs, mask=mq_mask, other=0.0)

    # Load logsumexp
    lm_mask = offs_m < S
    l_ptrs = l_base + offs_m
    L_tile = tl.load(l_ptrs, mask=lm_mask, other=0.0)
    L_tile = L_tile[:, None]

    # D[i] = sum_k (dO[i,k] * O[i,k])
    D_tile = tl.sum(dO_tile * O_tile, axis=1)[:, None]

    dq_acc = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)

    num_n_blocks = tl.cdiv(S, BLOCK_N)

    for pid_n in range(num_n_blocks):
        cur_n = pid_n * BLOCK_N + offs_n

        kn_mask = (cur_n[:, None] < S) & (offs_d[None, :] < D)

        # Load K tile
        k_ptrs = k_base + cur_n[:, None] * stride_qs + offs_d[None, :] * stride_qd
        K_tile = tl.load(k_ptrs, mask=kn_mask, other=0.0)

        # Load V tile
        v_ptrs = v_base + cur_n[:, None] * stride_qs + offs_d[None, :] * stride_qd
        V_tile = tl.load(v_ptrs, mask=kn_mask, other=0.0)

        # S = Q @ K^T * scale
        S_mat = tl.dot(Q_tile, K_tile.T) * scale

        # P = exp(S - L)
        P_mat = tl.exp(S_mat - L_tile)

        # dP = dO @ V^T
        dP_mat = tl.dot(dO_tile, V_tile.T)

        # dS = P * (dP - D) * scale
        dS_mat = P_mat * (dP_mat - D_tile) * scale

        # Accumulate dQ += dS @ K
        dq_acc = tl.dot(dS_mat, K_tile, dq_acc)

    # Store dQ
    dq_ptrs = dq_base + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    tl.store(dq_ptrs, dq_acc.to(tl.bfloat16), mask=mq_mask)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Multi-head attention backward: compute dQ, dK, dV from forward cache."""
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    BH = B * H

    scale = 1.0 / math.sqrt(float(D))

    stride_qh = Q.stride(1)
    stride_qs = Q.stride(2)
    stride_qd = Q.stride(3)
    stride_lhs = L.stride(1)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_DMODEL = 128

    # Kernel 1: compute dK and dV
    grid_dkv = (BH, triton.cdiv(S, BLOCK_N))
    _sdpa_bwd_dkv_kernel[grid_dkv](
        Q, K, V, O, dO, L,
        dK, dV,
        stride_qh, stride_qs, stride_qd,
        stride_lhs,
        S, D,
        scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=4,
        num_stages=2,
    )

    # Kernel 2: compute dQ
    grid_dq = (BH, triton.cdiv(S, BLOCK_M))
    _sdpa_bwd_dq_kernel[grid_dq](
        Q, K, V, O, dO, L,
        dQ,
        stride_qh, stride_qs, stride_qd,
        stride_lhs,
        S, D,
        scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=4,
        num_stages=2,
    )