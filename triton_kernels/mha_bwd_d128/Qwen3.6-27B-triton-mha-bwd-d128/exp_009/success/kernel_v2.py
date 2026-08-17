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

    q_base = Q + pid_head * stride_qh
    k_base = K + pid_head * stride_qh
    v_base = V + pid_head * stride_qh
    o_base = O + pid_head * stride_qh
    do_base = dO + pid_head * stride_qh
    l_base = L + pid_head * stride_lhs
    dk_base = dK_out + pid_head * stride_qh
    dv_base = dV_out + pid_head * stride_qh

    kv_mask = (offs_n[:, None] < S) & (offs_d[None, :] < D)

    # Load K and V once - keep bf16
    k_ptrs = k_base + offs_n[:, None] * stride_qs + offs_d[None, :] * stride_qd
    K_tile = tl.load(k_ptrs, mask=kv_mask, other=0.0)

    v_ptrs = v_base + offs_n[:, None] * stride_qs + offs_d[None, :] * stride_qd
    V_tile = tl.load(v_ptrs, mask=kv_mask, other=0.0)

    dk_acc = tl.zeros((BLOCK_N, BLOCK_DMODEL), dtype=tl.float32)
    dv_acc = tl.zeros((BLOCK_N, BLOCK_DMODEL), dtype=tl.float32)
    acc_s = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_dp = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_dk = tl.zeros((BLOCK_N, BLOCK_DMODEL), dtype=tl.float32)
    acc_dv = tl.zeros((BLOCK_N, BLOCK_DMODEL), dtype=tl.float32)

    num_m_blocks = tl.cdiv(S, BLOCK_M)

    for pid_m in range(num_m_blocks):
        cur_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        mq_mask = (cur_m[:, None] < S) & (offs_d[None, :] < D)

        Q_tile = tl.load(q_base + cur_m[:, None] * stride_qs + offs_d[None, :] * stride_qd,
                         mask=mq_mask, other=0.0)
        dO_tile = tl.load(do_base + cur_m[:, None] * stride_qs + offs_d[None, :] * stride_qd,
                          mask=mq_mask, other=0.0)
        O_tile = tl.load(o_base + cur_m[:, None] * stride_qs + offs_d[None, :] * stride_qd,
                         mask=mq_mask, other=0.0)

        lm_mask = cur_m < S
        L_tile = tl.load(l_base + cur_m, mask=lm_mask, other=0.0)[:, None]

        # D[i] = sum(dO*O, dim=-1) — computed in bf16 then widened
        D_tile = tl.sum(dO_tile * O_tile, axis=1)[:, None]

        # S = Q@K^T * scale using explicit fp32 accumulator
        S_mat = tl.dot(Q_tile, K_tile.T, acc_s) * scale
        # Reset for reuse next iteration
        acc_s = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        P_mat = tl.exp(S_mat - L_tile)

        # dP = dO @ V^T
        dP_mat = tl.dot(dO_tile, V_tile.T, acc_dp)
        acc_dp = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # dS = P * (dP - D) * scale
        dS_mat = P_mat * (dP_mat - D_tile) * scale

        # dK += dS^T @ Q  (RS-GEMM via fp32 accumulator)
        dk_acc = tl.dot(dS_mat.T, Q_tile.to(tl.float32), dk_acc)

        # dV += P^T @ dO  (RS-GEMM via fp32 accumulator)
        dv_acc = tl.dot(P_mat.T, dO_tile.to(tl.float32), dv_acc)

    dk_ptrs = dk_base + offs_n[:, None] * stride_qs + offs_d[None, :] * stride_qd
    tl.store(dk_ptrs, dk_acc.to(tl.bfloat16), mask=kv_mask)

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
    offs_d = tl.arange(0, BLOCK_DMODEL)

    q_base = Q + pid_head * stride_qh
    k_base = K + pid_head * stride_qh
    v_base = V + pid_head * stride_qh
    o_base = O + pid_head * stride_qh
    do_base = dO + pid_head * stride_qh
    l_base = L + pid_head * stride_lhs
    dq_base = dQ_out + pid_head * stride_qh

    mq_mask = (offs_m[:, None] < S) & (offs_d[None, :] < D)

    Q_tile = tl.load(q_base + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd,
                     mask=mq_mask, other=0.0)
    dO_tile = tl.load(do_base + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd,
                      mask=mq_mask, other=0.0)
    O_tile = tl.load(o_base + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd,
                     mask=mq_mask, other=0.0)

    lm_mask = offs_m < S
    L_tile = tl.load(l_base + offs_m, mask=lm_mask, other=0.0)[:, None]

    D_tile = tl.sum(dO_tile * O_tile, axis=1)[:, None]

    dq_acc = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)
    acc_s = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_dp = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    num_n_blocks = tl.cdiv(S, BLOCK_N)

    for pid_n in range(num_n_blocks):
        cur_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        kn_mask = (cur_n[:, None] < S) & (offs_d[None, :] < D)

        k_ptrs = k_base + cur_n[:, None] * stride_qs + offs_d[None, :] * stride_qd
        K_tile = tl.load(k_ptrs, mask=kn_mask, other=0.0)

        v_ptrs = v_base + cur_n[:, None] * stride_qs + offs_d[None, :] * stride_qd
        V_tile = tl.load(v_ptrs, mask=kn_mask, other=0.0)

        # S = Q @ K^T * scale
        S_mat = tl.dot(Q_tile, K_tile.T, acc_s) * scale
        acc_s = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        P_mat = tl.exp(S_mat - L_tile)

        # dP = dO @ V^T
        dP_mat = tl.dot(dO_tile, V_tile.T, acc_dp)
        acc_dp = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        dS_mat = P_mat * (dP_mat - D_tile) * scale

        # dQ += dS @ K
        dq_acc = tl.dot(dS_mat, K_tile.to(tl.float32), dq_acc)

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

    # Smaller blocks for better occupancy and reuse
    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_DMODEL = 128

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