import torch
import triton
import triton.language as tl


@triton.jit
def _mha_bwd_dQ_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, O_ptr, L_ptr, dQ_ptr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lb, stride_lh, stride_ls,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    B, H, S, d,
    SCALE,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute dQ for causal multi-head attention backward."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(S, BLOCK_M)

    bh = pid // num_pid_m
    b = bh // H
    h = bh % H
    pid_m = pid % num_pid_m
    offset_m = pid_m * BLOCK_M

    offs_m = offset_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    m_mask = offs_m < S
    d_mask = offs_d < d

    # Base pointers for this (batch, head)
    q_base = Q_ptr + b * stride_qb + h * stride_qh
    k_base = K_ptr + b * stride_kb + h * stride_kh
    v_base = V_ptr + b * stride_vb + h * stride_vh
    do_base = dO_ptr + b * stride_dob + h * stride_doh
    o_base = O_ptr + b * stride_ob + h * stride_oh
    l_base = L_ptr + b * stride_lb + h * stride_lh
    dq_base = dQ_ptr + b * stride_dqb + h * stride_dqh

    # Load L (logsumexp) for this row tile
    L_vals = tl.load(l_base + offs_m * stride_ls, mask=m_mask, other=0.0)

    # Load Q, dO, O tiles (reused across column-tile loop iterations)
    Q_tile = tl.load(
        q_base + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd,
        mask=m_mask[:, None] & d_mask[None, :], other=0.0,
    )
    dO_tile = tl.load(
        do_base + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod,
        mask=m_mask[:, None] & d_mask[None, :], other=0.0,
    )
    O_tile = tl.load(
        o_base + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od,
        mask=m_mask[:, None] & d_mask[None, :], other=0.0,
    )

    # D = rowsum(dO * O) — same for all column tiles of this row
    D = tl.sum(dO_tile * O_tile, axis=1)  # [BLOCK_M]

    # Absolute row indices for causal mask construction
    actual_m = offset_m + tl.arange(0, BLOCK_M)[:, None]  # [BLOCK_M, 1]

    # Accumulator for dQ in FP32
    dQ_acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    # Limit column iteration: rows [offset_m, offset_m+BLOCK_M) can attend to
    # columns [0, offset_m+BLOCK_M), so we stop at that column boundary
    max_n = tl.cdiv(offset_m + BLOCK_M, BLOCK_N)

    for i_n in range(max_n):
        start_n = i_n * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask_flat = offs_n < S

        # Causal mask: column n is valid for row m iff n <= m
        actual_n = start_n + tl.arange(0, BLOCK_N)[None, :]  # [1, BLOCK_N]
        causal_mask = actual_n <= actual_m                    # [BLOCK_M, BLOCK_N]
        valid_mask = n_mask_flat[None, :] & causal_mask       # [BLOCK_M, BLOCK_N]

        # Load K, V tiles for this column
        K_tile = tl.load(
            k_base + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd,
            mask=n_mask_flat[:, None] & d_mask[None, :], other=0.0,
        )
        V_tile = tl.load(
            v_base + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd,
            mask=n_mask_flat[:, None] & d_mask[None, :], other=0.0,
        )

        # Scaled scores: Q @ K^T * tau  (both bf16 -> fp32 result)
        scores = tl.dot(Q_tile, K_tile.T) * SCALE             # [BLOCK_M, BLOCK_N] fp32

        # Attention probabilities with causal masking
        P = tl.exp(scores - L_vals[:, None])                  # [BLOCK_M, BLOCK_N] fp32
        P = P * valid_mask                                    # zero out invalid positions

        # Gradient of unnormalized output: dO @ V^T
        dP = tl.dot(dO_tile, V_tile.T)                        # [BLOCK_M, BLOCK_N] fp32

        # Backprop through softmax and scaling
        # d_scores_scaled = P * (dP - D)
        # d_scores_unscaled = d_scores_scaled * SCALE (gradient w.r.t unscaled q·k)
        dS = P * (dP - D[:, None]) * SCALE                    # [BLOCK_M, BLOCK_N] fp32

        # Accumulate dQ = dS @ K
        # Cast dS to bf16 for tensor core compatibility
        dQ_acc = tl.dot(dS.to(tl.bfloat16), K_tile, dQ_acc)   # [BLOCK_M, BLOCK_D]

    # Write dQ tile
    out_ptrs = dq_base + offs_m[:, None] * stride_dqs + offs_d[None, :] * stride_dqd
    tl.store(out_ptrs, dQ_acc.to(tl.bfloat16), mask=m_mask[:, None] & d_mask[None, :])


@triton.jit
def _mha_bwd_dKV_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, O_ptr, L_ptr, dK_ptr, dV_ptr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lb, stride_lh, stride_ls,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    B, H, S, d,
    SCALE,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute dK and dV for causal multi-head attention backward."""
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(S, BLOCK_N)

    bh = pid // num_pid_n
    b = bh // H
    h = bh % H
    pid_n = pid % num_pid_n
    offset_n = pid_n * BLOCK_N

    offs_n = offset_n + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    n_mask = offs_n < S
    d_mask = offs_d < d

    # Base pointers for this (batch, head)
    q_base = Q_ptr + b * stride_qb + h * stride_qh
    k_base = K_ptr + b * stride_kb + h * stride_kh
    v_base = V_ptr + b * stride_vb + h * stride_vh
    do_base = dO_ptr + b * stride_dob + h * stride_doh
    o_base = O_ptr + b * stride_ob + h * stride_oh
    l_base = L_ptr + b * stride_lb + h * stride_lh
    dk_base = dK_ptr + b * stride_dkb + h * stride_dkh
    dv_base = dV_ptr + b * stride_dvb + h * stride_dvh

    # Load K tile (reused across row-tile loop iterations)
    K_tile = tl.load(
        k_base + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd,
        mask=n_mask[:, None] & d_mask[None, :], other=0.0,
    )

    # Absolute column indices for causal mask construction
    actual_n = offset_n + tl.arange(0, BLOCK_N)[None, :]  # [1, BLOCK_N]

    # Accumulators for dK, dV in FP32
    dK_acc = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    dV_acc = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)

    # Optimization: skip row tiles entirely before this column tile
    # Row m attends to col n iff m >= n, so we start from the first row tile
    # that could possibly overlap with column [offset_n, offset_n+BLOCK_N)
    first_m = offset_n // BLOCK_M
    num_m_iters = tl.cdiv(S, BLOCK_M)

    for i_m in range(first_m, num_m_iters):
        start_m = i_m * BLOCK_M
        offs_m = start_m + tl.arange(0, BLOCK_M)
        m_mask_flat = offs_m < S

        # Causal mask within this (row_tile, col_tile) pair
        actual_m = start_m + tl.arange(0, BLOCK_M)[:, None]  # [BLOCK_M, 1]
        causal_mask = actual_n <= actual_m                    # [BLOCK_M, BLOCK_N]
        valid_mask = m_mask_flat[:, None] & n_mask[None, :] & causal_mask

        # Load Q, dO, O, V tiles for this row
        Q_tile = tl.load(
            q_base + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd,
            mask=m_mask_flat[:, None] & d_mask[None, :], other=0.0,
        )
        dO_tile = tl.load(
            do_base + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod,
            mask=m_mask_flat[:, None] & d_mask[None, :], other=0.0,
        )
        O_tile = tl.load(
            o_base + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od,
            mask=m_mask_flat[:, None] & d_mask[None, :], other=0.0,
        )
        V_tile = tl.load(
            v_base + offs_m[:, None] * stride_vs + offs_d[None, :] * stride_vd,
            mask=m_mask_flat[:, None] & d_mask[None, :], other=0.0,
        )

        # Load L for this row tile
        L_vals = tl.load(l_base + offs_m * stride_ls, mask=m_mask_flat, other=0.0)

        # D = rowsum(dO * O)
        D = tl.sum(dO_tile * O_tile, axis=1)  # [BLOCK_M]

        # Scaled scores
        scores = tl.dot(Q_tile, K_tile.T) * SCALE  # [BLOCK_M, BLOCK_N] fp32

        # Attention probabilities with causal masking
        P = tl.exp(scores - L_vals[:, None])  # [BLOCK_M, BLOCK_N] fp32
        P = P * valid_mask

        # Gradient of unnormalized output
        dP = tl.dot(dO_tile, V_tile.T)  # [BLOCK_M, BLOCK_N] fp32

        # Backprop through softmax and scaling
        dS = P * (dP - D[:, None]) * SCALE  # [BLOCK_M, BLOCK_N] fp32

        # Accumulate dV: dV += P^T @ dO  [BLOCK_N, BLOCK_D]
        dV_acc = tl.dot(P.to(tl.bfloat16).T, dO_tile, dV_acc)

        # Accumulate dK: dK += dS^T @ Q  [BLOCK_N, BLOCK_D]
        dK_acc = tl.dot(dS.to(tl.bfloat16).T, Q_tile, dK_acc)

    # Write dK tile
    dk_ptrs = dk_base + offs_n[:, None] * stride_dks + offs_d[None, :] * stride_dkd
    tl.store(dk_ptrs, dK_acc.to(tl.bfloat16), mask=n_mask[:, None] & d_mask[None, :])

    # Write dV tile
    dv_ptrs = dv_base + offs_n[:, None] * stride_dvs + offs_d[None, :] * stride_dvd
    tl.store(dv_ptrs, dV_acc.to(tl.bfloat16), mask=n_mask[:, None] & d_mask[None, :])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Causal multi-head attention backward. Writes dQ, dK, dV into preallocated outputs."""
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape

    if S <= 0:
        return

    SCALE = 1.0 / (d ** 0.5)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = 128

    num_pid_m = triton.cdiv(S, BLOCK_M)
    num_pid_n = triton.cdiv(S, BLOCK_N)
    num_bh = B * H

    # --- Kernel 1: compute dQ ---
    grid_dQ = (num_bh * num_pid_m,)
    _mha_bwd_dQ_kernel[grid_dQ](
        Q, K, V, dO, O, L, dQ,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        B, H, S, d,
        SCALE,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=8, num_stages=3,
    )

    # --- Kernel 2: compute dK, dV ---
    grid_dKV = (num_bh * num_pid_n,)
    _mha_bwd_dKV_kernel[grid_dKV](
        Q, K, V, dO, O, L, dK, dV,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        B, H, S, d,
        SCALE,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=8, num_stages=3,
    )