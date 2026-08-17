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
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(S, BLOCK_M)

    bh = pid // num_pid_m
    b = bh // H
    h = bh % H
    pid_m = pid % num_pid_m
    offset_m = pid_m * BLOCK_M

    offs_m = offset_m + tl.arange(0, BLOCK_M)
    m_mask = offs_m < S

    q_base = Q_ptr + b * stride_qb + h * stride_qh
    k_base = K_ptr + b * stride_kb + h * stride_kh
    v_base = V_ptr + b * stride_vb + h * stride_vh
    do_base = dO_ptr + b * stride_dob + h * stride_doh
    o_base = O_ptr + b * stride_ob + h * stride_oh
    l_base = L_ptr + b * stride_lb + h * stride_lh
    dq_base = dQ_ptr + b * stride_dqb + h * stride_dqh

    # Precompute D = rowsum(dO * O) by iterating over d tiles
    D = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for off_d in range(0, d, BLOCK_D):
        offs_d = off_d + tl.arange(0, BLOCK_D)
        d_mask = offs_d < d

        dO_t = tl.load(
            do_base + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod,
            mask=m_mask[:, None] & d_mask[None, :], other=0.0,
        ).to(tl.float32)
        O_t = tl.load(
            o_base + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od,
            mask=m_mask[:, None] & d_mask[None, :], other=0.0,
        ).to(tl.float32)
        D += tl.sum(dO_t * O_t, axis=1)

    L_vals = tl.load(l_base + offs_m * stride_ls, mask=m_mask, other=0.0)

    actual_m = offset_m + tl.arange(0, BLOCK_M)[:, None]
    dQ_acc = tl.zeros((BLOCK_M, d), dtype=tl.float32)

    max_n = tl.cdiv(offset_m + BLOCK_M, BLOCK_N)

    for i_n in range(max_n):
        start_n = i_n * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask_flat = offs_n < S

        actual_n = start_n + tl.arange(0, BLOCK_N)[None, :]
        causal_mask = actual_n <= actual_m
        valid_mask = n_mask_flat[None, :] & causal_mask

        # Load full K, V row tiles into registers via d-iteration
        # Accumulate contribution to dQ across all d tiles
        dK_temp = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
        dV_temp = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

        scores = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        dP = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for off_d in range(0, d, BLOCK_D):
            offs_d = off_d + tl.arange(0, BLOCK_D)
            d_mask = offs_d < d

            Q_t = tl.load(
                q_base + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd,
                mask=m_mask[:, None] & d_mask[None, :], other=0.0,
            ).to(tl.float32)
            K_t = tl.load(
                k_base + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd,
                mask=n_mask_flat[:, None] & d_mask[None, :], other=0.0,
            ).to(tl.float32)
            V_t = tl.load(
                v_base + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd,
                mask=n_mask_flat[:, None] & d_mask[None, :], other=0.0,
            ).to(tl.float32)
            dO_t = tl.load(
                do_base + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod,
                mask=m_mask[:, None] & d_mask[None, :], other=0.0,
            ).to(tl.float32)

            scores += tl.dot(Q_t, K_t.T)
            dP += tl.dot(dO_t, V_t.T)

        scores *= SCALE
        P = tl.exp(scores - L_vals[:, None]) * valid_mask
        dS = P * (dP - D[:, None]) * SCALE

        # Now accumulate dQ = dS @ K (full d)
        for off_d in range(0, d, BLOCK_D):
            offs_d = off_d + tl.arange(0, BLOCK_D)
            d_mask = offs_d < d

            K_t = tl.load(
                k_base + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd,
                mask=n_mask_flat[:, None] & d_mask[None, :], other=0.0,
            ).to(tl.float32)

            slice_out = tl.reshape(
                tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32),
                (BLOCK_M, BLOCK_D),
            )
            dQ_slice = tl.dot(dS, K_t)
            # Scatter into accumulator
            dQ_acc_offsets = off_d + tl.arange(0, BLOCK_D)
            for lane_idx in range(BLOCK_D):
                dQ_acc = tl.where(
                    dQ_acc == dQ_acc,  # always true trick won't work elementwise
                    dQ_acc, dQ_acc
                )

    # Write dQ tile (need per-d-tile store since we have full d accumulator)
    pass  # complicated — let me use a simpler approach below


# Simpler approach: just use BLOCK_D = d since d=128 is manageable
@triton.jit
def _mha_bwd_dQ_kernel_simple(
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

    q_base = Q_ptr + b * stride_qb + h * stride_qh
    k_base = K_ptr + b * stride_kb + h * stride_kh
    v_base = V_ptr + b * stride_vb + h * stride_vh
    do_base = dO_ptr + b * stride_dob + h * stride_doh
    o_base = O_ptr + b * stride_ob + h * stride_oh
    l_base = L_ptr + b * stride_lb + h * stride_lh
    dq_base = dQ_ptr + b * stride_dqb + h * stride_dqh

    L_vals = tl.load(l_base + offs_m * stride_ls, mask=m_mask, other=0.0)

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

    Q_f = Q_tile.to(tl.float32)
    dO_f = dO_tile.to(tl.float32)
    O_f = O_tile.to(tl.float32)

    D = tl.sum(dO_f * O_f, axis=1)

    actual_m = offset_m + tl.arange(0, BLOCK_M)[:, None]
    dQ_acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    max_n = tl.cdiv(offset_m + BLOCK_M, BLOCK_N)

    for i_n in range(max_n):
        start_n = i_n * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask_flat = offs_n < S

        actual_n = start_n + tl.arange(0, BLOCK_N)[None, :]
        causal_mask = actual_n <= actual_m
        valid_mask = n_mask_flat[None, :] & causal_mask

        K_tile = tl.load(
            k_base + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd,
            mask=n_mask_flat[:, None] & d_mask[None, :], other=0.0,
        )
        V_tile = tl.load(
            v_base + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd,
            mask=n_mask_flat[:, None] & d_mask[None, :], other=0.0,
        )

        K_f = K_tile.to(tl.float32)
        V_f = V_tile.to(tl.float32)

        scores = tl.dot(Q_f, K_f.T) * SCALE
        P = tl.exp(scores - L_vals[:, None]) * valid_mask
        dP = tl.dot(dO_f, V_f.T)
        dS = P * (dP - D[:, None]) * SCALE
        dQ_acc = tl.dot(dS, K_f, dQ_acc)

    out_ptrs = dq_base + offs_m[:, None] * stride_dqs + offs_d[None, :] * stride_dqd
    tl.store(out_ptrs, dQ_acc.to(tl.bfloat16), mask=m_mask[:, None] & d_mask[None, :])


@triton.jit
def _mha_bwd_dKV_kernel_simple(
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

    q_base = Q_ptr + b * stride_qb + h * stride_qh
    k_base = K_ptr + b * stride_kb + h * stride_kh
    v_base = V_ptr + b * stride_vb + h * stride_vh
    do_base = dO_ptr + b * stride_dob + h * stride_doh
    o_base = O_ptr + b * stride_ob + h * stride_oh
    l_base = L_ptr + b * stride_lb + h * stride_lh
    dk_base = dK_ptr + b * stride_dkb + h * stride_dkh
    dv_base = dV_ptr + b * stride_dvb + h * stride_dvh

    K_tile = tl.load(
        k_base + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd,
        mask=n_mask[:, None] & d_mask[None, :], other=0.0,
    )
    K_f = K_tile.to(tl.float32)

    actual_n = offset_n + tl.arange(0, BLOCK_N)[None, :]
    dK_acc = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    dV_acc = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)

    first_m = offset_n // BLOCK_M
    num_m_iters = tl.cdiv(S, BLOCK_M)

    for i_m in range(first_m, num_m_iters):
        start_m = i_m * BLOCK_M
        offs_m = start_m + tl.arange(0, BLOCK_M)
        m_mask_flat = offs_m < S

        actual_m = start_m + tl.arange(0, BLOCK_M)[:, None]
        causal_mask = actual_n <= actual_m
        valid_mask = m_mask_flat[:, None] & n_mask[None, :] & causal_mask

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

        Q_f = Q_tile.to(tl.float32)
        dO_f = dO_tile.to(tl.float32)
        O_f = O_tile.to(tl.float32)
        V_f = V_tile.to(tl.float32)

        L_vals = tl.load(l_base + offs_m * stride_ls, mask=m_mask_flat, other=0.0)

        D = tl.sum(dO_f * O_f, axis=1)

        scores = tl.dot(Q_f, K_f.T) * SCALE
        P = tl.exp(scores - L_vals[:, None]) * valid_mask
        dP = tl.dot(dO_f, V_f.T)
        dS = P * (dP - D[:, None]) * SCALE

        dV_acc = tl.dot(P.T, dO_f, dV_acc)
        dK_acc = tl.dot(dS.T, Q_f, dK_acc)

    dk_ptrs = dk_base + offs_n[:, None] * stride_dks + offs_d[None, :] * stride_dkd
    tl.store(dk_ptrs, dK_acc.to(tl.bfloat16), mask=n_mask[:, None] & d_mask[None, :])

    dv_ptrs = dv_base + offs_n[:, None] * stride_dvs + offs_d[None, :] * stride_dvd
    tl.store(dv_ptrs, dV_acc.to(tl.bfloat16), mask=n_mask[:, None] & d_mask[None, :])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape

    if S <= 0:
        return

    SCALE = 1.0 / (d ** 0.5)

    # Use BLOCK_D = d so we process the full head dimension in one go
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = d  # = 128

    num_pid_m = triton.cdiv(S, BLOCK_M)
    num_pid_n = triton.cdiv(S, BLOCK_N)
    num_bh = B * H

    grid_dQ = (num_bh * num_pid_m,)
    _mha_bwd_dQ_kernel_simple[grid_dQ](
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

    grid_dKV = (num_bh * num_pid_n,)
    _mha_bwd_dKV_kernel_simple[grid_dKV](
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