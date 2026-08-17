import torch
import triton
import triton.language as tl
import math


@triton.jit
def _mha_bwd_dQ_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, dQ_ptr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lsb, stride_lsh, stride_ls,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    B, H, S, d, SCALE,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
):
    pid_m = tl.program_id(0)
    off_h = tl.program_id(1)
    off_b = tl.program_id(2)
    if off_b >= B or off_h >= H:
        return

    Q_base = Q_ptr + off_b * stride_qb + off_h * stride_qh
    K_base = K_ptr + off_b * stride_kb + off_h * stride_kh
    V_base = V_ptr + off_b * stride_vb + off_h * stride_vh
    dO_base = dO_ptr + off_b * stride_dob + off_h * stride_doh
    L_base = L_ptr + off_b * stride_lsb + off_h * stride_lsh
    dQ_base = dQ_ptr + off_b * stride_dqb + off_h * stride_dqh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    mask_m = offs_m < S

    Q_tile = tl.load(
        Q_base + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd,
        mask=mask_m[:, None], other=0.0,
    )
    dO_tile = tl.load(
        dO_base + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod,
        mask=mask_m[:, None], other=0.0,
    )
    L_tile = tl.load(L_base + offs_m * stride_ls, mask=mask_m, other=0.0)

    acc = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)

    end_n = min((pid_m + 1) * BLOCK_M, S)
    start_sn = 0
    end_sn = tl.cdiv(end_n, BLOCK_N)

    for sn in range(start_sn, end_sn):
        offs_n_base = sn * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n_base < S

        K_tile = tl.load(
            K_base + offs_n_base[:, None] * stride_ks + offs_d[None, :] * stride_kd,
            mask=mask_n[:, None], other=0.0,
        )
        V_tile = tl.load(
            V_base + offs_n_base[:, None] * stride_vs + offs_d[None, :] * stride_vd,
            mask=mask_n[:, None], other=0.0,
        )

        scores = tl.dot(Q_tile, tl.trans(K_tile)) * SCALE

        causal_mask = (offs_n_base[None, :] <= offs_m[:, None])
        valid = causal_mask & mask_m[:, None] & mask_n[None, :]

        attn = tl.where(valid, tl.exp(scores - L_tile[:, None]), 0.0)

        D_mat = tl.dot(dO_tile, tl.trans(V_tile))
        corr = tl.sum(attn * D_mat, axis=1, keep_dims=True)
        dp = attn * (D_mat - corr)

        acc += tl.dot(dp.to(tl.bfloat16), K_tile) * SCALE

    tl.store(
        dQ_base + offs_m[:, None] * stride_dqs + offs_d[None, :] * stride_dqd,
        acc.to(tl.bfloat16), mask=mask_m[:, None],
    )


@triton.jit
def _mha_bwd_dK_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, dK_ptr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lsb, stride_lsh, stride_ls,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    B, H, S, d, SCALE,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
):
    pid_n = tl.program_id(0)
    off_h = tl.program_id(1)
    off_b = tl.program_id(2)
    if off_b >= B or off_h >= H:
        return

    Q_base = Q_ptr + off_b * stride_qb + off_h * stride_qh
    K_base = K_ptr + off_b * stride_kb + off_h * stride_kh
    V_base = V_ptr + off_b * stride_vb + off_h * stride_vh
    dO_base = dO_ptr + off_b * stride_dob + off_h * stride_doh
    L_base = L_ptr + off_b * stride_lsb + off_h * stride_lsh
    dK_base = dK_ptr + off_b * stride_dkb + off_h * stride_dkh

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    mask_n = offs_n < S

    K_tile = tl.load(
        K_base + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd,
        mask=mask_n[:, None], other=0.0,
    )
    V_tile = tl.load(
        V_base + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd,
        mask=mask_n[:, None], other=0.0,
    )

    acc = tl.zeros((BLOCK_N, BLOCK_DMODEL), dtype=tl.float32)

    end_sm = tl.cdiv(S, BLOCK_M)

    for sm in range(end_sm):
        # For causal: only process query tiles where some m >= n exists
        end_q_tile = min((sm + 1) * BLOCK_M, S)
        if end_q_tile <= pid_n * BLOCK_N:
            continue

        offs_m = sm * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S

        Q_tile = tl.load(
            Q_base + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd,
            mask=mask_m[:, None], other=0.0,
        )
        dO_tile = tl.load(
            dO_base + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod,
            mask=mask_m[:, None], other=0.0,
        )
        L_tile = tl.load(L_base + offs_m * stride_ls, mask=mask_m, other=0.0)

        scores = tl.dot(Q_tile, tl.trans(K_tile)) * SCALE

        causal_mask = (offs_n[None, :] <= offs_m[:, None])
        valid = causal_mask & mask_m[:, None] & mask_n[None, :]

        attn = tl.where(valid, tl.exp(scores - L_tile[:, None]), 0.0)

        D_mat = tl.dot(dO_tile, tl.trans(V_tile))
        corr = tl.sum(attn * D_mat, axis=1, keep_dims=True)
        dp = attn * (D_mat - corr)

        acc += tl.dot(tl.trans(dp).to(tl.bfloat16), Q_tile) * SCALE

    tl.store(
        dK_base + offs_n[:, None] * stride_dks + offs_d[None, :] * stride_dkd,
        acc.to(tl.bfloat16), mask=mask_n[:, None],
    )


@triton.jit
def _mha_bwd_dV_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, dV_ptr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lsb, stride_lsh, stride_ls,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    B, H, S, d, SCALE,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
):
    pid_n = tl.program_id(0)
    off_h = tl.program_id(1)
    off_b = tl.program_id(2)
    if off_b >= B or off_h >= H:
        return

    Q_base = Q_ptr + off_b * stride_qb + off_h * stride_qh
    K_base = K_ptr + off_b * stride_kb + off_h * stride_kh
    dO_base = dO_ptr + off_b * stride_dob + off_h * stride_doh
    L_base = L_ptr + off_b * stride_lsb + off_h * stride_lsh
    dV_base = dV_ptr + off_b * stride_dvb + off_h * stride_dvh

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    mask_n = offs_n < S

    K_tile = tl.load(
        K_base + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd,
        mask=mask_n[:, None], other=0.0,
    )
    V_tile = tl.load(
        V_base + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd,
        mask=mask_n[:, None], other=0.0,
    )

    acc = tl.zeros((BLOCK_N, BLOCK_DMODEL), dtype=tl.float32)

    end_sm = tl.cdiv(S, BLOCK_M)

    for sm in range(end_sm):
        end_q_tile = min((sm + 1) * BLOCK_M, S)
        if end_q_tile <= pid_n * BLOCK_N:
            continue

        offs_m = sm * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S

        Q_tile = tl.load(
            Q_base + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd,
            mask=mask_m[:, None], other=0.0,
        )
        dO_tile = tl.load(
            dO_base + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod,
            mask=mask_m[:, None], other=0.0,
        )
        L_tile = tl.load(L_base + offs_m * stride_ls, mask=mask_m, other=0.0)

        scores = tl.dot(Q_tile, tl.trans(K_tile)) * SCALE

        causal_mask = (offs_n[None, :] <= offs_m[:, None])
        valid = causal_mask & mask_m[:, None] & mask_n[None, :]

        attn = tl.where(valid, tl.exp(scores - L_tile[:, None]), 0.0)

        acc += tl.dot(tl.trans(attn).to(tl.bfloat16), dO_tile)

    tl.store(
        dV_base + offs_n[:, None] * stride_dvs + offs_d[None, :] * stride_dvd,
        acc.to(tl.bfloat16), mask=mask_n[:, None],
    )


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Destination-passing entry point for causal MHA backward (bf16, d=128)."""
    torch.cuda.set_device(Q.device)
    B, H, S, d_dim = Q.shape

    scale = 1.0 / math.sqrt(d_dim)

    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_DMODEL = 128

    nqt = triton.cdiv(S, BLOCK_M)
    nkt = triton.cdiv(S, BLOCK_N)

    grid_common = (H, B)

    # Pass 1: compute dQ
    _mha_bwd_dQ_kernel[(nqt, *grid_common)](
        Q, K, V, dO, L, dQ,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        B, H, S, d_dim, scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=4, num_stages=2,
    )

    # Pass 2: compute dK
    _mha_bwd_dK_kernel[(nkt, *grid_common)](
        Q, K, V, dO, L, dK,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        B, H, S, d_dim, scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=4, num_stages=2,
    )

    # Pass 3: compute dV
    _mha_bwd_dV_kernel[(nkt, *grid_common)](
        Q, K, V, dO, L, dV,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        B, H, S, d_dim, scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=4, num_stages=2,
    )