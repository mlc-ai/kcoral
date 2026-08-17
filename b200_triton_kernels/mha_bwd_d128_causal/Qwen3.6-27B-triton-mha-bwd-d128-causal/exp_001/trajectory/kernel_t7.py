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
    pid_bh = tl.program_id(1)
    off_b = pid_bh // H
    off_h = pid_bh % H
    if off_b >= B or off_h >= H:
        return

    qb = off_b * stride_qb + off_h * stride_qh
    kb = off_b * stride_kb + off_h * stride_kh
    vb = off_b * stride_vb + off_h * stride_vh
    dob = off_b * stride_dob + off_h * stride_doh
    lsb_off = off_b * stride_lsb + off_h * stride_lsh
    dqb = off_b * stride_dqb + off_h * stride_dqh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    mask_m = offs_m < S

    Q_base = Q_ptr + qb + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    dO_base = dO_ptr + dob + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod
    
    Q_tile = tl.load(Q_base, mask=mask_m[:, None], other=0.0)
    dO_tile = tl.load(dO_base, mask=mask_m[:, None], other=0.0)
    L_tile = tl.load(L_ptr + lsb_off + offs_m * stride_ls, mask=mask_m, other=0.0)

    acc = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)

    end_n = min((pid_m + 1) * BLOCK_M, S)
    num_sn = tl.cdiv(end_n, BLOCK_N)

    for sn in range(num_sn):
        offs_n = sn * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S

        K_tile = tl.load(K_ptr + kb + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd,
                         mask=mask_n[:, None], other=0.0)
        V_tile = tl.load(V_ptr + vb + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd,
                         mask=mask_n[:, None], other=0.0)

        scores = tl.dot(Q_tile, tl.trans(K_tile)) * SCALE

        causal_mask = (offs_n[None, :] <= offs_m[:, None])
        valid = causal_mask & mask_m[:, None] & mask_n[None, :]

        attn = tl.where(valid, tl.exp(scores - L_tile[:, None]), 0.0)

        D_mat = tl.dot(dO_tile, tl.trans(V_tile))
        corr = tl.sum(attn * D_mat, axis=1, keep_dims=True)
        dp = attn * (D_mat - corr)

        acc += tl.dot(dp.to(tl.bfloat16), K_tile) * SCALE

    tl.store(dQ_ptr + dqb + offs_m[:, None] * stride_dqs + offs_d[None, :] * stride_dqd,
             acc.to(tl.bfloat16), mask=mask_m[:, None])


@triton.jit
def _mha_bwd_dKV_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, dK_ptr, dV_ptr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lsb, stride_lsh, stride_ls,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    B, H, S, d, SCALE,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    off_b = pid_bh // H
    off_h = pid_bh % H
    if off_b >= B or off_h >= H:
        return

    qb = off_b * stride_qb + off_h * stride_qh
    kb = off_b * stride_kb + off_h * stride_kh
    vb = off_b * stride_vb + off_h * stride_vh
    dob = off_b * stride_dob + off_h * stride_doh
    lsb_off = off_b * stride_lsb + off_h * stride_lsh
    dkb = off_b * stride_dkb + off_h * stride_dkh
    dvb = off_b * stride_dvb + off_h * stride_dvh

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    mask_n = offs_n < S

    K_tile = tl.load(K_ptr + kb + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd,
                     mask=mask_n[:, None], other=0.0)
    V_tile = tl.load(V_ptr + vb + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd,
                     mask=mask_n[:, None], other=0.0)

    acc_K = tl.zeros((BLOCK_N, BLOCK_DMODEL), dtype=tl.float32)
    acc_V = tl.zeros((BLOCK_N, BLOCK_DMODEL), dtype=tl.float32)

    num_sm = tl.cdiv(S, BLOCK_M)

    for sm in range(num_sm):
        end_q = min((sm + 1) * BLOCK_M, S)
        if end_q <= pid_n * BLOCK_N:
            continue

        offs_m = sm * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S

        Q_tile = tl.load(Q_ptr + qb + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd,
                         mask=mask_m[:, None], other=0.0)
        dO_tile = tl.load(dO_ptr + dob + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod,
                          mask=mask_m[:, None], other=0.0)
        L_tile = tl.load(L_ptr + lsb_off + offs_m * stride_ls, mask=mask_m, other=0.0)

        scores = tl.dot(Q_tile, tl.trans(K_tile)) * SCALE

        causal_mask = (offs_n[None, :] <= offs_m[:, None])
        valid = causal_mask & mask_m[:, None] & mask_n[None, :]

        attn = tl.where(valid, tl.exp(scores - L_tile[:, None]), 0.0)

        D_mat = tl.dot(dO_tile, tl.trans(V_tile))
        corr = tl.sum(attn * D_mat, axis=1, keep_dims=True)
        dp = attn * (D_mat - corr)

        acc_K += tl.dot(tl.trans(dp).to(tl.bfloat16), Q_tile) * SCALE
        acc_V += tl.dot(tl.trans(attn).to(tl.bfloat16), dO_tile)

    tl.store(dK_ptr + dkb + offs_n[:, None] * stride_dks + offs_d[None, :] * stride_dkd,
             acc_K.to(tl.bfloat16), mask=mask_n[:, None])
    tl.store(dV_ptr + dvb + offs_n[:, None] * stride_dvs + offs_d[None, :] * stride_dvd,
             acc_V.to(tl.bfloat16), mask=mask_n[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Destination-passing entry point for causal MHA backward (bf16, d=128)."""
    torch.cuda.set_device(Q.device)
    B, H, S, d_dim = Q.shape

    # Zero outputs
    dQ.zero_()
    dK.zero_()
    dV.zero_()

    scale = 1.0 / math.sqrt(d_dim)

    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_DMODEL = 64

    nqt = triton.cdiv(S, BLOCK_M)
    nkt = triton.cdiv(S, BLOCK_N)
    total_bh = B * H

    grid_common = (total_bh,)

    # Process each half of the d-dimension separately
    for dc in range(2):
        d_off = dc * BLOCK_DMODEL

        @triton.jit
        def _dQ_chunk(
            Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, dQ_ptr,
            stride_qb, stride_qh, stride_qs, stride_qd,
            stride_kb, stride_kh, stride_ks, stride_kd,
            stride_vb, stride_vh, stride_vs, stride_vd,
            stride_dob, stride_doh, stride_dos, stride_dod,
            stride_lsb, stride_lsh, stride_ls,
            stride_dqb, stride_dqh, stride_dqs, stride_dqd,
            B, H, S, d, SCALE, DOFF,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
        ):
            pid_m = tl.program_id(0)
            pid_bh = tl.program_id(1)
            off_b = pid_bh // H
            off_h = pid_bh % H
            if off_b >= B or off_h >= H:
                return

            qb = off_b * stride_qb + off_h * stride_qh
            kb = off_b * stride_kb + off_h * stride_kh
            vb = off_b * stride_vb + off_h * stride_vh
            dob = off_b * stride_dob + off_h * stride_doh
            lsb_off = off_b * stride_lsb + off_h * stride_lsh
            dqb = off_b * stride_dqb + off_h * stride_dqh

            offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_d = DOFF + tl.arange(0, BLOCK_DMODEL)
            mask_m = offs_m < S
            mask_d = offs_d < d

            Q_tile = tl.load(Q_ptr + qb + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd,
                             mask=mask_m[:, None] & mask_d[None, :], other=0.0)
            dO_tile = tl.load(dO_ptr + dob + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod,
                              mask=mask_m[:, None] & mask_d[None, :], other=0.0)
            L_tile = tl.load(L_ptr + lsb_off + offs_m * stride_ls, mask=mask_m, other=0.0)

            acc = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)

            end_n = min((pid_m + 1) * BLOCK_M, S)
            num_sn = tl.cdiv(end_n, BLOCK_N)

            for sn in range(num_sn):
                offs_n = sn * BLOCK_N + tl.arange(0, BLOCK_N)
                mask_n = offs_n < S

                K_tile = tl.load(K_ptr + kb + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd,
                                 mask=mask_n[:, None] & mask_d[None, :], other=0.0)
                V_tile = tl.load(V_ptr + vb + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd,
                                 mask=mask_n[:, None] & mask_d[None, :], other=0.0)

                scores = tl.dot(Q_tile, tl.trans(K_tile)) * SCALE

                causal_mask = (offs_n[None, :] <= offs_m[:, None])
                valid = causal_mask & mask_m[:, None] & mask_n[None, :]

                attn = tl.where(valid, tl.exp(scores - L_tile[:, None]), 0.0)

                D_mat = tl.dot(dO_tile, tl.trans(V_tile))
                corr = tl.sum(attn * D_mat, axis=1, keep_dims=True)
                dp = attn * (D_mat - corr)

                acc += tl.dot(dp.to(tl.bfloat16), K_tile) * SCALE

            tl.store(dQ_ptr + dqb + offs_m[:, None] * stride_dqs + offs_d[None, :] * stride_dqd,
                     acc.to(tl.bfloat16), mask=mask_m[:, None] & mask_d[None, :])

        _dQ_chunk[(nqt, total_bh)](
            Q, K, V, dO, L, dQ,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
            L.stride(0), L.stride(1), L.stride(2),
            dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
            B, H, S, d_dim, scale, d_off,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=BLOCK_DMODEL,
            num_warps=2, num_stages=1,
        )

        @triton.jit
        def _dKV_chunk(
            Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, dK_ptr, dV_ptr,
            stride_qb, stride_qh, stride_qs, stride_qd,
            stride_kb, stride_kh, stride_ks, stride_kd,
            stride_vb, stride_vh, stride_vs, stride_vd,
            stride_dob, stride_doh, stride_dos, stride_dod,
            stride_lsb, stride_lsh, stride_ls,
            stride_dkb, stride_dkh, stride_dks, stride_dkd,
            stride_dvb, stride_dvh, stride_dvs, stride_dvd,
            B, H, S, d, SCALE, DOFF,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
        ):
            pid_n = tl.program_id(0)
            pid_bh = tl.program_id(1)
            off_b = pid_bh // H
            off_h = pid_bh % H
            if off_b >= B or off_h >= H:
                return

            qb = off_b * stride_qb + off_h * stride_qh
            kb = off_b * stride_kb + off_h * stride_kh
            vb = off_b * stride_vb + off_h * stride_vh
            dob = off_b * stride_dob + off_h * stride_doh
            lsb_off = off_b * stride_lsb + off_h * stride_lsh
            dkb = off_b * stride_dkb + off_h * stride_dkh
            dvb = off_b * stride_dvb + off_h * stride_dvh

            offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            offs_d = DOFF + tl.arange(0, BLOCK_DMODEL)
            mask_n = offs_n < S
            mask_d = offs_d < d

            K_tile = tl.load(K_ptr + kb + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd,
                             mask=mask_n[:, None] & mask_d[None, :], other=0.0)
            V_tile = tl.load(V_ptr + vb + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd,
                             mask=mask_n[:, None] & mask_d[None, :], other=0.0)

            acc_K = tl.zeros((BLOCK_N, BLOCK_DMODEL), dtype=tl.float32)
            acc_V = tl.zeros((BLOCK_N, BLOCK_DMODEL), dtype=tl.float32)

            num_sm = tl.cdiv(S, BLOCK_M)

            for sm in range(num_sm):
                end_q = min((sm + 1) * BLOCK_M, S)
                if end_q <= pid_n * BLOCK_N:
                    continue

                offs_m = sm * BLOCK_M + tl.arange(0, BLOCK_M)
                mask_m = offs_m < S

                Q_tile = tl.load(Q_ptr + qb + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd,
                                 mask=mask_m[:, None] & mask_d[None, :], other=0.0)
                dO_tile = tl.load(dO_ptr + dob + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod,
                                  mask=mask_m[:, None] & mask_d[None, :], other=0.0)
                L_tile = tl.load(L_ptr + lsb_off + offs_m * stride_ls, mask=mask_m, other=0.0)

                scores = tl.dot(Q_tile, tl.trans(K_tile)) * SCALE

                causal_mask = (offs_n[None, :] <= offs_m[:, None])
                valid = causal_mask & mask_m[:, None] & mask_n[None, :]

                attn = tl.where(valid, tl.exp(scores - L_tile[:, None]), 0.0)

                D_mat = tl.dot(dO_tile, tl.trans(V_tile))
                corr = tl.sum(attn * D_mat, axis=1, keep_dims=True)
                dp = attn * (D_mat - corr)

                acc_K += tl.dot(tl.trans(dp).to(tl.bfloat16), Q_tile) * SCALE
                acc_V += tl.dot(tl.trans(attn).to(tl.bfloat16), dO_tile)

            tl.store(dK_ptr + dkb + offs_n[:, None] * stride_dks + offs_d[None, :] * stride_dkd,
                     acc_K.to(tl.bfloat16), mask=mask_n[:, None] & mask_d[None, :])
            tl.store(dV_ptr + dvb + offs_n[:, None] * stride_dvs + offs_d[None, :] * stride_dvd,
                     acc_V.to(tl.bfloat16), mask=mask_n[:, None] & mask_d[None, :])

        _dKV_chunk[(nkt, total_bh)](
            Q, K, V, dO, L, dK, dV,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
            L.stride(0), L.stride(1), L.stride(2),
            dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
            dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
            B, H, S, d_dim, scale, d_off,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=BLOCK_DMODEL,
            num_warps=2, num_stages=1,
        )