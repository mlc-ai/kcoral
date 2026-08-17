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
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    off_h = tl.program_id(1)
    off_b = tl.program_id(2)
    if off_b >= B or off_h >= H:
        return

    base_offset = off_b * stride_qb + off_h * stride_qh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < S

    # Process each chunk of d dimension
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for dc in range(tl.cdiv(d, BLOCK_D)):
        start_d = dc * BLOCK_D
        offs_d = start_d + tl.arange(0, BLOCK_D)
        mask_d = offs_d < d

        Q_ptrs = Q_ptr + base_offset + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
        K_ptrs = K_ptr + base_offset + offs_d[None, :] * stride_kd
        V_ptrs = V_ptr + base_offset + offs_d[None, :] * stride_vd
        dO_ptrs = dO_ptr + base_offset + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod

        Q_tile = tl.load(Q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)
        dO_tile = tl.load(dO_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)
        L_tile = tl.load(L_ptr + base_offset + offs_m * stride_ls - off_h * 0, mask=mask_m, other=0.0)

        end_sn = tl.cdiv(min((pid_m + 1) * BLOCK_M, S), BLOCK_N)

        for sn in range(end_sn):
            offs_n_base = sn * BLOCK_N + tl.arange(0, BLOCK_N)
            mask_n = offs_n_base < S

            K_tile = tl.load(K_ptrs + offs_n_base[:, None] * stride_ks,
                             mask=mask_n[:, None] & mask_d[None, :], other=0.0)
            V_tile = tl.load(V_ptrs + offs_n_base[:, None] * stride_vs,
                             mask=mask_n[:, None] & mask_d[None, :], other=0.0)

            scores = tl.dot(Q_tile, tl.trans(K_tile)) * SCALE

            causal_mask = (offs_n_base[None, :] <= offs_m[:, None])
            valid = causal_mask & mask_m[:, None] & mask_n[None, :]

            attn = tl.where(valid, tl.exp(scores - L_tile[:, None]), 0.0)
            D_mat = tl.dot(dO_tile, tl.trans(V_tile))
            corr = tl.sum(attn * D_mat, axis=1, keep_dims=True)
            dp = attn * (D_mat - corr)

            acc += tl.dot(dp.to(tl.bfloat16), K_tile) * SCALE

    dQ_ptrs = dQ_ptr + base_offset + offs_m[:, None] * stride_dqs
    for dc in range(tl.cdiv(d, BLOCK_D)):
        start_d = dc * BLOCK_D
        offs_d = start_d + tl.arange(0, BLOCK_D)
        mask_d = offs_d < d
        dQ_idx = acc
        tl.store(dQ_ptrs + offs_d[None, :] * stride_dqd,
                 dQ_idx[:, start_d:start_d+BLOCK_D].to(tl.bfloat16),
                 mask=mask_m[:, None] & mask_d[None, :])


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
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_n = tl.program_id(0)
    off_h = tl.program_id(1)
    off_b = tl.program_id(2)
    if off_b >= B or off_h >= H:
        return

    base_offset = off_b * stride_qb + off_h * stride_qh

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < S

    acc_K = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    acc_V = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)

    num_sm = tl.cdiv(S, BLOCK_M)

    for sm in range(num_sm):
        end_q = min((sm + 1) * BLOCK_M, S)
        if end_q <= pid_n * BLOCK_N:
            continue

        offs_m = sm * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S

        for dc in range(tl.cdiv(d, BLOCK_D)):
            start_d = dc * BLOCK_D
            offs_d = start_d + tl.arange(0, BLOCK_D)
            mask_d = offs_d < d

            Q_ptrs = Q_ptr + base_offset + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
            K_ptrs = K_ptr + base_offset + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
            V_ptrs = V_ptr + base_offset + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
            dO_ptrs = dO_ptr + base_offset + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod

            Q_tile = tl.load(Q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)
            K_tile = tl.load(K_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
            V_tile = tl.load(V_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
            dO_tile = tl.load(dO_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)

            L_tile = tl.load(L_ptr + off_b * stride_lsb + off_h * stride_lsh + offs_m * stride_ls,
                             mask=mask_m, other=0.0)

            scores = tl.dot(Q_tile, tl.trans(K_tile)) * SCALE
            causal_mask = (offs_n[None, :] <= offs_m[:, None])
            valid = causal_mask & mask_m[:, None] & mask_n[None, :]

            attn = tl.where(valid, tl.exp(scores - L_tile[:, None]), 0.0)
            D_mat = tl.dot(dO_tile, tl.trans(V_tile))
            corr = tl.sum(attn * D_mat, axis=1, keep_dims=True)
            dp = attn * (D_mat - corr)

            acc_K += tl.dot(tl.trans(dp).to(tl.bfloat16), Q_tile) * SCALE
            acc_V += tl.dot(tl.trans(attn).to(tl.bfloat16), dO_tile)

    dK_base = dK_ptr + off_b * stride_dkb + off_h * stride_dkh + offs_n[:, None] * stride_dks
    dV_base = dV_ptr + off_b * stride_dvb + off_h * stride_dvh + offs_n[:, None] * stride_dvs

    for dc in range(tl.cdiv(d, BLOCK_D)):
        start_d = dc * BLOCK_D
        offs_d = start_d + tl.arange(0, BLOCK_D)
        mask_d = offs_d < d

        dK_tile = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
        dV_tile = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
        for di in tl.static_range(BLOCK_D):
            dK_tile = tl.where(mask_d[di:di+1], acc_K[:, start_d+di:start_d+di+1], 0.0)
            dV_tile = tl.where(mask_d[di:di+1], acc_V[:, start_d+di:start_d+di+1], 0.0)

        tl.store(dK_base + offs_d[None, :] * stride_dkd,
                 acc_K[:, start_d:start_d+BLOCK_D].to(tl.bfloat16),
                 mask=mask_n[:, None] & mask_d[None, :])
        tl.store(dV_base + offs_d[None, :] * stride_dvd,
                 acc_V[:, start_d:start_d+BLOCK_D].to(tl.bfloat16),
                 mask=mask_n[:, None] & mask_d[None, :])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Destination-passing entry point for causal MHA backward (bf16, d=128)."""
    torch.cuda.set_device(Q.device)
    B, H, S, d_dim = Q.shape

    scale = 1.0 / math.sqrt(d_dim)

    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_D = 32

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
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=4, num_stages=2,
    )

    # Pass 2: compute dK and dV
    _mha_bwd_dKV_kernel[(nkt, *grid_common)](
        Q, K, V, dO, L, dK, dV,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        B, H, S, d_dim, scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=4, num_stages=2,
    )