import math
import torch
import triton
import triton.language as tl


@triton.jit
def _dKdV_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
    dK_ptr, dV_ptr,
    stride_bq, stride_hq, stride_sq, stride_dq,
    stride_bk, stride_hk, stride_sk, stride_dk_in,
    stride_bv, stride_hv, stride_sv, stride_dv_in,
    stride_bo, stride_ho, stride_so, stride_do,
    stride_bdO, stride_hdO, stride_sdO, stride_ddO,
    stride_bL, stride_hL, stride_sL,
    stride_bdK, stride_hdK, stride_sdK, stride_ddK,
    stride_bdV, stride_hdV, stride_sdV, stride_ddV,
    B, H, S, D,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute dK and dV for one (batch, head, col_block), iterating all row blocks."""
    pid = tl.program_id(0)
    num_col_blocks = tl.cdiv(S, BLOCK_N)
    col_block = pid % num_col_blocks
    bh = pid // num_col_blocks
    head = bh % H
    batch = bh // H

    offs_n = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < S
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    k_base = batch * stride_bk + head * stride_hk
    k_ptrs = K_ptr + k_base + offs_n[:, None] * stride_sk + offs_d[None, :] * stride_dk_in
    K_tile = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    v_base = batch * stride_bv + head * stride_hv
    v_ptrs = V_ptr + v_base + offs_n[:, None] * stride_sv + offs_d[None, :] * stride_dv_in
    V_tile = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    acc_dK = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    acc_dV = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)

    num_row_blocks = tl.cdiv(S, BLOCK_M)

    for rb in range(num_row_blocks):
        offs_m = rb * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S

        q_base = batch * stride_bq + head * stride_hq
        q_ptrs = Q_ptr + q_base + offs_m[:, None] * stride_sq + offs_d[None, :] * stride_dq
        Q_tile = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        do_base = batch * stride_bdO + head * stride_hdO
        do_ptrs = dO_ptr + do_base + offs_m[:, None] * stride_sdO + offs_d[None, :] * stride_ddO
        dO_tile = tl.load(do_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        o_base = batch * stride_bo + head * stride_ho
        o_ptrs = O_ptr + o_base + offs_m[:, None] * stride_so + offs_d[None, :] * stride_do
        O_tile = tl.load(o_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        l_base = batch * stride_bL + head * stride_hL
        l_ptrs = L_ptr + l_base + offs_m * stride_sL
        L_tile = tl.load(l_ptrs, mask=mask_m, other=0.0)

        D_tile = tl.sum(dO_tile * O_tile, axis=1)

        S_tile = tl.dot(Q_tile, K_tile.T) * scale

        causal = offs_n[None, :] <= offs_m[:, None]
        valid = mask_m[:, None] & mask_n[None, :]
        eff = causal & valid

        P_tile = tl.exp(tl.where(eff, S_tile - L_tile[:, None], float('-inf')))

        dP_tile = tl.dot(dO_tile, V_tile.T)

        dS_fp32 = tl.where(eff, P_tile * (dP_tile - D_tile[:, None]) * scale, 0.0)

        acc_dK = tl.dot(dS_fp32.T, Q_tile, acc=acc_dK)
        acc_dV = tl.dot(P_tile.T, dO_tile, acc=acc_dV)

    dk_base = batch * stride_bdK + head * stride_hdK
    dk_ptrs = dK_ptr + dk_base + offs_n[:, None] * stride_sdK + offs_d[None, :] * stride_ddK
    tl.store(dk_ptrs, acc_dK.to(tl.bfloat16), mask=mask_n[:, None] & mask_d[None, :])

    dv_base = batch * stride_bdV + head * stride_hdV
    dv_ptrs = dV_ptr + dv_base + offs_n[:, None] * stride_sdV + offs_d[None, :] * stride_ddV
    tl.store(dv_ptrs, acc_dV.to(tl.bfloat16), mask=mask_n[:, None] & mask_d[None, :])


@triton.jit
def _dQ_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
    dQ_ptr,
    stride_bq, stride_hq, stride_sq, stride_dq,
    stride_bk, stride_hk, stride_sk, stride_dk_in,
    stride_bv, stride_hv, stride_sv, stride_dv_in,
    stride_bo, stride_ho, stride_so, stride_do,
    stride_bdO, stride_hdO, stride_sdO, stride_ddO,
    stride_bL, stride_hL, stride_sL,
    stride_bdQ, stride_hdQ, stride_sdQ, stride_ddQ,
    B, H, S, D,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute dQ for one (batch, head, row_block), iterating all column blocks."""
    pid = tl.program_id(0)
    num_row_blocks = tl.cdiv(S, BLOCK_M)
    row_block = pid % num_row_blocks
    bh = pid // num_row_blocks
    head = bh % H
    batch = bh // H

    offs_m = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < S
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    q_base = batch * stride_bq + head * stride_hq
    q_ptrs = Q_ptr + q_base + offs_m[:, None] * stride_sq + offs_d[None, :] * stride_dq
    Q_tile = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    do_base = batch * stride_bdO + head * stride_hdO
    do_ptrs = dO_ptr + do_base + offs_m[:, None] * stride_sdO + offs_d[None, :] * stride_ddO
    dO_tile = tl.load(do_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    o_base = batch * stride_bo + head * stride_ho
    o_ptrs = O_ptr + o_base + offs_m[:, None] * stride_so + offs_d[None, :] * stride_do
    O_tile = tl.load(o_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    l_base = batch * stride_bL + head * stride_hL
    l_ptrs = L_ptr + l_base + offs_m * stride_sL
    L_tile = tl.load(l_ptrs, mask=mask_m, other=0.0)

    D_tile = tl.sum(dO_tile * O_tile, axis=1)

    acc_dQ = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    num_col_blocks = tl.cdiv(S, BLOCK_N)

    for cb in range(num_col_blocks):
        offs_n = cb * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S

        k_base = batch * stride_bk + head * stride_hk
        k_ptrs = K_ptr + k_base + offs_n[:, None] * stride_sk + offs_d[None, :] * stride_dk_in
        K_tile = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        v_base = batch * stride_bv + head * stride_hv
        v_ptrs = V_ptr + v_base + offs_n[:, None] * stride_sv + offs_d[None, :] * stride_dv_in
        V_tile = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        S_tile = tl.dot(Q_tile, K_tile.T) * scale

        causal = offs_n[None, :] <= offs_m[:, None]
        valid = mask_m[:, None] & mask_n[None, :]
        eff = causal & valid

        P_tile = tl.exp(tl.where(eff, S_tile - L_tile[:, None], float('-inf')))

        dP_tile = tl.dot(dO_tile, V_tile.T)

        dS_fp32 = tl.where(eff, P_tile * (dP_tile - D_tile[:, None]) * scale, 0.0)

        acc_dQ = tl.dot(dS_fp32, K_tile, acc=acc_dQ)

    dq_base = batch * stride_bdQ + head * stride_hdQ
    dq_ptrs = dQ_ptr + dq_base + offs_m[:, None] * stride_sdQ + offs_d[None, :] * stride_ddQ
    tl.store(dq_ptrs, acc_dQ.to(tl.bfloat16), mask=mask_m[:, None] & mask_d[None, :])


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
    ],
    key=['B', 'H', 'S'],
)
@triton.jit
def _dKdV_tuned_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
    dK_ptr, dV_ptr,
    stride_bq, stride_hq, stride_sq, stride_dq,
    stride_bk, stride_hk, stride_sk, stride_dk_in,
    stride_bv, stride_hv, stride_sv, stride_dv_in,
    stride_bo, stride_ho, stride_so, stride_do,
    stride_bdO, stride_hdO, stride_sdO, stride_ddO,
    stride_bL, stride_hL, stride_sL,
    stride_bdK, stride_hdK, stride_sdK, stride_ddK,
    stride_bdV, stride_hdV, stride_sdV, stride_ddV,
    B, H, S, D,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    num_col_blocks = tl.cdiv(S, BLOCK_N)
    col_block = pid % num_col_blocks
    bh = pid // num_col_blocks
    head = bh % H
    batch = bh // H

    offs_n = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < S
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    k_base = batch * stride_bk + head * stride_hk
    k_ptrs = K_ptr + k_base + offs_n[:, None] * stride_sk + offs_d[None, :] * stride_dk_in
    K_tile = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    v_base = batch * stride_bv + head * stride_hv
    v_ptrs = V_ptr + v_base + offs_n[:, None] * stride_sv + offs_d[None, :] * stride_dv_in
    V_tile = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    acc_dK = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    acc_dV = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)

    num_row_blocks = tl.cdiv(S, BLOCK_M)

    for rb in range(num_row_blocks):
        offs_m = rb * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S

        q_base = batch * stride_bq + head * stride_hq
        q_ptrs = Q_ptr + q_base + offs_m[:, None] * stride_sq + offs_d[None, :] * stride_dq
        Q_tile = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        do_base = batch * stride_bdO + head * stride_hdO
        do_ptrs = dO_ptr + do_base + offs_m[:, None] * stride_sdO + offs_d[None, :] * stride_ddO
        dO_tile = tl.load(do_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        o_base = batch * stride_bo + head * stride_ho
        o_ptrs = O_ptr + o_base + offs_m[:, None] * stride_so + offs_d[None, :] * stride_do
        O_tile = tl.load(o_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        l_base = batch * stride_bL + head * stride_hL
        l_ptrs = L_ptr + l_base + offs_m * stride_sL
        L_tile = tl.load(l_ptrs, mask=mask_m, other=0.0)

        D_tile = tl.sum(dO_tile * O_tile, axis=1)
        S_tile = tl.dot(Q_tile, K_tile.T) * scale

        causal = offs_n[None, :] <= offs_m[:, None]
        valid = mask_m[:, None] & mask_n[None, :]
        eff = causal & valid

        P_tile = tl.exp(tl.where(eff, S_tile - L_tile[:, None], float('-inf')))
        dP_tile = tl.dot(dO_tile, V_tile.T)
        dS_fp32 = tl.where(eff, P_tile * (dP_tile - D_tile[:, None]) * scale, 0.0)

        acc_dK = tl.dot(dS_fp32.T, Q_tile, acc=acc_dK)
        acc_dV = tl.dot(P_tile.T, dO_tile, acc=acc_dV)

    dk_base = batch * stride_bdK + head * stride_hdK
    dk_ptrs = dK_ptr + dk_base + offs_n[:, None] * stride_sdK + offs_d[None, :] * stride_ddK
    tl.store(dk_ptrs, acc_dK.to(tl.bfloat16), mask=mask_n[:, None] & mask_d[None, :])

    dv_base = batch * stride_bdV + head * stride_hdV
    dv_ptrs = dV_ptr + dv_base + offs_n[:, None] * stride_sdV + offs_d[None, :] * stride_ddV
    tl.store(dv_ptrs, acc_dV.to(tl.bfloat16), mask=mask_n[:, None] & mask_d[None, :])


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
    ],
    key=['B', 'H', 'S'],
)
@triton.jit
def _dQ_tuned_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
    dQ_ptr,
    stride_bq, stride_hq, stride_sq, stride_dq,
    stride_bk, stride_hk, stride_sk, stride_dk_in,
    stride_bv, stride_hv, stride_sv, stride_dv_in,
    stride_bo, stride_ho, stride_so, stride_do,
    stride_bdO, stride_hdO, stride_sdO, stride_ddO,
    stride_bL, stride_hL, stride_sL,
    stride_bdQ, stride_hdQ, stride_sdQ, stride_ddQ,
    B, H, S, D,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    num_row_blocks = tl.cdiv(S, BLOCK_M)
    row_block = pid % num_row_blocks
    bh = pid // num_row_blocks
    head = bh % H
    batch = bh // H

    offs_m = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < S
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    q_base = batch * stride_bq + head * stride_hq
    q_ptrs = Q_ptr + q_base + offs_m[:, None] * stride_sq + offs_d[None, :] * stride_dq
    Q_tile = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    do_base = batch * stride_bdO + head * stride_hdO
    do_ptrs = dO_ptr + do_base + offs_m[:, None] * stride_sdO + offs_d[None, :] * stride_ddO
    dO_tile = tl.load(do_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    o_base = batch * stride_bo + head * stride_ho
    o_ptrs = O_ptr + o_base + offs_m[:, None] * stride_so + offs_d[None, :] * stride_do
    O_tile = tl.load(o_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    l_base = batch * stride_bL + head * stride_hL
    l_ptrs = L_ptr + l_base + offs_m * stride_sL
    L_tile = tl.load(l_ptrs, mask=mask_m, other=0.0)

    D_tile = tl.sum(dO_tile * O_tile, axis=1)
    acc_dQ = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    num_col_blocks = tl.cdiv(S, BLOCK_N)

    for cb in range(num_col_blocks):
        offs_n = cb * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S

        k_base = batch * stride_bk + head * stride_hk
        k_ptrs = K_ptr + k_base + offs_n[:, None] * stride_sk + offs_d[None, :] * stride_dk_in
        K_tile = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        v_base = batch * stride_bv + head * stride_hv
        v_ptrs = V_ptr + v_base + offs_n[:, None] * stride_sv + offs_d[None, :] * stride_dv_in
        V_tile = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        S_tile = tl.dot(Q_tile, K_tile.T) * scale
        causal = offs_n[None, :] <= offs_m[:, None]
        valid = mask_m[:, None] & mask_n[None, :]
        eff = causal & valid

        P_tile = tl.exp(tl.where(eff, S_tile - L_tile[:, None], float('-inf')))
        dP_tile = tl.dot(dO_tile, V_tile.T)
        dS_fp32 = tl.where(eff, P_tile * (dP_tile - D_tile[:, None]) * scale, 0.0)

        acc_dQ = tl.dot(dS_fp32, K_tile, acc=acc_dQ)

    dq_base = batch * stride_bdQ + head * stride_hdQ
    dq_ptrs = dQ_ptr + dq_base + offs_m[:, None] * stride_sdQ + offs_d[None, :] * stride_ddQ
    tl.store(dq_ptrs, acc_dQ.to(tl.bfloat16), mask=mask_m[:, None] & mask_d[None, :])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Causal MHA backward: compute dQ, dK, dV from Q, K, V, O, dO, L."""
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    scale = 1.0 / math.sqrt(D)

    BLOCK_D = 128

    sb_q, sh_q, ss_q, sd_q = Q.stride()
    sb_k, sh_k, ss_k, sd_k = K.stride()
    sb_v, sh_v, ss_v, sd_v = V.stride()
    sb_o, sh_o, ss_o, sd_o = O.stride()
    sb_dO, sh_dO, ss_dO, sd_dO = dO.stride()
    sb_L, sh_L, ss_L = L.stride()
    sb_dQ, sh_dQ, ss_dQ, sd_dQ = dQ.stride()
    sb_dK, sh_dK, ss_dK, sd_dK = dK.stride()
    sb_dV, sh_dV, ss_dV, sd_dV = dV.stride()

    all_strides = [
        sb_q, sh_q, ss_q, sd_q,
        sb_k, sh_k, ss_k, sd_k,
        sb_v, sh_v, ss_v, sd_v,
        sb_o, sh_o, ss_o, sd_o,
        sb_dO, sh_dO, ss_dO, sd_dO,
        sb_L, sh_L, ss_L,
    ]

    meta_common = dict(BLOCK_D=BLOCK_D)

    # --- dK / dV kernel ---
    nk_pids = B * H * triton.cdiv(S, 64)  # approx upper bound for tuning
    if nk_pids > 0:
        grid_dkv = lambda META: (B * H * triton.cdiv(S, META['BLOCK_N']),)
        _dKdV_tuned_kernel[grid_dkv](
            Q, K, V, O, dO, L,
            dK, dV,
            *all_strides,
            sb_dK, sh_dK, ss_dK, sd_dK,
            sb_dV, sh_dV, ss_dV, sd_dV,
            B, H, S, D,
            scale,
            **meta_common,
        )

    # --- dQ kernel ---
    if nk_pids > 0:
        grid_dq = lambda META: (B * H * triton.cdiv(S, META['BLOCK_M']),)
        _dQ_tuned_kernel[grid_dq](
            Q, K, V, O, dO, L,
            dQ,
            *all_strides,
            sb_dQ, sh_dQ, ss_dQ, sd_dQ,
            B, H, S, D,
            scale,
            **meta_common,
        )