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
    """Compute dK and dV for one (batch, head, col_block)."""
    pid = tl.program_id(0)
    num_col_blocks = tl.cdiv(S, BLOCK_N)
    col_block = pid % num_col_blocks
    bh = pid // num_col_blocks
    head = bh % H
    batch = bh // H

    col_offs = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = col_offs < S
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D

    # Load K and V tiles once (constant across row-block loop)
    k_base = K_ptr + batch * stride_bk + head * stride_hk
    k_ptrs = k_base + col_offs[:, None] * stride_sk + d_offs[None, :] * stride_dk_in
    K_tile = tl.load(k_ptrs, mask=col_mask[:, None] & d_mask[None, :], other=0.0)

    v_base = V_ptr + batch * stride_bv + head * stride_hv
    v_ptrs = v_base + col_offs[:, None] * stride_sv + d_offs[None, :] * stride_dv_in
    V_tile = tl.load(v_ptrs, mask=col_mask[:, None] & d_mask[None, :], other=0.0)

    dK_acc = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    dV_acc = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)

    num_row_blocks = tl.cdiv(S, BLOCK_M)

    for rb in range(num_row_blocks):
        row_offs = rb * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = row_offs < S

        # Load Q tile
        q_base = Q_ptr + batch * stride_bq + head * stride_hq
        q_ptrs = q_base + row_offs[:, None] * stride_sq + d_offs[None, :] * stride_dq
        Q_tile = tl.load(q_ptrs, mask=row_mask[:, None] & d_mask[None, :], other=0.0)

        # Load dO tile
        do_base = dO_ptr + batch * stride_bdO + head * stride_hdO
        do_ptrs = do_base + row_offs[:, None] * stride_sdO + d_offs[None, :] * stride_ddO
        dO_tile = tl.load(do_ptrs, mask=row_mask[:, None] & d_mask[None, :], other=0.0)

        # Load O tile (needed for D computation)
        o_base = O_ptr + batch * stride_bo + head * stride_ho
        o_ptrs = o_base + row_offs[:, None] * stride_so + d_offs[None, :] * stride_do
        O_tile = tl.load(o_ptrs, mask=row_mask[:, None] & d_mask[None, :], other=0.0)

        # Load L tile
        l_base = L_ptr + batch * stride_bL + head * stride_hL
        l_ptrs = l_base + row_offs * stride_sL
        L_tile = tl.load(l_ptrs, mask=row_mask, other=0.0)

        # D = sum(dO * O, axis=1) along the d-dimension
        D_tile = tl.sum(dO_tile * O_tile, axis=1)

        # Attention scores: S = Q @ K.T * scale
        S_tile = tl.dot(Q_tile, K_tile.T) * scale

        # Causal mask: key position <= query position, AND both in-bounds
        causal = col_offs[None, :] <= row_offs[:, None]
        valid = row_mask[:, None] & col_mask[None, :]
        eff = causal & valid

        # P = exp(S - L) where effective, else 0 (via exp(-inf))
        P_tile = tl.exp(tl.where(eff, S_tile - L_tile[:, None], float('-inf')))

        # dP = dO @ V.T
        dP_tile = tl.dot(dO_tile, V_tile.T)

        # dS = P * (dP - D) * scale, zeroed where not effective
        dS_tile = tl.where(eff, P_tile * (dP_tile - D_tile[:, None]) * scale, 0.0)

        # Accumulate gradients
        dK_acc = tl.dot(dS_tile.T, Q_tile, acc=dK_acc)
        dV_acc = tl.dot(P_tile.T, dO_tile, acc=dV_acc)

    # Store dK
    dk_base = dK_ptr + batch * stride_bdK + head * stride_hdK
    dk_ptrs = dk_base + col_offs[:, None] * stride_sdK + d_offs[None, :] * stride_ddK
    tl.store(dk_ptrs, dK_acc.to(tl.bfloat16), mask=col_mask[:, None] & d_mask[None, :])

    # Store dV
    dv_base = dV_ptr + batch * stride_bdV + head * stride_hdV
    dv_ptrs = dv_base + col_offs[:, None] * stride_sdV + d_offs[None, :] * stride_ddV
    tl.store(dv_ptrs, dV_acc.to(tl.bfloat16), mask=col_mask[:, None] & d_mask[None, :])


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
    """Compute dQ for one (batch, head, row_block)."""
    pid = tl.program_id(0)
    num_row_blocks = tl.cdiv(S, BLOCK_M)
    row_block = pid % num_row_blocks
    bh = pid // num_row_blocks
    head = bh % H
    batch = bh // H

    row_offs = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = row_offs < S
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D

    # Load Q, dO, O once (constant across col-block loop)
    q_base = Q_ptr + batch * stride_bq + head * stride_hq
    q_ptrs = q_base + row_offs[:, None] * stride_sq + d_offs[None, :] * stride_dq
    Q_tile = tl.load(q_ptrs, mask=row_mask[:, None] & d_mask[None, :], other=0.0)

    do_base = dO_ptr + batch * stride_bdO + head * stride_hdO
    do_ptrs = do_base + row_offs[:, None] * stride_sdO + d_offs[None, :] * stride_ddO
    dO_tile = tl.load(do_ptrs, mask=row_mask[:, None] & d_mask[None, :], other=0.0)

    o_base = O_ptr + batch * stride_bo + head * stride_ho
    o_ptrs = o_base + row_offs[:, None] * stride_so + d_offs[None, :] * stride_do
    O_tile = tl.load(o_ptrs, mask=row_mask[:, None] & d_mask[None, :], other=0.0)

    # Load L tile
    l_base = L_ptr + batch * stride_bL + head * stride_hL
    l_ptrs = l_base + row_offs * stride_sL
    L_tile = tl.load(l_ptrs, mask=row_mask, other=0.0)

    # D = sum(dO * O, axis=1) along the d-dimension
    D_tile = tl.sum(dO_tile * O_tile, axis=1)

    dQ_acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    num_col_blocks = tl.cdiv(S, BLOCK_N)

    for cb in range(num_col_blocks):
        col_offs = cb * BLOCK_N + tl.arange(0, BLOCK_N)
        col_mask = col_offs < S

        # Load K tile
        k_base = K_ptr + batch * stride_bk + head * stride_hk
        k_ptrs = k_base + col_offs[:, None] * stride_sk + d_offs[None, :] * stride_dk_in
        K_tile = tl.load(k_ptrs, mask=col_mask[:, None] & d_mask[None, :], other=0.0)

        # Load V tile
        v_base = V_ptr + batch * stride_bv + head * stride_hv
        v_ptrs = v_base + col_offs[:, None] * stride_sv + d_offs[None, :] * stride_dv_in
        V_tile = tl.load(v_ptrs, mask=col_mask[:, None] & d_mask[None, :], other=0.0)

        # Attention scores
        S_tile = tl.dot(Q_tile, K_tile.T) * scale

        # Causal + validity mask
        causal = col_offs[None, :] <= row_offs[:, None]
        valid = row_mask[:, None] & col_mask[None, :]
        eff = causal & valid

        # P = exp(S - L) where effective, else 0
        P_tile = tl.exp(tl.where(eff, S_tile - L_tile[:, None], float('-inf')))

        # dP = dO @ V.T
        dP_tile = tl.dot(dO_tile, V_tile.T)

        # dS = P * (dP - D) * scale, zeroed where not effective
        dS_tile = tl.where(eff, P_tile * (dP_tile - D_tile[:, None]) * scale, 0.0)

        # Accumulate: dQ += dS @ K
        dQ_acc = tl.dot(dS_tile, K_tile, acc=dQ_acc)

    # Store dQ
    dq_base = dQ_ptr + batch * stride_bdQ + head * stride_hdQ
    dq_ptrs = dq_base + row_offs[:, None] * stride_sdQ + d_offs[None, :] * stride_ddQ
    tl.store(dq_ptrs, dQ_acc.to(tl.bfloat16), mask=row_mask[:, None] & d_mask[None, :])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Causal MHA backward: compute dQ, dK, dV from Q, K, V, O, dO, L."""
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    scale = 1.0 / math.sqrt(D)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = 128
    NUM_WARPS = 4
    NUM_STAGES = 2

    # Extract element strides for all tensors
    sb_q, sh_q, ss_q, sd_q = Q.stride()
    sb_k, sh_k, ss_k, sd_k = K.stride()
    sb_v, sh_v, ss_v, sd_v = V.stride()
    sb_o, sh_o, ss_o, sd_o = O.stride()
    sb_dO, sh_dO, ss_dO, sd_dO = dO.stride()
    sb_L, sh_L, ss_L = L.stride()
    sb_dQ, sh_dQ, ss_dQ, sd_dQ = dQ.stride()
    sb_dK, sh_dK, ss_dK, sd_dK = dK.stride()
    sb_dV, sh_dV, ss_dV, sd_dV = dV.stride()

    common_args = (
        Q, K, V, O, dO, L,
        sb_q, sh_q, ss_q, sd_q,
        sb_k, sh_k, ss_k, sd_k,
        sb_v, sh_v, ss_v, sd_v,
        sb_o, sh_o, ss_o, sd_o,
        sb_dO, sh_dO, ss_dO, sd_dO,
        sb_L, sh_L, ss_L,
        B, H, S, D,
        scale,
    )
    meta_kwargs = dict(
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=NUM_WARPS, num_stages=NUM_STAGES,
    )

    # --- dK / dV kernel: each program owns one (batch, head, col_block) ---
    num_col_blocks = triton.cdiv(S, BLOCK_N)
    nk_pids = B * H * num_col_blocks
    if nk_pids > 0:
        _dKdV_kernel[(nk_pids,)](
            *common_args,
            dK, dV,
            sb_dK, sh_dK, ss_dK, sd_dK,
            sb_dV, sh_dV, ss_dV, sd_dV,
            **meta_kwargs,
        )

    # --- dQ kernel: each program owns one (batch, head, row_block) ---
    num_row_blocks = triton.cdiv(S, BLOCK_M)
    dq_pids = B * H * num_row_blocks
    if dq_pids > 0:
        _dQ_kernel[(dq_pids,)](
            *common_args,
            dQ,
            sb_dQ, sh_dQ, ss_dQ, sd_dQ,
            **meta_kwargs,
        )