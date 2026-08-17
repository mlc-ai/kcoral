import torch
import triton
import triton.language as tl


@triton.jit
def _mha_bwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
    dQ_ptr, dK_ptr, dV_ptr,
    B, H, S, d,
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Fused MHA backward: compute dQ, dK, dV in one pass per (b,h) pair.

    Each program owns one (batch, head) and partitions over (M-tile, N-tile)
    pairs in a wavefront. Within each tile pair we materialize the attention
    block P and dS, then scatter contributions into dQ, dK, dV.

    Grid: (B*H * total_pair_count,)  -- linear grid, each lane picks a (pm,pn)
    """
    pid = tl.program_id(0)
    pid_bh = pid % (B * H)
    pair_id = pid // (B * H)

    bh_stride = S * d
    bh_base = pid_bh * bh_stride

    offs_d = tl.arange(0, d)

    pm = pair_id % tl.cdiv(S, BLOCK_M)
    pn = pair_id // tl.cdiv(S, BLOCK_M)

    offs_m = pm * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < S
    offs_n = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < S
    mask_md = mask_m[:, None] & (offs_d[None, :] < d)
    mask_nd = mask_n[:, None] & (offs_d[None, :] < d)
    mask_mn = mask_m[:, None] & mask_n[None, :]

    # Load Q, dO, O for the M-block
    Q_row  = bh_base + offs_m[:, None] * d + offs_d[None, :]
    dO_row = Q_row  # share base; stride layout is identical
    O_row  = Q_row

    Q_reg  = tl.load(Q_ptr  + Q_row,  mask=mask_md, other=0.0).to(tl.float32)
    dO_reg = tl.load(dO_ptr + dO_row, mask=mask_md, other=0.0).to(tl.float32)
    O_reg  = tl.load(O_ptr  + O_row,  mask=mask_md, other=0.0).to(tl.float32)

    # Load K, V for the N-block
    K_col  = bh_base + offs_n[:, None] * d + offs_d[None, :]
    K_reg  = tl.load(K_ptr  + K_col, mask=mask_nd, other=0.0).to(tl.float32)
    V_reg  = tl.load(V_ptr  + K_col, mask=mask_nd, other=0.0).to(tl.float32)

    # Load L for the M-block
    L_reg = tl.load(L_ptr + pid_bh * S + offs_m, mask=mask_m, other=0.0)

    # === Forward ===
    S_mat = tl.dot(Q_reg, K_reg.T) * sm_scale           # [BM, BN]
    P     = tl.exp(S_mat - L_reg[:, None])              # [BM, BN]

    # === Backward signals ===
    D = tl.sum(dO_reg * O_reg, axis=1)                  # [BM]
    dP = tl.dot(dO_reg, V_reg.T)                        # [BM, BN]
    dS = P * (dP - D[:, None]) * sm_scale               # [BM, BN]

    # Partial contributions for this (pm, pn) pair.
    # These must accumulate INTO the global output via atomics or
    # a second reduction pass. Since atomics would serialize badly,
    # we use atomic_add for the cross-scatter terms.

    # dQ_contrib[m,d] = sum_n dS[m,n] * K[n,d]
    #   shape [BM, D]
    dq_part = tl.dot(dS, K_reg)                                    # [BM, BD]

    # dK_contrib[n,d] = sum_m dS[m,n] * Q[m,d]
    #   shape [BN, D]
    dk_part = tl.dot(dS.T, Q_reg)                                  # [BN, BD]

    # dV_contrib[n,d] = sum_m P[m,n] * dO[m,d]
    #   shape [BN, D]
    dv_part = tl.dot(P.T, dO_reg)                                  # [BN, BD]

    # Scatter-add dQ into output (no collisions across programs
    # because each (pid_bh, pm) writes distinct rows)
    tl.store(dQ_ptr + Q_row, dq_part.to(tl.bfloat16), mask=mask_md)

    # dK and dV: multiple pm values contribute to the SAME n-row,
    # so we need atomics on dK and dV columns.
    # Use per-element atomic_add.
    for i in range(BLOCK_N):
        row_idx = offs_n[i]
        if mask_n[i]:
            for j in range(d):
                col_idx = offs_d[j]
                if col_idx < d:
                    offset_k = bh_base + row_idx * d + col_idx
                    val_k = dk_part[i, j]
                    tl.atomic_add(dK_ptr + offset_k, val_k.to(tl.bfloat16))
                    val_v = dv_part[i, j]
                    tl.atomic_add(dV_ptr + offset_k, val_v.to(tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Multi-head attention backward: compute dQ, dK, dV."""
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    sm_scale = 1.0 / (d ** 0.5)

    BLOCK_M = 64
    BLOCK_N = 64
    BH = B * H
    num_m = triton.cdiv(S, BLOCK_M)
    num_n = triton.cdiv(S, BLOCK_N)
    total_pairs = num_m * num_n
    grid_size = BH * total_pairs

    num_warps = 8
    num_stages = 3

    # Zero outputs before atomics
    dQ.zero_()
    dK.zero_()
    dV.zero_()

    _mha_bwd_kernel[(grid_size,)](
        Q, K, V, O, dO, L,
        dQ, dK, dV,
        B, H, S, d, sm_scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=num_warps, num_stages=num_stages,
    )