import torch
import triton
import triton.language as tl


@triton.jit
def _fill_zeros(out_ptr, num_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < num_elements
    tl.store(out_ptr + offsets, 0.0, mask=mask)


@triton.jit
def _mha_bwd_dv(
    Q, K, dO, L, dV,
    B, H, S, d_head,
    qsb, qsh, qss, qsd,
    ksb, ksh, kss, ksd,
    dOsb, dOsh, dOss, dOsd,
    Lsb, Lsh, Lss,
    dVsb, dVsh, dVss, dVsd,
    scale,
    NUM_M_ITER: tl.constexpr,
    INF: tl.constexpr,
    TILE_Q: tl.constexpr,
    TILE_K: tl.constexpr,
    TILE_D: tl.constexpr,
):
    """Each program owns one key-tile of dV. Sweeps all query tiles."""
    pid_bh = tl.program_id(0)
    pid_k = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    start_k = pid_k * TILE_K
    off_k = start_k + tl.arange(0, TILE_K)
    k_mask = off_k < S
    off_d = tl.arange(0, TILE_D)
    d_mask = off_d < d_head

    k_base = b * ksb + h * ksh
    K_tile = tl.load(
        K + k_base + off_k[:, None] * kss + off_d[None, :] * ksd,
        mask=k_mask[:, None] & d_mask[None, :], other=0.0,
    ).to(tl.float32)

    acc = tl.zeros((TILE_K, TILE_D), dtype=tl.float32)

    for qi in range(NUM_M_ITER):
        start_q = qi * TILE_Q
        off_q = start_q + tl.arange(0, TILE_Q)
        q_mask = off_q < S

        q_base = b * qsb + h * qsh
        Q_tile = tl.load(
            Q + q_base + off_q[:, None] * qss + off_d[None, :] * qsd,
            mask=q_mask[:, None] & d_mask[None, :], other=0.0,
        ).to(tl.float32)

        logits = tl.dot(Q_tile, K_tile) * scale

        l_base = b * Lsb + h * Lsh
        L_val = tl.load(L + l_base + off_q * Lss, mask=q_mask, other=INF).to(tl.float32)
        L_safe = tl.where(q_mask[:, None], L_val[:, None], INF)

        attn = tl.exp(logits - L_safe)
        attn = tl.where(q_mask[:, None] & k_mask[None, :], attn, 0.0)

        do_base = b * dOsb + h * dOsh
        dO_tile = tl.load(
            dO + do_base + off_q[:, None] * dOss + off_d[None, :] * dOsd,
            mask=q_mask[:, None] & d_mask[None, :], other=0.0,
        ).to(tl.float32)

        acc = tl.dot(attn.T, dO_tile, acc)

    dv_base = b * dVsb + h * dVsh
    tl.store(
        dV + dv_base + off_k[:, None] * dVss + off_d[None, :] * dVsd,
        acc.to(tl.bfloat16),
        mask=k_mask[:, None] & d_mask[None, :],
    )


@triton.jit
def _mha_bwd_dq(
    Q, K, V, dO, L, dQ,
    B, H, S, d_head,
    qsb, qsh, qss, qsd,
    ksb, ksh, kss, ksd,
    vbs, vsh, vss, vsd,
    dOsb, dOsh, dOss, dOsd,
    Lsb, Lsh, Lss,
    dQsb, dQsh, dQss, dQsd,
    scale,
    NUM_K_ITER: tl.constexpr,
    INF: tl.constexpr,
    TILE_Q: tl.constexpr,
    TILE_K: tl.constexpr,
    TILE_D: tl.constexpr,
):
    """Each program owns one query-tile of dQ. Sweeps all key tiles.
    Computes dQ = scale * (sum_k(A*dA*^t[K]) - r[:,None] * sum_k(A*^t[K])).
    """
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    start_q = pid_q * TILE_Q
    off_q = start_q + tl.arange(0, TILE_Q)
    q_mask = off_q < S
    off_d = tl.arange(0, TILE_D)
    d_mask = off_d < d_head

    q_base = b * qsb + h * qsh
    Q_tile = tl.load(
        Q + q_base + off_q[:, None] * qss + off_d[None, :] * qsd,
        mask=q_mask[:, None] & d_mask[None, :], other=0.0,
    ).to(tl.float32)

    do_base = b * dOsb + h * dOsh
    dO_tile = tl.load(
        dO + do_base + off_q[:, None] * dOss + off_d[None, :] * dOsd,
        mask=q_mask[:, None] & d_mask[None, :], other=0.0,
    ).to(tl.float32)

    l_base = b * Lsb + h * Lsh
    L_val = tl.load(L + l_base + off_q * Lss, mask=q_mask, other=INF).to(tl.float32)

    k_base = b * ksb + h * ksh
    v_base = b * vbs + h * vsh
    dqb = b * dQsb + h * dQsh

    C1 = tl.zeros((TILE_Q, TILE_D), dtype=tl.float32)
    C2 = tl.zeros((TILE_Q, TILE_D), dtype=tl.float32)
    r_acc = tl.zeros((TILE_Q,), dtype=tl.float32)

    for ki in range(NUM_K_ITER):
        start_k = ki * TILE_K
        off_k = start_k + tl.arange(0, TILE_K)
        k_mask = off_k < S
        valid = q_mask[:, None] & k_mask[None, :]

        K_tile = tl.load(
            K + k_base + off_k[:, None] * kss + off_d[None, :] * ksd,
            mask=k_mask[:, None] & d_mask[None, :], other=0.0,
        ).to(tl.float32)
        V_tile = tl.load(
            V + v_base + off_k[:, None] * vss + off_d[None, :] * vsd,
            mask=k_mask[:, None] & d_mask[None, :], other=0.0,
        ).to(tl.float32)

        logits = tl.dot(Q_tile, K_tile) * scale
        L_safe = tl.where(q_mask[:, None], L_val[:, None], INF)
        attn = tl.exp(logits - L_safe)
        attn = tl.where(valid, attn, 0.0)

        dA = tl.dot(dO_tile, V_tile.T)
        aw = attn * dA

        r_acc = r_acc + tl.sum(aw, axis=-1)
        C1 = tl.dot(aw, K_tile, C1)
        C2 = tl.dot(attn, K_tile, C2)

    dQ_out = (C1 - r_acc[:, None] * C2) * scale
    tl.store(
        dQ + dqb + off_q[:, None] * dQss + off_d[None, :] * dQsd,
        dQ_out.to(tl.bfloat16),
        mask=q_mask[:, None] & d_mask[None, :],
    )


@triton.jit
def _mha_bwd_dk(
    Q, K, V, dO, L, dK,
    B, H, S, d_head,
    qsb, qsh, qss, qsd,
    ksb, ksh, kss, ksd,
    vbs, vsh, vss, vsd,
    dOsb, dOsh, dOss, dOsd,
    Lsb, Lsh, Lss,
    dKsb, dKsh, dKss, dKsd,
    scale,
    NUM_M_ITER: tl.constexpr,
    NUM_K_ITER: tl.constexpr,
    INF: tl.constexpr,
    TILE_Q: tl.constexpr,
    TILE_K: tl.constexpr,
    TILE_D: tl.constexpr,
):
    """Each program owns one key-tile of dK.
    For each query-tile, first computes r[m] by scanning all key-tiles,
    then accumulates dK contribution for this key-tile.
    """
    pid_bh = tl.program_id(0)
    pid_k = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    start_k_own = pid_k * TILE_K
    off_k_own = start_k_own + tl.arange(0, TILE_K)
    k_own_mask = off_k_own < S
    off_d = tl.arange(0, TILE_D)
    d_mask = off_d < d_head

    k_base = b * ksb + h * ksh
    v_base = b * vbs + h * vsh
    K_own = tl.load(
        K + k_base + off_k_own[:, None] * kss + off_d[None, :] * ksd,
        mask=k_own_mask[:, None] & d_mask[None, :], other=0.0,
    ).to(tl.float32)
    V_own = tl.load(
        V + v_base + off_k_own[:, None] * vss + off_d[None, :] * vsd,
        mask=k_own_mask[:, None] & d_mask[None, :], other=0.0,
    ).to(tl.float32)

    acc = tl.zeros((TILE_K, TILE_D), dtype=tl.float32)

    for mi in range(NUM_M_ITER):
        start_q = mi * TILE_Q
        off_q = start_q + tl.arange(0, TILE_Q)
        q_mask = off_q < S

        q_base = b * qsb + h * qsh
        Q_tile = tl.load(
            Q + q_base + off_q[:, None] * qss + off_d[None, :] * qsd,
            mask=q_mask[:, None] & d_mask[None, :], other=0.0,
        ).to(tl.float32)

        do_base = b * dOsb + h * dOsh
        dO_tile = tl.load(
            dO + do_base + off_q[:, None] * dOss + off_d[None, :] * dOsd,
            mask=q_mask[:, None] & d_mask[None, :], other=0.0,
        ).to(tl.float32)

        l_base = b * Lsb + h * Lsh
        L_val = tl.load(L + l_base + off_q * Lss, mask=q_mask, other=INF).to(tl.float32)

        # Step A: compute r[m] by scanning ALL key tiles
        r_local = tl.zeros((TILE_Q,), dtype=tl.float32)
        for ki in range(NUM_K_ITER):
            start_k_all = ki * TILE_K
            off_k_all = start_k_all + tl.arange(0, TILE_K)
            k_all_mask = off_k_all < S
            valid = q_mask[:, None] & k_all_mask[None, :]

            Ka = tl.load(
                K + k_base + off_k_all[:, None] * kss + off_d[None, :] * ksd,
                mask=k_all_mask[:, None] & d_mask[None, :], other=0.0,
            ).to(tl.float32)
            Va = tl.load(
                V + v_base + off_k_all[:, None] * vss + off_d[None, :] * vsd,
                mask=k_all_mask[:, None] & d_mask[None, :], other=0.0,
            ).to(tl.float32)

            logits_a = tl.dot(Q_tile, Ka) * scale
            Lsa = tl.where(q_mask[:, None], L_val[:, None], INF)
            attn_a = tl.exp(logits_a - Lsa)
            attn_a = tl.where(valid, attn_a, 0.0)

            dA_a = tl.dot(dO_tile, Va.T)
            r_local = r_local + tl.sum(attn_a * dA_a, axis=-1)

        # Step B: compute dK contribution for THIS key tile
        valid_own = q_mask[:, None] & k_own_mask[None, :]
        logits_own = tl.dot(Q_tile, K_own) * scale
        Lso = tl.where(q_mask[:, None], L_val[:, None], INF)
        attn_own = tl.exp(logits_own - Lso)
        attn_own = tl.where(valid_own, attn_own, 0.0)

        dA_own = tl.dot(dO_tile, V_own.T)
        dS_own = attn_own * (dA_own - r_local[:, None])

        acc = tl.dot(dS_own.T, Q_tile, acc)

    dk_base = b * dKsb + h * dKsh
    tl.store(
        dK + dk_base + off_k_own[:, None] * dKss + off_d[None, :] * dKsd,
        (acc * scale).to(tl.bfloat16),
        mask=k_own_mask[:, None] & d_mask[None, :],
    )


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute gradients dQ, dK, dV for non-causal multi-head attention backward."""
    torch.cuda.set_device(Q.device)

    B, H, S, D = Q.shape
    scale = 1.0 / float(D ** 0.5)

    TILE_Q = 128
    TILE_K = 128
    TILE_D = 128
    INF = float(1e20)

    num_bh = B * H
    num_qtiles = triton.cdiv(S, TILE_Q)
    num_ktiles = triton.cdiv(S, TILE_K)

    qs = list(Q.stride())
    ks = list(K.stride())
    vs = list(V.stride())
    ds = list(dO.stride())
    ls = list(L.stride())
    dqs = list(dQ.stride())
    dks = list(dK.stride())
    dvs = list(dV.stride())

    # Zero-initialize all outputs
    for out in (dQ, dK, dV):
        n = out.numel()
        if n > 0:
            _fill_zeros[(triton.cdiv(n, 1024),)](out, n, BLOCK=1024)

    if S == 0:
        return

    # dV kernel: grid over (bh, k_tiles), sweeps q_tiles
    _mha_bwd_dv[(num_bh, num_ktiles)](
        Q, K, dO, L, dV,
        B, H, S, D,
        *qs, *ks, *ds, *ls, *dvs,
        scale,
        NUM_M_ITER=num_qtiles, INF=INF,
        TILE_Q=TILE_Q, TILE_K=TILE_K, TILE_D=TILE_D,
        num_warps=4, num_stages=2,
    )

    # dQ kernel: grid over (bh, q_tiles), sweeps k_tiles
    _mha_bwd_dq[(num_bh, num_qtiles)](
        Q, K, V, dO, L, dQ,
        B, H, S, D,
        *qs, *ks, *vs, *ds, *ls, *dqs,
        scale,
        NUM_K_ITER=num_ktiles, INF=INF,
        TILE_Q=TILE_Q, TILE_K=TILE_K, TILE_D=TILE_D,
        num_warps=4, num_stages=2,
    )

    # dK kernel: grid over (bh, k_tiles), sweeps q_tiles (each with full k-scan for r)
    _mha_bwd_dk[(num_bh, num_ktiles)](
        Q, K, V, dO, L, dK,
        B, H, S, D,
        *qs, *ks, *vs, *ds, *ls, *dks,
        scale,
        NUM_M_ITER=num_qtiles, NUM_K_ITER=num_ktiles, INF=INF,
        TILE_Q=TILE_Q, TILE_K=TILE_K, TILE_D=TILE_D,
        num_warps=4, num_stages=2,
    )