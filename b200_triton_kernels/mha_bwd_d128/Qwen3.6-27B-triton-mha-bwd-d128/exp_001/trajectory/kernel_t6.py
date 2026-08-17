import torch
import triton
import triton.language as tl


@triton.jit
def _fill_zeros_kernel(out_ptr, num_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < num_elements
    tl.store(out_ptr + offsets, 0.0, mask=mask)


@triton.jit
def _compute_r_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, R_storage_ptr,
    B, H, S, d_head,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_dOb, stride_dOh, stride_dOs, stride_dOd,
    stride_Lb, stride_Lh, stride_Ls,
    stride_Rsb, stride_Rsh, stride_Rss,
    scale,
    INF_VAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    R_COL: tl.constexpr,
):
    """Compute r[b,h,m] = sum_n(attn[m,n]*dA[m,n]), store at R_storage[b,h,m,R_COL]."""
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    q_bh_off = b * stride_qb + h * stride_qh
    k_bh_off = b * stride_kb + h * stride_kh
    v_bh_off = b * stride_vb + h * stride_vh
    dov_bh_off = b * stride_dOb + h * stride_dOh
    l_bh_off = b * stride_Lb + h * stride_Lh
    r_bh_off = b * stride_Rsb + h * stride_Rsh

    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    m_mask = offs_m < S
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < d_head

    q_ptrs = Q_ptr + q_bh_off + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    raw_q = tl.load(q_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)
    Q_tile = raw_q.to(tl.float32)

    do_ptrs = dO_ptr + dov_bh_off + offs_m[:, None] * stride_dOs + offs_d[None, :] * stride_dOd
    raw_do = tl.load(do_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)
    dO_tile = raw_do.to(tl.float32)

    l_ptrs = L_ptr + l_bh_off + offs_m * stride_Ls
    raw_l = tl.load(l_ptrs, mask=m_mask, other=INF_VAL)
    L_tile = raw_l.to(tl.float32)

    r_acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    num_n_steps = tl.cdiv(S, BLOCK_N)
    for step_n in range(num_n_steps):
        start_n = step_n * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < S
        valid = m_mask[:, None] & n_mask[None, :]

        k_ptrs = K_ptr + k_bh_off + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        raw_k = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
        K_tile = raw_k.to(tl.float32)

        v_ptrs = V_ptr + v_bh_off + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
        raw_v = tl.load(v_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
        V_tile = raw_v.to(tl.float32)

        logits = tl.dot(Q_tile, K_tile) * scale
        L_safe = tl.where(m_mask[:, None], L_tile[:, None], INF_VAL)
        attn = tl.exp(logits - L_safe)
        attn = tl.where(valid, attn, 0.0)

        dA = tl.dot(dO_tile, V_tile.T)
        aw = attn * dA
        r_acc = r_acc + tl.sum(aw, axis=-1)

    r_out = r_acc.to(tl.bfloat16)
    r_ptrs = R_storage_ptr + r_bh_off + offs_m * stride_Rss + R_COL
    tl.store(r_ptrs, r_out, mask=m_mask)


@triton.jit
def _mha_bwd_dv_kernel(
    Q_ptr, K_ptr, dO_ptr, L_ptr, dV_ptr,
    B, H, S, d_head,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_dOb, stride_dOh, stride_dOs, stride_dOd,
    stride_Lb, stride_Lh, stride_Ls,
    stride_dVb, stride_dVh, stride_dVs, stride_dVd,
    scale,
    INF_VAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute dV[b,h,:,d] = sum_m(attention[m,n]*dO[m,:])."""
    pid_bh = tl.program_id(0)
    pid_n = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    k_bh_off = b * stride_kb + h * stride_kh
    dv_bh_off = b * stride_dVb + h * stride_dVh
    dov_bh_off = b * stride_dOb + h * stride_dOh
    q_bh_off = b * stride_qb + h * stride_qh
    l_bh_off = b * stride_Lb + h * stride_Lh

    start_n = pid_n * BLOCK_N
    offs_n = start_n + tl.arange(0, BLOCK_N)
    n_mask = offs_n < S
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < d_head

    k_ptrs = K_ptr + k_bh_off + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    raw_k = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
    K_tile = raw_k.to(tl.float32)

    dv_acc = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    num_m_steps = tl.cdiv(S, BLOCK_M)
    for step_m in range(num_m_steps):
        start_m = step_m * BLOCK_M
        offs_m = start_m + tl.arange(0, BLOCK_M)
        m_mask = offs_m < S
        valid = m_mask[:, None] & n_mask[None, :]

        q_ptrs = Q_ptr + q_bh_off + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
        raw_q = tl.load(q_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)
        Q_tile = raw_q.to(tl.float32)

        logits = tl.dot(Q_tile, K_tile) * scale
        l_ptrs = L_ptr + l_bh_off + offs_m * stride_Ls
        raw_l = tl.load(l_ptrs, mask=m_mask, other=INF_VAL)
        L_tile = raw_l.to(tl.float32)

        L_safe = tl.where(m_mask[:, None], L_tile[:, None], INF_VAL)
        attn = tl.exp(logits - L_safe)
        attn = tl.where(valid, attn, 0.0)

        do_ptrs = dO_ptr + dov_bh_off + offs_m[:, None] * stride_dOs + offs_d[None, :] * stride_dOd
        raw_do = tl.load(do_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)
        dO_tile = raw_do.to(tl.float32)
        dv_acc = tl.dot(attn.T, dO_tile, dv_acc)

    dv_ptrs = dV_ptr + dv_bh_off + offs_n[:, None] * stride_dVs + offs_d[None, :] * stride_dVd
    tl.store(dv_ptrs, dv_acc.to(tl.bfloat16), mask=n_mask[:, None] & d_mask[None, :])


@triton.jit
def _mha_bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, dQ_ptr,
    B, H, S, d_head,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_dOb, stride_dOh, stride_dOs, stride_dOd,
    stride_Lb, stride_Lh, stride_Ls,
    stride_dQb, stride_dQh, stride_dQs, stride_dQd,
    scale,
    INF_VAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute dQ[b,h,:,d] = scale*(C1 - r*C2)."""
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    q_bh_off = b * stride_qb + h * stride_qh
    k_bh_off = b * stride_kb + h * stride_kh
    v_bh_off = b * stride_vb + h * stride_vh
    dov_bh_off = b * stride_dOb + h * stride_dOh
    l_bh_off = b * stride_Lb + h * stride_Lh
    dq_bh_off = b * stride_dQb + h * stride_dQh

    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    m_mask = offs_m < S
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < d_head

    q_ptrs = Q_ptr + q_bh_off + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    raw_q = tl.load(q_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)
    Q_tile = raw_q.to(tl.float32)

    do_ptrs = dO_ptr + dov_bh_off + offs_m[:, None] * stride_dOs + offs_d[None, :] * stride_dOd
    raw_do = tl.load(do_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)
    dO_tile = raw_do.to(tl.float32)

    l_ptrs = L_ptr + l_bh_off + offs_m * stride_Ls
    raw_l = tl.load(l_ptrs, mask=m_mask, other=INF_VAL)
    L_tile = raw_l.to(tl.float32)

    C1 = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    C2 = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    r_acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    num_n_steps = tl.cdiv(S, BLOCK_N)
    for step_n in range(num_n_steps):
        start_n = step_n * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < S
        valid = m_mask[:, None] & n_mask[None, :]

        k_ptrs = K_ptr + k_bh_off + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        raw_k = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
        K_tile = raw_k.to(tl.float32)

        v_ptrs = V_ptr + v_bh_off + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
        raw_v = tl.load(v_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
        V_tile = raw_v.to(tl.float32)

        logits = tl.dot(Q_tile, K_tile) * scale
        L_safe = tl.where(m_mask[:, None], L_tile[:, None], INF_VAL)
        attn = tl.exp(logits - L_safe)
        attn = tl.where(valid, attn, 0.0)

        dA = tl.dot(dO_tile, V_tile.T)
        aw = attn * dA
        r_acc = r_acc + tl.sum(aw, axis=-1)
        C1 = tl.dot(aw, K_tile, C1)
        C2 = tl.dot(attn, K_tile, C2)

    dQ_out = (C1 - r_acc[:, None] * C2) * scale
    dq_ptrs = dQ_ptr + dq_bh_off + offs_m[:, None] * stride_dQs + offs_d[None, :] * stride_dQd
    tl.store(dq_ptrs, dQ_out.to(tl.bfloat16), mask=m_mask[:, None] & d_mask[None, :])


@triton.jit
def _mha_bwd_dk_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, R_storage_ptr, dK_ptr,
    B, H, S, d_head,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_dOb, stride_dOh, stride_dOs, stride_dOd,
    stride_Lb, stride_Lh, stride_Ls,
    stride_Rsb, stride_Rsh, stride_Rss,
    stride_dKb, stride_dKh, stride_dKs, stride_dKd,
    scale,
    INF_VAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    R_COL: tl.constexpr,
):
    """Compute dK[b,h,:,d] = scale * sum_m(attn*(dA-r)*Q), using precomputed r."""
    pid_bh = tl.program_id(0)
    pid_n = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    q_bh_off = b * stride_qb + h * stride_qh
    k_bh_off = b * stride_kb + h * stride_kh
    v_bh_off = b * stride_vb + h * stride_vh
    dov_bh_off = b * stride_dOb + h * stride_dOh
    l_bh_off = b * stride_Lb + h * stride_Lh
    r_bh_off = b * stride_Rsb + h * stride_Rsh
    dk_bh_off = b * stride_dKb + h * stride_dKh

    start_n = pid_n * BLOCK_N
    offs_n = start_n + tl.arange(0, BLOCK_N)
    n_mask = offs_n < S
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < d_head

    k_ptrs = K_ptr + k_bh_off + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    raw_k = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
    K_tile = raw_k.to(tl.float32)

    v_ptrs = V_ptr + v_bh_off + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
    raw_v = tl.load(v_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
    V_tile = raw_v.to(tl.float32)

    dk_acc = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    num_m_steps = tl.cdiv(S, BLOCK_M)
    for step_m in range(num_m_steps):
        start_m = step_m * BLOCK_M
        offs_m = start_m + tl.arange(0, BLOCK_M)
        m_mask = offs_m < S
        valid = m_mask[:, None] & n_mask[None, :]

        q_ptrs = Q_ptr + q_bh_off + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
        raw_q = tl.load(q_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)
        Q_tile = raw_q.to(tl.float32)

        do_ptrs = dO_ptr + dov_bh_off + offs_m[:, None] * stride_dOs + offs_d[None, :] * stride_dOd
        raw_do = tl.load(do_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)
        dO_tile = raw_do.to(tl.float32)

        l_ptrs = L_ptr + l_bh_off + offs_m * stride_Ls
        raw_l = tl.load(l_ptrs, mask=m_mask, other=INF_VAL)
        L_tile = raw_l.to(tl.float32)

        logits = tl.dot(Q_tile, K_tile) * scale
        L_safe = tl.where(m_mask[:, None], L_tile[:, None], INF_VAL)
        attn = tl.exp(logits - L_safe)
        attn = tl.where(valid, attn, 0.0)

        dA = tl.dot(dO_tile, V_tile.T)

        # Load precomputed r from storage
        r_ptrs = R_storage_ptr + r_bh_off + offs_m * stride_Rss + R_COL
        raw_r = tl.load(r_ptrs, mask=m_mask, other=0.0)
        r_tile = raw_r.to(tl.float32)

        dS = attn * (dA - r_tile[:, None])
        dk_acc = tl.dot(dS.T, Q_tile, dk_acc)

    dk_out = dk_acc * scale
    dk_ptrs = dK_ptr + dk_bh_off + offs_n[:, None] * stride_dKs + offs_d[None, :] * stride_dKd
    tl.store(dk_ptrs, dk_out.to(tl.bfloat16), mask=n_mask[:, None] & d_mask[None, :])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute gradients dQ, dK, dV for non-causal multi-head attention backward."""
    torch.cuda.set_device(Q.device)

    B, H, S, d_head = Q.shape
    scale = 1.0 / (d_head ** 0.5)

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_D = 128
    INF_VAL = 1e20
    R_COL = BLOCK_D - 1  # Store r in the last column of dK buffer

    num_bh = B * H
    num_mtiles = triton.cdiv(S, BLOCK_M)
    num_ntiles = triton.cdiv(S, BLOCK_N)

    qs = Q.stride()
    ks = K.stride()
    vs = V.stride()
    dos = dO.stride()
    ls = L.stride()
    dqs = dQ.stride()
    dks = dK.stride()
    dvs = dV.stride()

    # Zero-initialize outputs
    for out in (dQ, dK, dV):
        n = out.numel()
        if n > 0:
            z_grid = (triton.cdiv(n, 1024),)
            _fill_zeros_kernel[z_grid](out, n, BLOCK=1024)

    if S == 0:
        return

    # Step 1: Compute r[b,h,m] = sum_n(attn*m_n * dA[m,n])
    # Store r in the last column of dK buffer temporarily
    _compute_r_kernel[(num_bh, num_mtiles)](
        Q, K, V, dO, L, dK,
        B, H, S, d_head,
        qs[0], qs[1], qs[2], qs[3],
        ks[0], ks[1], ks[2], ks[3],
        vs[0], vs[1], vs[2], vs[3],
        dos[0], dos[1], dos[2], dos[3],
        ls[0], ls[1], ls[2],
        dks[0], dks[1], dks[2],
        scale,
        INF_VAL=INF_VAL,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        R_COL=R_COL,
        num_warps=4, num_stages=2,
    )

    # Step 2: Compute dV (independent of r and dQ)
    _mha_bwd_dv_kernel[(num_bh, num_ntiles)](
        Q, K, dO, L, dV,
        B, H, S, d_head,
        qs[0], qs[1], qs[2], qs[3],
        ks[0], ks[1], ks[2], ks[3],
        dos[0], dos[1], dos[2], dos[3],
        ls[0], ls[1], ls[2],
        dvs[0], dvs[1], dvs[2], dvs[3],
        scale,
        INF_VAL=INF_VAL,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=4, num_stages=2,
    )

    # Step 3: Compute dQ (independent of dK; overwrites r column harmlessly)
    _mha_bwd_dq_kernel[(num_bh, num_mtiles)](
        Q, K, V, dO, L, dQ,
        B, H, S, d_head,
        qs[0], qs[1], qs[2], qs[3],
        ks[0], ks[1], ks[2], ks[3],
        vs[0], vs[1], vs[2], vs[3],
        dos[0], dos[1], dos[2], dos[3],
        ls[0], ls[1], ls[2],
        dqs[0], dqs[1], dqs[2], dqs[3],
        scale,
        INF_VAL=INF_VAL,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=4, num_stages=2,
    )

    # Step 4: Compute dK using precomputed r from step 1
    # R_storage and dK both point to dK; step 1 wrote r at col R_COL,
    # step 4 reads r then writes full dK, overwriting r (sequential, safe)
    _mha_bwd_dk_kernel[(num_bh, num_ntiles)](
        Q, K, V, dO, L, dK, dK,
        B, H, S, d_head,
        qs[0], qs[1], qs[2], qs[3],
        ks[0], ks[1], ks[2], ks[3],
        vs[0], vs[1], vs[2], vs[3],
        dos[0], dos[1], dos[2], dos[3],
        ls[0], ls[1], ls[2],
        dks[0], dks[1], dks[2],
        dks[0], dks[1], dks[2], dks[3],
        scale,
        INF_VAL=INF_VAL,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        R_COL=R_COL,
        num_warps=4, num_stages=2,
    )