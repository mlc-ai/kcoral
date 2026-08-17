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
    Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, R_ptr,
    B, H, S, d_head,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_dOb, stride_dOh, stride_dOs, stride_dOd,
    stride_Lb, stride_Lh, stride_Ls,
    stride_Rb, stride_Rh, stride_Rs,
    scale,
    INF_VAL: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Precompute r[b,h,q] = sum_k(attention[q,k] * dA[q,k])."""
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    q_bh_off = b * stride_qb + h * stride_qh
    k_bh_off = b * stride_kb + h * stride_kh
    v_bh_off = b * stride_vb + h * stride_vh
    dov_bh_off = b * stride_dOb + h * stride_dOh
    l_bh_off = b * stride_Lb + h * stride_Lh
    r_bh_off = b * stride_Rb + h * stride_Rh

    start_q = pid_q * BLOCK_Q
    offs_q = start_q + tl.arange(0, BLOCK_Q)
    q_mask = offs_q < S
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < d_head

    q_ptrs = Q_ptr + q_bh_off + offs_q[:, None] * stride_qs + offs_d[None, :] * stride_qd
    raw_q = tl.load(q_ptrs, mask=q_mask[:, None] & d_mask[None, :], other=0.0)
    Q_tile = raw_q.to(tl.float32)

    dov_ptrs = dO_ptr + dov_bh_off + offs_q[:, None] * stride_dOs + offs_d[None, :] * stride_dOd
    raw_do = tl.load(dov_ptrs, mask=q_mask[:, None] & d_mask[None, :], other=0.0)
    dO_tile = raw_do.to(tl.float32)

    l_ptrs = L_ptr + l_bh_off + offs_q * stride_Ls
    raw_l = tl.load(l_ptrs, mask=q_mask, other=INF_VAL)
    L_tile = raw_l.to(tl.float32)

    # Accumulate r = sum_k(attn * dA)
    r_acc = tl.zeros((BLOCK_Q,), dtype=tl.float32)

    num_k_steps = tl.cdiv(S, BLOCK_K)
    for step_k in range(num_k_steps):
        start_k = step_k * BLOCK_K
        offs_k = start_k + tl.arange(0, BLOCK_K)
        k_mask = offs_k < S
        valid_qk = q_mask[:, None] & k_mask[None, :]

        k_ptrs = K_ptr + k_bh_off + offs_k[:, None] * stride_ks + offs_d[None, :] * stride_kd
        raw_k = tl.load(k_ptrs, mask=k_mask[:, None] & d_mask[None, :], other=0.0)
        K_tile = raw_k.to(tl.float32)

        v_ptrs = V_ptr + v_bh_off + offs_k[:, None] * stride_vs + offs_d[None, :] * stride_vd
        raw_v = tl.load(v_ptrs, mask=k_mask[:, None] & d_mask[None, :], other=0.0)
        V_tile = raw_v.to(tl.float32)

        logits = tl.dot(Q_tile, K_tile) * scale
        L_safe = tl.where(q_mask[:, None], L_tile[:, None], INF_VAL)
        attn = tl.exp(logits - L_safe)
        attn = tl.where(valid_qk, attn, 0.0)

        dA = tl.dot(dO_tile, V_tile.T)
        aw = attn * dA
        r_acc = r_acc + tl.sum(aw, axis=-1)

    r_ptrs = R_ptr + r_bh_off + offs_q * stride_Rs
    tl.store(r_ptrs, r_acc.to(tl.bfloat16), mask=q_mask)


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
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute dV[b,h,:,d] = attention[b,h,:,:].T @ dO[b,h,:,d]."""
    pid_bh = tl.program_id(0)
    pid_k = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    k_bh_off = b * stride_kb + h * stride_kh
    dv_bh_off = b * stride_dVb + h * stride_dVh
    dov_bh_off = b * stride_dOb + h * stride_dOh
    q_bh_off = b * stride_qb + h * stride_qh
    l_bh_off = b * stride_Lb + h * stride_Lh

    start_k = pid_k * BLOCK_K
    offs_k = start_k + tl.arange(0, BLOCK_K)
    k_mask = offs_k < S
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < d_head

    k_ptrs = K_ptr + k_bh_off + offs_k[:, None] * stride_ks + offs_d[None, :] * stride_kd
    raw_k = tl.load(k_ptrs, mask=k_mask[:, None] & d_mask[None, :], other=0.0)
    K_tile = raw_k.to(tl.float32)

    dv_acc = tl.zeros((BLOCK_K, BLOCK_D), dtype=tl.float32)

    num_q_steps = tl.cdiv(S, BLOCK_Q)
    for step_q in range(num_q_steps):
        start_q = step_q * BLOCK_Q
        offs_q = start_q + tl.arange(0, BLOCK_Q)
        q_mask = offs_q < S
        valid_qk = q_mask[:, None] & k_mask[None, :]

        q_ptrs = Q_ptr + q_bh_off + offs_q[:, None] * stride_qs + offs_d[None, :] * stride_qd
        raw_q = tl.load(q_ptrs, mask=q_mask[:, None] & d_mask[None, :], other=0.0)
        Q_tile = raw_q.to(tl.float32)

        logits = tl.dot(Q_tile, K_tile) * scale

        l_ptrs = L_ptr + l_bh_off + offs_q * stride_Ls
        raw_l = tl.load(l_ptrs, mask=q_mask, other=INF_VAL)
        L_tile = raw_l.to(tl.float32)

        L_safe = tl.where(q_mask[:, None], L_tile[:, None], INF_VAL)
        attn = tl.exp(logits - L_safe)
        attn = tl.where(valid_qk, attn, 0.0)

        dov_ptrs = dO_ptr + dov_bh_off + offs_q[:, None] * stride_dOs + offs_d[None, :] * stride_dOd
        raw_do = tl.load(dov_ptrs, mask=q_mask[:, None] & d_mask[None, :], other=0.0)
        dO_tile = raw_do.to(tl.float32)

        dv_acc = tl.dot(attn.T, dO_tile, dv_acc)

    dv_ptrs = dV_ptr + dv_bh_off + offs_k[:, None] * stride_dVs + offs_d[None, :] * stride_dVd
    tl.store(dv_ptrs, dv_acc.to(tl.bfloat16), mask=k_mask[:, None] & d_mask[None, :])


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
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute dQ[b,h,:,d] = scale * (C1 - r * C2) where
       C1 = sum_k(attn*q_k * dA[q,k] * K[k,:]),
       C2 = sum_k(attn[q,k] * K[k,:]),
       r[q] = sum_k(attn[q,k] * dA[q,k]).
    """
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    q_bh_off = b * stride_qb + h * stride_qh
    k_bh_off = b * stride_kb + h * stride_kh
    v_bh_off = b * stride_vb + h * stride_vh
    dov_bh_off = b * stride_dOb + h * stride_dOh
    l_bh_off = b * stride_Lb + h * stride_Lh
    dq_bh_off = b * stride_dQb + h * stride_dQh

    start_q = pid_q * BLOCK_Q
    offs_q = start_q + tl.arange(0, BLOCK_Q)
    q_mask = offs_q < S
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < d_head

    q_ptrs = Q_ptr + q_bh_off + offs_q[:, None] * stride_qs + offs_d[None, :] * stride_qd
    raw_q = tl.load(q_ptrs, mask=q_mask[:, None] & d_mask[None, :], other=0.0)
    Q_tile = raw_q.to(tl.float32)

    dov_ptrs = dO_ptr + dov_bh_off + offs_q[:, None] * stride_dOs + offs_d[None, :] * stride_dOd
    raw_do = tl.load(dov_ptrs, mask=q_mask[:, None] & d_mask[None, :], other=0.0)
    dO_tile = raw_do.to(tl.float32)

    l_ptrs = L_ptr + l_bh_off + offs_q * stride_Ls
    raw_l = tl.load(l_ptrs, mask=q_mask, other=INF_VAL)
    L_tile = raw_l.to(tl.float32)

    num_k_steps = tl.cdiv(S, BLOCK_K)

    C1 = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)
    C2 = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)
    r = tl.zeros((BLOCK_Q,), dtype=tl.float32)

    for step_k in range(num_k_steps):
        start_k = step_k * BLOCK_K
        offs_k = start_k + tl.arange(0, BLOCK_K)
        k_mask = offs_k < S
        valid_qk = q_mask[:, None] & k_mask[None, :]

        k_ptrs = K_ptr + k_bh_off + offs_k[:, None] * stride_ks + offs_d[None, :] * stride_kd
        raw_k = tl.load(k_ptrs, mask=k_mask[:, None] & d_mask[None, :], other=0.0)
        K_tile = raw_k.to(tl.float32)

        v_ptrs = V_ptr + v_bh_off + offs_k[:, None] * stride_vs + offs_d[None, :] * stride_vd
        raw_v = tl.load(v_ptrs, mask=k_mask[:, None] & d_mask[None, :], other=0.0)
        V_tile = raw_v.to(tl.float32)

        logits = tl.dot(Q_tile, K_tile) * scale
        L_safe = tl.where(q_mask[:, None], L_tile[:, None], INF_VAL)
        attn = tl.exp(logits - L_safe)
        attn = tl.where(valid_qk, attn, 0.0)

        dA = tl.dot(dO_tile, V_tile.T)
        aw = attn * dA

        r = r + tl.sum(aw, axis=-1)
        C1 = tl.dot(aw, K_tile, C1)
        C2 = tl.dot(attn, K_tile, C2)

    dQ_out = (C1 - r[:, None] * C2) * scale
    dq_ptrs = dQ_ptr + dq_bh_off + offs_q[:, None] * stride_dQs + offs_d[None, :] * stride_dQd
    tl.store(dq_ptrs, dQ_out.to(tl.bfloat16), mask=q_mask[:, None] & d_mask[None, :])


@triton.jit
def _mha_bwd_dk_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, R_ptr, dK_ptr,
    B, H, S, d_head,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_dOb, stride_dOh, stride_dOs, stride_dOd,
    stride_Lb, stride_Lh, stride_Ls,
    stride_Rb, stride_Rh, stride_Rs,
    stride_dKb, stride_dKh, stride_dKs, stride_dKd,
    scale,
    INF_VAL: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute dK[b,h,:,d] = scale * sum_q( attn[q,k] * (dA[q,k] - r[q]) * Q[q,:] ).
    
    Each (bh, key-tile) program owns its output tile, no atomics needed.
    Uses precomputed r from _compute_r_kernel.
    """
    pid_bh = tl.program_id(0)
    pid_k = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    q_bh_off = b * stride_qb + h * stride_qh
    k_bh_off = b * stride_kb + h * stride_kh
    v_bh_off = b * stride_vb + h * stride_vh
    dov_bh_off = b * stride_dOb + h * stride_dOh
    l_bh_off = b * stride_Lb + h * stride_Lh
    r_bh_off = b * stride_Rb + h * stride_Rh
    dk_bh_off = b * stride_dKb + h * stride_dKh

    start_k = pid_k * BLOCK_K
    offs_k = start_k + tl.arange(0, BLOCK_K)
    k_mask = offs_k < S
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < d_head

    k_ptrs = K_ptr + k_bh_off + offs_k[:, None] * stride_ks + offs_d[None, :] * stride_kd
    raw_k = tl.load(k_ptrs, mask=k_mask[:, None] & d_mask[None, :], other=0.0)
    K_tile = raw_k.to(tl.float32)

    dk_acc = tl.zeros((BLOCK_K, BLOCK_D), dtype=tl.float32)

    num_q_steps = tl.cdiv(S, BLOCK_Q)
    for step_q in range(num_q_steps):
        start_q = step_q * BLOCK_Q
        offs_q = start_q + tl.arange(0, BLOCK_Q)
        q_mask = offs_q < S
        valid_qk = q_mask[:, None] & k_mask[None, :]

        q_ptrs = Q_ptr + q_bh_off + offs_q[:, None] * stride_qs + offs_d[None, :] * stride_qd
        raw_q = tl.load(q_ptrs, mask=q_mask[:, None] & d_mask[None, :], other=0.0)
        Q_tile = raw_q.to(tl.float32)

        dov_ptrs = dO_ptr + dov_bh_off + offs_q[:, None] * stride_dOs + offs_d[None, :] * stride_dOd
        raw_do = tl.load(dov_ptrs, mask=q_mask[:, None] & d_mask[None, :], other=0.0)
        dO_tile = raw_do.to(tl.float32)

        l_ptrs = L_ptr + l_bh_off + offs_q * stride_Ls
        raw_l = tl.load(l_ptrs, mask=q_mask, other=INF_VAL)
        L_tile = raw_l.to(tl.float32)

        r_ptrs = R_ptr + r_bh_off + offs_q * stride_Rs
        raw_r = tl.load(r_ptrs, mask=q_mask, other=0.0)
        r_tile = raw_r.to(tl.float32)

        logits = tl.dot(Q_tile, K_tile) * scale
        L_safe = tl.where(q_mask[:, None], L_tile[:, None], INF_VAL)
        attn = tl.exp(logits - L_safe)
        attn = tl.where(valid_qk, attn, 0.0)

        dA = tl.dot(dO_tile, K_tile.T)
        # Wait: dA[q,k] = sum_d(dO[q,d] * V[k,d]) -- need V not K!
        # Recompute with V
        v_ptrs = V_ptr + k_bh_off + offs_k[:, None] * stride_vs + offs_d[None, :] * stride_vd
        raw_v = tl.load(v_ptrs, mask=k_mask[:, None] & d_mask[None, :], other=0.0)
        V_tile = raw_v.to(tl.float32)
        dA = tl.dot(dO_tile, V_tile.T)

        # dS[q,k] = attn[q,k] * (dA[q,k] - r[q])
        dS = attn * (dA - r_tile[:, None])

        # dK += dS^T @ Q  => [BLOCK_K, BLOCK_Q]^T @ [BLOCK_Q, BLOCK_D] => [BLOCK_D, BLOCK_Q] wrong shape
        # Actually dS has shape [BLOCK_Q, BLOCK_K], Q_tile has shape [BLOCK_Q, BLOCK_D]
        # dK contribution: sum_q(dS[q,k] * Q[q,d]) = dS.T @ Q_tile => [BLOCK_K, BLOCK_D]
        dk_acc = tl.dot(dS.T, Q_tile, dk_acc)

    dk_out = dk_acc * scale
    dk_ptrs = dK_ptr + dk_bh_off + offs_k[:, None] * stride_dKs + offs_d[None, :] * stride_dKd
    tl.store(dk_ptrs, dk_out.to(tl.bfloat16), mask=k_mask[:, None] & d_mask[None, :])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute gradients dQ, dK, dV for non-causal multi-head attention backward."""
    torch.cuda.set_device(Q.device)

    B, H, S, d_head = Q.shape
    scale = 1.0 / (d_head ** 0.5)

    BLOCK_Q = 128
    BLOCK_K = 128
    BLOCK_D = 128
    INF_VAL = 1e20

    num_bh = B * H
    num_qtiles = triton.cdiv(S, BLOCK_Q)
    num_ktiles = triton.cdiv(S, BLOCK_K)

    qs = Q.stride()
    ks = K.stride()
    vs = V.stride()
    dos = dO.stride()
    ls = L.stride()
    dqs = dQ.stride()
    dks = dK.stride()
    dvs = dV.stride()

    # Zero-initialize all outputs
    for out in (dQ, dK, dV):
        n = out.numel()
        if n > 0:
            z_grid = (triton.cdiv(n, 1024),)
            _fill_zeros_kernel[z_grid](out, n, BLOCK=1024)

    if S == 0:
        return

    # Allocate temporary for r[b,h,S] in fp32 precision (stored as bf16 to minimize footprint)
    # Reuse dQ buffer temporarily: first write r there, then overwrite with actual dQ
    # Actually we can't safely reuse since dQ is the same shape. Let's use a different approach:
    # Store r as fp32 in a fresh allocation... but we can't allocate.
    # Solution: compute r inline within the dK kernel by reusing the dQ computation pattern,
    # or simply pass r through a bf16 tensor that fits the output format.
    # Since dK doesn't read dQ, we can temporarily use dQ for r storage.
    r_ptr = dQ  # temporarily store r here; dQ will be overwritten later

    rs = dQ.stride()  # same strides as dQ (B,H,S,d=1 effectively since we store per-q)

    # Step 1: Precompute r[b,h,q] = sum_k(attn[q,k]*dA[q,k])
    _compute_r_kernel[(num_bh, num_qtiles)](
        Q, K, V, dO, L, r_ptr,
        B, H, S, d_head,
        qs[0], qs[1], qs[2], qs[3],
        ks[0], ks[1], ks[2], ks[3],
        vs[0], vs[1], vs[2], vs[3],
        dos[0], dos[1], dos[2], dos[3],
        ls[0], ls[1], ls[2],
        rs[0], rs[1], rs[2], 1,
        scale,
        INF_VAL=INF_VAL,
        BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D,
        num_warps=4, num_stages=2,
    )

    # Step 2: Compute dV (independent of r)
    _mha_bwd_dv_kernel[(num_bh, num_ktiles)](
        Q, K, dO, L, dV,
        B, H, S, d_head,
        qs[0], qs[1], qs[2], qs[3],
        ks[0], ks[1], ks[2], ks[3],
        dos[0], dos[1], dos[2], dos[3],
        ls[0], ls[1], ls[2],
        dvs[0], dvs[1], dvs[2], dvs[3],
        scale,
        INF_VAL=INF_VAL,
        BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D,
        num_warps=4, num_stages=2,
    )

    # Step 3: Compute dQ (also needs the same r, plus C1 and C2 accumulators)
    _mha_bwd_dq_kernel[(num_bh, num_qtiles)](
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
        BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D,
        num_warps=4, num_stages=2,
    )

    # Step 4: Compute dK using precomputed r
    _mha_bwd_dk_kernel[(num_bh, num_ktiles)](
        Q, K, V, dO, L, r_ptr, dK,
        B, H, S, d_head,
        qs[0], qs[1], qs[2], qs[3],
        ks[0], ks[1], ks[2], ks[3],
        vs[0], vs[1], vs[2], vs[3],
        dos[0], dos[1], dos[2], dos[3],
        ls[0], ls[1], ls[2],
        rs[0], rs[1], rs[2],
        dks[0], dks[1], dks[2], dks[3],
        scale,
        INF_VAL=INF_VAL,
        BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D,
        num_warps=4, num_stages=2,
    )