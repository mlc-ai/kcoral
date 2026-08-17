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

    bh_off = b * stride_qb + h * stride_qh
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

    q_ptrs = Q_ptr + bh_off + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
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
    """Compute dV[b,h,:,d] = sum_q(attention[q,k]*dO[q,:])."""
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
def _mha_bwd_dk_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, dK_ptr,
    B, H, S, d_head,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_dOb, stride_dOh, stride_dOs, stride_dOd,
    stride_Lb, stride_Lh, stride_Ls,
    stride_dKb, stride_dKh, stride_dKs, stride_dKd,
    scale,
    INF_VAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute dK[b,h,:,d] = scale * sum_q(attn*(dA-r)*Q)."""
    pid_bh = tl.program_id(0)
    pid_n = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    q_bh_off = b * stride_qb + h * stride_qh
    k_bh_off = b * stride_kb + h * stride_kh
    v_bh_off = b * stride_vb + h * stride_vh
    dov_bh_off = b * stride_dOb + h * stride_dOh
    l_bh_off = b * stride_Lb + h * stride_Lh
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

        # r[m] = sum_k(attn[m,k]*dA[m,k]) -- need full reduction across k
        # For softmax backward: dS[m,n] = attn[m,n]*(dA[m,n] - r[m])
        # We need r[m] which requires iterating ALL k tiles. 
        # This per-query r is expensive to recompute for each n-tile.
        # BUT actually we can decompose dK differently:
        # dK[n,:] = scale * sum_m( attn[m,n]*dA[m,n]*Q[m,:] - attn[m,n]*r[m]*Q[m,:] )
        # The first part (without r) can accumulate here. The r part must be subtracted.
        
        # Let me compute everything correctly inline:
        # For this partial iteration, I need both the local contribution and eventually r[m].
        # Strategy: compute r first in a pre-pass, then accumulate dK.
        
        # Actually, let's use a simpler decomposition. For dK we need:
        # dK[n] += scale * sum_m( (attn*m_n * (dA_mn - r_m)) * Q_m )
        # We'll compute contributions piece-wise and track r during accumulation.
        
        # Re-think: the most efficient is to precompute r. But we can't allocate.
        # Alternative: accept the double work and just iterate twice - once for r, once for dK.
        # Or: use the fact that for a given (bh, ktile), we need all r[m] values.
        # Let me compute r array first, then dK.
        pass  # placeholder handled below
    
    # Better strategy: single pass that computes both r and dK contributions
    # Reset accumulator and redo
    dk_acc = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    
    # First, compute all r[m] values (needs full k-sweep)
    # Load per-query state for r computation
    r_val = tl.zeros((BLOCK_M,), dtype=tl.float32)  # Will be computed incrementally
    # Actually BLOCK_M changes per inner loop iteration; we need fixed-size arrays.
    # Let me restructure: use smaller blocks and compute r+dk in the same sweep.
    pass
    
    # SIMPLEST CORRECT APPROACH: split into two phases in one kernel.
    # Phase 1: compute r[m] for all m in this query batch
    # Phase 2: accumulate dK using r
    
    # Since BLOCK_M may have multiple tiles, let's compute r for all queries via a helper pattern.
    # Actually, for each (bh, n_tile) program, we iterate over m_tiles.
    # For each m_tile, we need r[m] for ALL m in [0,S). This is hard without precomputation.
    
    # CLEAN SOLUTION: compute r[m] inside the loop with a full-k scan per m_tile.
    # Then compute dK contribution. Both steps use the same loops.
    
    dk_acc = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    
    for step_m in range(num_m_steps):
        start_m = step_m * BLOCK_M
        offs_m = start_m + tl.arange(0, BLOCK_M)
        m_mask = offs_m < S
        
        q_ptrs = Q_ptr + q_bh_off + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
        raw_q = tl.load(q_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)
        Q_tile = raw_q.to(tl.float32)
        
        do_ptrs = dO_ptr + dov_bh_off + offs_m[:, None] * stride_dOs + offs_d[None, :] * stride_dOd
        raw_do = tl.load(do_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)
        dO_tile = raw_do.to(tl.float32)
        
        l_ptrs = L_ptr + l_bh_off + offs_m * stride_Ls
        raw_l = tl.load(l_ptrs, mask=m_mask, other=INF_VAL)
        L_tile = raw_l.to(tl.float32)
        
        # Compute r[m] by scanning all k tiles (BLOCK_N sized chunks covering S)
        r_m = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for sk in range(0, S, BLOCK_N):
            sn = sk + tl.arange(0, BLOCK_N)
            nk_mask = sn < S
            vk_valid = m_mask[:, None] & nk_mask[None, :]
            
            sk_ptrs = K_ptr + k_bh_off + sn[:, None] * stride_ks + offs_d[None, :] * stride_kd
            srk = tl.load(sk_ptrs, mask=nk_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
            sv_ptrs = V_ptr + v_bh_off + sn[:, None] * stride_vs + offs_d[None, :] * stride_vd
            srv = tl.load(sv_ptrs, mask=nk_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
            
            s_logits = tl.dot(Q_tile, srk) * scale
            s_Lsafe = tl.where(m_mask[:, None], L_tile[:, None], INF_VAL)
            s_attn = tl.exp(s_logits - s_Lsafe)
            s_attn = tl.where(vk_valid, s_attn, 0.0)
            
            s_dA = tl.dot(dO_tile, srv.T)
            s_aw = s_attn * s_dA
            r_m = r_m + tl.sum(s_aw, axis=-1)
        
        # Now accumulate dK contribution for THIS n_tile using r_m
        vn_valid = m_mask[:, None] & n_mask[None, :]
        vn_logits = tl.dot(Q_tile, K_tile) * scale
        vn_Lsafe = tl.where(m_mask[:, None], L_tile[:, None], INF_VAL)
        vn_attn = tl.exp(vn_logits - vn_Lsafe)
        vn_attn = tl.where(vn_valid, vn_attn, 0.0)
        
        vn_dA = tl.dot(dO_tile, V_tile.T)
        vn_dS = vn_attn * (vn_dA - r_m[:, None])
        
        dk_acc = tl.dot(vn_dS.T, Q_tile, dk_acc)
    
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

    # Kernel 1: dQ
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

    # Kernel 2: dV
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

    # Kernel 3: dK
    _mha_bwd_dk_kernel[(num_bh, num_ntiles)](
        Q, K, V, dO, L, dK,
        B, H, S, d_head,
        qs[0], qs[1], qs[2], qs[3],
        ks[0], ks[1], ks[2], ks[3],
        vs[0], vs[1], vs[2], vs[3],
        dos[0], dos[1], dos[2], dos[3],
        ls[0], ls[1], ls[2],
        dks[0], dks[1], dks[2], dks[3],
        scale,
        INF_VAL=INF_VAL,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=4, num_stages=2,
    )