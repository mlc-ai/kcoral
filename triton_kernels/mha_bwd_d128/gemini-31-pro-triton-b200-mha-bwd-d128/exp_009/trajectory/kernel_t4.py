import math
import torch
import triton
import triton.language as tl


def _get_dq_configs():
    # Safe configs that stay strictly within 228KB shared memory limits for dq pass
    return [
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
    ]


@triton.autotune(configs=_get_dq_configs(), key=['S'])
@triton.jit
def _bwd_dq_kernel(
    Q, K, V, O, dO, L, dQ,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    S, scale, H,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    # Cast to int64 to avoid overflow in pointer arithmetic
    b_idx = tl.cast(pid_bh // H, tl.int64)
    h_idx = tl.cast(pid_bh % H, tl.int64)

    m_start = pid_m * BLOCK_M
    m_offs = m_start + tl.arange(0, BLOCK_M)
    m_mask = m_offs < S
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < BLOCK_D

    Q_base = Q + b_idx * tl.cast(stride_qb, tl.int64) + h_idx * tl.cast(stride_qh, tl.int64)
    K_base = K + b_idx * tl.cast(stride_kb, tl.int64) + h_idx * tl.cast(stride_kh, tl.int64)
    V_base = V + b_idx * tl.cast(stride_vb, tl.int64) + h_idx * tl.cast(stride_vh, tl.int64)
    O_base = O + b_idx * tl.cast(stride_ob, tl.int64) + h_idx * tl.cast(stride_oh, tl.int64)
    dO_base = dO + b_idx * tl.cast(stride_dob, tl.int64) + h_idx * tl.cast(stride_doh, tl.int64)
    L_base = L + b_idx * tl.cast(stride_lb, tl.int64) + h_idx * tl.cast(stride_lh, tl.int64)

    q_ptrs = Q_base + m_offs[:, None] * stride_qs + d_offs[None, :] * stride_qd
    o_ptrs = O_base + m_offs[:, None] * stride_os + d_offs[None, :] * stride_od
    do_ptrs = dO_base + m_offs[:, None] * stride_dos + d_offs[None, :] * stride_dod
    l_ptrs = L_base + m_offs * stride_ls

    q = tl.load(q_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)
    o = tl.load(o_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)
    do = tl.load(do_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)
    lse = tl.load(l_ptrs, mask=m_mask, other=0.0)

    # Compute rowwise delta: sum(O * dO, axis=-1)
    delta = tl.sum(tl.cast(o, tl.float32) * tl.cast(do, tl.float32), axis=1)

    dq = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    for n_start in range(0, S, BLOCK_N):
        n_offs = n_start + tl.arange(0, BLOCK_N)
        n_mask = n_offs < S

        k_ptrs = K_base + n_offs[:, None] * stride_ks + d_offs[None, :] * stride_kd
        v_ptrs = V_base + n_offs[:, None] * stride_vs + d_offs[None, :] * stride_vd

        k = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
        v = tl.load(v_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)

        # scores: [BLOCK_M, BLOCK_N]
        scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale
        scores = tl.where(m_mask[:, None] & n_mask[None, :], scores, float("-inf"))
        
        # LogSumExp processing (natural log)
        p = tl.exp(scores - lse[:, None])
        
        # dP: [BLOCK_M, BLOCK_N]
        dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32)
        
        # dS: [BLOCK_M, BLOCK_N]
        ds = p * (dp - delta[:, None]) * scale
        ds = tl.where(m_mask[:, None] & n_mask[None, :], ds, 0.0)

        # Accumulate fully into local dQ registers
        dq += tl.dot(tl.cast(ds, tl.bfloat16), k, out_dtype=tl.float32)

    dQ_base = dQ + b_idx * tl.cast(stride_dqb, tl.int64) + h_idx * tl.cast(stride_dqh, tl.int64)
    dq_ptrs = dQ_base + m_offs[:, None] * stride_dqs + d_offs[None, :] * stride_dqd
    tl.store(dq_ptrs, tl.cast(dq, tl.bfloat16), mask=m_mask[:, None] & d_mask[None, :])


def _get_dk_dv_configs():
    # Safe configs that stay strictly within 228KB shared memory limits for dk/dv pass
    return [
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
    ]


@triton.autotune(configs=_get_dk_dv_configs(), key=['S'])
@triton.jit
def _bwd_dk_dv_kernel(
    Q, K, V, O, dO, L, dK, dV,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    S, scale, H,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    b_idx = tl.cast(pid_bh // H, tl.int64)
    h_idx = tl.cast(pid_bh % H, tl.int64)

    n_start = pid_n * BLOCK_N
    n_offs = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_offs < S
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < BLOCK_D

    Q_base = Q + b_idx * tl.cast(stride_qb, tl.int64) + h_idx * tl.cast(stride_qh, tl.int64)
    K_base = K + b_idx * tl.cast(stride_kb, tl.int64) + h_idx * tl.cast(stride_kh, tl.int64)
    V_base = V + b_idx * tl.cast(stride_vb, tl.int64) + h_idx * tl.cast(stride_vh, tl.int64)
    O_base = O + b_idx * tl.cast(stride_ob, tl.int64) + h_idx * tl.cast(stride_oh, tl.int64)
    dO_base = dO + b_idx * tl.cast(stride_dob, tl.int64) + h_idx * tl.cast(stride_doh, tl.int64)
    L_base = L + b_idx * tl.cast(stride_lb, tl.int64) + h_idx * tl.cast(stride_lh, tl.int64)

    k_ptrs = K_base + n_offs[:, None] * stride_ks + d_offs[None, :] * stride_kd
    v_ptrs = V_base + n_offs[:, None] * stride_vs + d_offs[None, :] * stride_vd
    
    # Load K, V outside to prevent redundant requests
    k = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
    v = tl.load(v_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
    
    k_trans = tl.trans(k)
    v_trans = tl.trans(v)

    dk = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)

    for m_start in range(0, S, BLOCK_M):
        m_offs = m_start + tl.arange(0, BLOCK_M)
        m_mask = m_offs < S

        q_ptrs = Q_base + m_offs[:, None] * stride_qs + d_offs[None, :] * stride_qd
        o_ptrs = O_base + m_offs[:, None] * stride_os + d_offs[None, :] * stride_od
        do_ptrs = dO_base + m_offs[:, None] * stride_dos + d_offs[None, :] * stride_dod
        l_ptrs = L_base + m_offs * stride_ls

        q = tl.load(q_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)
        o = tl.load(o_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)
        do = tl.load(do_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)
        lse = tl.load(l_ptrs, mask=m_mask, other=0.0)

        # Recompute delta dynamically for each Q block step
        delta = tl.sum(tl.cast(o, tl.float32) * tl.cast(do, tl.float32), axis=1)

        scores = tl.dot(q, k_trans, out_dtype=tl.float32) * scale
        scores = tl.where(m_mask[:, None] & n_mask[None, :], scores, float("-inf"))
        
        p = tl.exp(scores - lse[:, None])
        
        dp = tl.dot(do, v_trans, out_dtype=tl.float32)
        ds = p * (dp - delta[:, None]) * scale
        ds = tl.where(m_mask[:, None] & n_mask[None, :], ds, 0.0)

        ds_trans = tl.trans(tl.cast(ds, tl.bfloat16))
        p_trans = tl.trans(tl.cast(p, tl.bfloat16))

        # Accumulate fully into local dK and dV registers
        dk += tl.dot(ds_trans, q, out_dtype=tl.float32)
        dv += tl.dot(p_trans, do, out_dtype=tl.float32)

    dK_base = dK + b_idx * tl.cast(stride_dkb, tl.int64) + h_idx * tl.cast(stride_dkh, tl.int64)
    dV_base = dV + b_idx * tl.cast(stride_dvb, tl.int64) + h_idx * tl.cast(stride_dvh, tl.int64)
    
    dk_ptrs = dK_base + n_offs[:, None] * stride_dks + d_offs[None, :] * stride_dkd
    dv_ptrs = dV_base + n_offs[:, None] * stride_dvs + d_offs[None, :] * stride_dvd

    # Finalize store for entire sequence chunk avoiding atomic congestion
    tl.store(dk_ptrs, tl.cast(dk, tl.bfloat16), mask=n_mask[:, None] & d_mask[None, :])
    tl.store(dv_ptrs, tl.cast(dv, tl.bfloat16), mask=n_mask[:, None] & d_mask[None, :])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes split-ownership standard flash attention backward without global atomics.
    The implementation streams dynamically evaluated row-wise deltas per block to strictly
    adhere to allocation conditions avoiding precomputed allocations. 
    """
    torch.cuda.set_device(Q.device)

    B, H, S, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    
    if S == 0:
        return

    # Pass 1: compute dQ locally without global reduction races
    grid_dq = lambda META: (triton.cdiv(S, META['BLOCK_M']), B * H)
    _bwd_dq_kernel[grid_dq](
        Q, K, V, O, dO, L, dQ,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        S, scale, H,
        BLOCK_D=128
    )

    # Pass 2: compute dK and dV locally leveraging reversed loop priorities
    grid_dk_dv = lambda META: (triton.cdiv(S, META['BLOCK_N']), B * H)
    _bwd_dk_dv_kernel[grid_dk_dv](
        Q, K, V, O, dO, L, dK, dV,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        S, scale, H,
        BLOCK_D=128
    )