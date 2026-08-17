import math
import torch
import triton
import triton.language as tl


def _get_autotune_configs():
    return [
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
    ]


@triton.autotune(configs=_get_autotune_configs(), key=['S'])
@triton.jit
def _bwd_single_kernel(
    Q, K, V, O, dO, L, dQ, dK, dV,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    S, scale, H,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    b_idx = pid_bh // H
    h_idx = pid_bh % H

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    d_offs = tl.arange(0, BLOCK_D)

    m_mask = m_offs < S
    d_mask = d_offs < BLOCK_D

    q_ptrs = Q + b_idx * stride_qb + h_idx * stride_qh + m_offs[:, None] * stride_qs + d_offs[None, :] * stride_qd
    o_ptrs = O + b_idx * stride_ob + h_idx * stride_oh + m_offs[:, None] * stride_os + d_offs[None, :] * stride_od
    do_ptrs = dO + b_idx * stride_dob + h_idx * stride_doh + m_offs[:, None] * stride_dos + d_offs[None, :] * stride_dod
    l_ptrs = L + b_idx * stride_lb + h_idx * stride_lh + m_offs * stride_ls

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

        k_ptrs = K + b_idx * stride_kb + h_idx * stride_kh + n_offs[:, None] * stride_ks + d_offs[None, :] * stride_kd
        v_ptrs = V + b_idx * stride_vb + h_idx * stride_vh + n_offs[:, None] * stride_vs + d_offs[None, :] * stride_vd

        k = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
        v = tl.load(v_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)

        # scores: [BLOCK_M, BLOCK_N]
        scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale
        scores = tl.where(m_mask[:, None] & n_mask[None, :], scores, float("-inf"))

        # Natural log LSE formulation
        p = tl.exp(scores - lse[:, None])
        
        # dp: [BLOCK_M, BLOCK_N]
        dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32)

        # ds: [BLOCK_M, BLOCK_N]
        ds = p * (dp - delta[:, None]) * scale
        ds = tl.where(m_mask[:, None] & n_mask[None, :], ds, 0.0)

        ds_cast = tl.cast(ds, tl.bfloat16)
        p_cast = tl.cast(p, tl.bfloat16)

        # dQ local accumulation
        dq = tl.dot(ds_cast, k, dq)
        
        # dK, dV partial accumulation
        dk_partial = tl.dot(tl.trans(ds_cast), q, out_dtype=tl.float32)
        dv_partial = tl.dot(tl.trans(p_cast), do, out_dtype=tl.float32)

        dk_ptrs = dK + b_idx * stride_dkb + h_idx * stride_dkh + n_offs[:, None] * stride_dks + d_offs[None, :] * stride_dkd
        dv_ptrs = dV + b_idx * stride_dvb + h_idx * stride_dvh + n_offs[:, None] * stride_dvs + d_offs[None, :] * stride_dvd
        
        mask_kv = n_mask[:, None] & d_mask[None, :]
        
        tl.atomic_add(dk_ptrs, tl.cast(dk_partial, tl.bfloat16), mask=mask_kv)
        tl.atomic_add(dv_ptrs, tl.cast(dv_partial, tl.bfloat16), mask=mask_kv)

    # Store fully accumulated dQ tile
    dq_ptrs = dQ + b_idx * stride_dqb + h_idx * stride_dqh + m_offs[:, None] * stride_dqs + d_offs[None, :] * stride_dqd
    tl.store(dq_ptrs, tl.cast(dq, tl.bfloat16), mask=m_mask[:, None] & d_mask[None, :])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes single-owner MHA backward.
    The implementation writes results into preallocated dQ, dK, dV tensors using atomic
    adds for dK and dV as required when multiple queries write to the same KV tiles.
    """
    torch.cuda.set_device(Q.device)
    
    # Initialize global accumulators for atomics
    dK.zero_()
    dV.zero_()

    B, H, S, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    
    if S == 0:
        return

    grid = lambda META: (triton.cdiv(S, META['BLOCK_M']), B * H)
    _bwd_single_kernel[grid](
        Q, K, V, O, dO, L, dQ, dK, dV,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        S, scale, H,
        BLOCK_D=128
    )