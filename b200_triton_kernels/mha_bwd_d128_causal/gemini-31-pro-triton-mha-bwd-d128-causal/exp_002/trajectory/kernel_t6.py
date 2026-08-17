import math
import torch
import triton
import triton.language as tl

# Provide Triton with a host-side allocation hook for device-created TMA descriptor structures.
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

# Configurations optimize for Hopper WGMMA footprint.
# Bounded heavily by num_stages to prevent exceeding the 228KB SM90 shared memory budget.
configs = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
]


@triton.autotune(configs=configs, key=['S'])
@triton.jit
def _bwd_kernel_single(
    Q_ptr, K_ptr, V_ptr, O_ptr, DO_ptr, DQ_ptr, DK_ptr, DV_ptr, L_ptr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    stride_lb, stride_lh, stride_ls,
    H, S, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D_HEAD: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    m_start = pid_m * BLOCK_M
    if m_start >= S:
        return
        
    b = pid_bh // H
    h = pid_bh % H
    
    off_m = m_start + tl.arange(0, BLOCK_M)
    off_n_base = tl.arange(0, BLOCK_N)
    off_d = tl.arange(0, D_HEAD)
    
    # Base pointers
    q_base = Q_ptr + b * stride_qb + h * stride_qh
    do_base = DO_ptr + b * stride_dob + h * stride_doh
    o_base = O_ptr + b * stride_ob + h * stride_oh
    dq_base = DQ_ptr + b * stride_dqb + h * stride_dqh
    k_base = K_ptr + b * stride_kb + h * stride_kh
    v_base = V_ptr + b * stride_vb + h * stride_vh
    dk_base = DK_ptr + b * stride_dkb + h * stride_dkh
    dv_base = DV_ptr + b * stride_dvb + h * stride_dvh
    l_base = L_ptr + b * stride_lb + h * stride_lh

    # TMA descriptors for M-dimension (Q, O, dO, dQ)
    q_desc = tl.make_tensor_descriptor(q_base, shape=[S, D_HEAD], strides=[stride_qs, stride_qd], block_shape=[BLOCK_M, D_HEAD], padding_option="zero")
    do_desc = tl.make_tensor_descriptor(do_base, shape=[S, D_HEAD], strides=[stride_dos, stride_dod], block_shape=[BLOCK_M, D_HEAD], padding_option="zero")
    o_desc = tl.make_tensor_descriptor(o_base, shape=[S, D_HEAD], strides=[stride_os, stride_od], block_shape=[BLOCK_M, D_HEAD], padding_option="zero")
    dq_desc = tl.make_tensor_descriptor(dq_base, shape=[S, D_HEAD], strides=[stride_dqs, stride_dqd], block_shape=[BLOCK_M, D_HEAD])

    # TMA descriptors for N-dimension (K, V)
    k_desc = tl.make_tensor_descriptor(k_base, shape=[S, D_HEAD], strides=[stride_ks, stride_kd], block_shape=[BLOCK_N, D_HEAD], padding_option="zero")
    v_desc = tl.make_tensor_descriptor(v_base, shape=[S, D_HEAD], strides=[stride_vs, stride_vd], block_shape=[BLOCK_N, D_HEAD], padding_option="zero")

    # Load M-dimension invariants
    q = q_desc.load([m_start, 0])
    do = do_desc.load([m_start, 0])
    o = o_desc.load([m_start, 0])
    l_m = tl.load(l_base + off_m * stride_ls, mask=off_m < S, other=0.0)

    delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
    dq = tl.zeros([BLOCK_M, D_HEAD], dtype=tl.float32)

    num_n_unmasked = tl.minimum(m_start // BLOCK_N, (S + BLOCK_N - 1) // BLOCK_N)
    num_n_total = tl.minimum((m_start + BLOCK_M + BLOCK_N - 1) // BLOCK_N, (S + BLOCK_N - 1) // BLOCK_N)

    # 1. Fully unmasked inner loop sequence. Causal conditions and masking checks are safely bypassed.
    for n_idx in tl.range(0, num_n_unmasked):
        n_start = n_idx * BLOCK_N
        off_n = n_start + off_n_base
        
        k = k_desc.load([n_start, 0])
        v = v_desc.load([n_start, 0])
        
        qk = tl.dot(q, k.T, out_dtype=tl.float32) * scale
        p = tl.math.exp(qk - l_m[:, None])
        
        dp = tl.dot(do, v.T, out_dtype=tl.float32)
        ds = p * (dp - delta[:, None]) * scale
        
        ds_bf16 = ds.to(tl.bfloat16)
        dq += tl.dot(ds_bf16, k, out_dtype=tl.float32)
        
        dK_n = tl.dot(ds_bf16.T, q, out_dtype=tl.float32)
        dV_n = tl.dot(p.to(tl.bfloat16).T, do, out_dtype=tl.float32)
        
        # Safely pointer-atomic increment to cross-boundary dK and dV.
        dk_ptrs = dk_base + off_n[:, None] * stride_dks + off_d[None, :] * stride_dkd
        dv_ptrs = dv_base + off_n[:, None] * stride_dvs + off_d[None, :] * stride_dvd
        
        tl.atomic_add(dk_ptrs, dK_n.to(tl.bfloat16), sem="relaxed")
        tl.atomic_add(dv_ptrs, dV_n.to(tl.bfloat16), sem="relaxed")

    # 2. Masked diagonal block loop containing boundaries for causality limits checking.
    for n_idx in range(num_n_unmasked, num_n_total):
        n_start = n_idx * BLOCK_N
        off_n = n_start + off_n_base
        
        k = k_desc.load([n_start, 0])
        v = v_desc.load([n_start, 0])
        
        qk = tl.dot(q, k.T, out_dtype=tl.float32) * scale
        
        mask_m = off_m < S
        mask_n = off_n < S
        valid = (off_m[:, None] >= off_n[None, :]) & mask_m[:, None] & mask_n[None, :]
        qk = tl.where(valid, qk, float('-inf'))
        
        p = tl.math.exp(qk - l_m[:, None])
        p = tl.where(valid, p, 0.0)
        
        dp = tl.dot(do, v.T, out_dtype=tl.float32)
        ds = p * (dp - delta[:, None]) * scale
        
        ds_bf16 = ds.to(tl.bfloat16)
        dq += tl.dot(ds_bf16, k, out_dtype=tl.float32)
        
        dK_n = tl.dot(ds_bf16.T, q, out_dtype=tl.float32)
        dV_n = tl.dot(p.to(tl.bfloat16).T, do, out_dtype=tl.float32)
        
        dk_ptrs = dk_base + off_n[:, None] * stride_dks + off_d[None, :] * stride_dkd
        dv_ptrs = dv_base + off_n[:, None] * stride_dvs + off_d[None, :] * stride_dvd
        
        # Strict memory-bounds constraints mask
        mask_d = tl.full([D_HEAD], True, dtype=tl.int1)
        mask_nd = mask_n[:, None] & mask_d[None, :]
        
        tl.atomic_add(dk_ptrs, dK_n.to(tl.bfloat16), mask=mask_nd, sem="relaxed")
        tl.atomic_add(dv_ptrs, dV_n.to(tl.bfloat16), mask=mask_nd, sem="relaxed")

    dq_desc.store([m_start, 0], dq.to(tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes backward pass for Causal Multi-Head Attention targeting Hopper TMA optimizations.
    Uses asynchronous Hopper native atomic_adds to accumulate dK/dV smoothly without intermediate locks.
    """
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    scale = 1.0 / math.sqrt(D)
    
    # Initialize targets directly on hardware queue for accumulation
    dK.zero_()
    dV.zero_()
    
    stride_lb, stride_lh, stride_ls = L.stride(0), L.stride(1), L.stride(2)
        
    grid = lambda META: (
        triton.cdiv(S, META['BLOCK_M']),
        B * H
    )

    _bwd_kernel_single[grid](
        Q, K, V, O, dO, dQ, dK, dV, L,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        stride_lb, stride_lh, stride_ls,
        H, S, scale,
        D_HEAD=D,
    )