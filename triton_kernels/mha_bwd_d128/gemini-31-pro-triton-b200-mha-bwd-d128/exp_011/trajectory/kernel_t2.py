import math
import torch
import triton
import triton.language as tl

@triton.jit
def zero_row_kernel(
    ptr, 
    stride_b, stride_h, stride_s, stride_d, 
    S, 
    D: tl.constexpr, BLOCK_S: tl.constexpr
):
    """Efficiently zero out the accumulated gradient tensors."""
    pid_s = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)
    
    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = offs_s < S
    
    offs_d = tl.arange(0, D)
    
    offs = pid_b * stride_b + pid_h * stride_h + offs_s[:, None] * stride_s + offs_d[None, :] * stride_d
    tl.store(ptr + offs, 0.0, mask=mask_s[:, None])


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
    ],
    key=['S'],
)
@triton.jit
def bwd_kernel_atomic(
    Q, K, V, O, dO, L, dQ, dK, dV,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    stride_lb, stride_lh, stride_ls,
    scale, S, H,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    
    q_ptrs = Q + pid_b * stride_qb + pid_h * stride_qh + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    o_ptrs = O + pid_b * stride_ob + pid_h * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    do_ptrs = dO + pid_b * stride_dob + pid_h * stride_doh + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod
    l_ptrs = L + pid_b * stride_lb + pid_h * stride_lh + offs_m * stride_ls

    mask_m = offs_m < S
    mask_md = mask_m[:, None]
    
    q = tl.load(q_ptrs, mask=mask_md, other=0.0)
    o = tl.load(o_ptrs, mask=mask_md, other=0.0)
    do = tl.load(do_ptrs, mask=mask_md, other=0.0)
    l = tl.load(l_ptrs, mask=mask_m, other=0.0)

    # Precompute rowwise delta from O and dO (matching standard attention backward logic)
    delta = tl.sum(do.to(tl.float32) * o.to(tl.float32), axis=1)

    dq = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    n_tiles = tl.cdiv(S, BLOCK_N)

    k_base = K + pid_b * stride_kb + pid_h * stride_kh
    v_base = V + pid_b * stride_vb + pid_h * stride_vh
    dk_base = dK + pid_b * stride_dkb + pid_h * stride_dkh
    dv_base = dV + pid_b * stride_dvb + pid_h * stride_dvh

    # Set up pointers for the first KV tile
    k_ptrs = k_base + tl.arange(0, BLOCK_N)[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = v_base + tl.arange(0, BLOCK_N)[:, None] * stride_vs + offs_d[None, :] * stride_vd
    dk_ptrs = dk_base + tl.arange(0, BLOCK_N)[:, None] * stride_dks + offs_d[None, :] * stride_dkd
    dv_ptrs = dv_base + tl.arange(0, BLOCK_N)[:, None] * stride_dvs + offs_d[None, :] * stride_dvd

    for n0 in range(n_tiles):
        offs_n = n0 * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S
        mask_nd = mask_n[:, None]
        
        k = tl.load(k_ptrs, mask=mask_nd, other=0.0)
        v = tl.load(v_ptrs, mask=mask_nd, other=0.0)
        
        # QK^T
        scores = tl.dot(q, k.T) * scale
        
        mask_mn = mask_m[:, None] & mask_n[None, :]
        scores = tl.where(mask_mn, scores, float("-inf"))
        
        # Natural-log LSE decoding
        p = tl.exp(scores - l[:, None])
        p = tl.where(mask_mn, p, 0.0)
        
        # dO V^T
        dp = tl.dot(do, v.T)
        
        # Gradient of softmax scores
        ds = p * (dp - delta[:, None]) * scale
        ds = tl.where(mask_mn, ds, 0.0)
        
        # Accumulate dQ locally
        dq += tl.dot(ds.to(q.dtype), k)
        
        # Compute partial dK and dV
        dk_partial = tl.dot(ds.T.to(q.dtype), q)
        dv_partial = tl.dot(p.T.to(q.dtype), do)
        
        # Atomically accumulate dK and dV globally
        tl.atomic_add(dk_ptrs, dk_partial.to(q.dtype), mask=mask_nd, sem="relaxed")
        tl.atomic_add(dv_ptrs, dv_partial.to(q.dtype), mask=mask_nd, sem="relaxed")
        
        # Advance pointers
        k_ptrs += BLOCK_N * stride_ks
        v_ptrs += BLOCK_N * stride_vs
        dk_ptrs += BLOCK_N * stride_dks
        dv_ptrs += BLOCK_N * stride_dvs

    # Store finalized dQ
    dq_ptrs = dQ + pid_b * stride_dqb + pid_h * stride_dqh + offs_m[:, None] * stride_dqs + offs_d[None, :] * stride_dqd
    tl.store(dq_ptrs, dq.to(q.dtype), mask=mask_md)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes SDPA backward pass using standard single-owner pattern.
    
    Output dK and dV gradients are accumulated concurrently via atomics,
    which introduces nondeterminism from floating point execution ordering.
    """
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    
    BLOCK_S = 128
    grid_zero = (triton.cdiv(S, BLOCK_S), H, B)
    zero_row_kernel[grid_zero](
        dK, dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        S, D=128, BLOCK_S=BLOCK_S
    )
    zero_row_kernel[grid_zero](
        dV, dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        S, D=128, BLOCK_S=BLOCK_S
    )

    grid_bwd = lambda META: (triton.cdiv(S, META['BLOCK_M']), B * H)
    bwd_kernel_atomic[grid_bwd](
        Q, K, V, O, dO, L, dQ, dK, dV,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        scale, S, H,
        BLOCK_D=128
    )