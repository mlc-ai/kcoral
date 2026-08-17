import math
import torch
import triton
import triton.language as tl

# Set up Triton memory allocator to allow device-side descriptors
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)
triton.set_allocator(alloc_fn)

@triton.jit
def zero_tensor_kernel(
    ptr,
    stride_b, stride_h, stride_s, stride_d,
    B, H, S, D,
    BLOCK: tl.constexpr
):
    """Zero out arbitrary-layout tensors for safe atomic accumulation."""
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    total = B * H * S * D
    mask = idx < total
    
    d_idx = idx % D
    s_idx = (idx // D) % S
    h_idx = (idx // (D * S)) % H
    b_idx = idx // (D * S * H)
    
    offs = b_idx * stride_b + h_idx * stride_h + s_idx * stride_s + d_idx * stride_d
    tl.store(ptr + offs, 0.0, mask=mask)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
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

    q_ptr = Q + pid_b * stride_qb + pid_h * stride_qh
    k_ptr = K + pid_b * stride_kb + pid_h * stride_kh
    v_ptr = V + pid_b * stride_vb + pid_h * stride_vh
    o_ptr = O + pid_b * stride_ob + pid_h * stride_oh
    do_ptr = dO + pid_b * stride_dob + pid_h * stride_doh
    dq_ptr = dQ + pid_b * stride_dqb + pid_h * stride_dqh
    dk_ptr = dK + pid_b * stride_dkb + pid_h * stride_dkh
    dv_ptr = dV + pid_b * stride_dvb + pid_h * stride_dvh

    q_desc = tl.make_tensor_descriptor(
        q_ptr, shape=[S, BLOCK_D], strides=[stride_qs, stride_qd],
        block_shape=[BLOCK_M, BLOCK_D], padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        o_ptr, shape=[S, BLOCK_D], strides=[stride_os, stride_od],
        block_shape=[BLOCK_M, BLOCK_D], padding_option="zero"
    )
    do_desc = tl.make_tensor_descriptor(
        do_ptr, shape=[S, BLOCK_D], strides=[stride_dos, stride_dod],
        block_shape=[BLOCK_M, BLOCK_D], padding_option="zero"
    )
    
    k_desc = tl.make_tensor_descriptor(
        k_ptr, shape=[S, BLOCK_D], strides=[stride_ks, stride_kd],
        block_shape=[BLOCK_N, BLOCK_D], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        v_ptr, shape=[S, BLOCK_D], strides=[stride_vs, stride_vd],
        block_shape=[BLOCK_N, BLOCK_D], padding_option="zero"
    )

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < S
    
    q = tl.load(q_desc, [pid_m * BLOCK_M, 0])
    o = tl.load(o_desc, [pid_m * BLOCK_M, 0])
    do = tl.load(do_desc, [pid_m * BLOCK_M, 0])
    
    l_ptrs = L + pid_b * stride_lb + pid_h * stride_lh + offs_m * stride_ls
    l = tl.load(l_ptrs, mask=mask_m, other=0.0)

    # Precompute rowwise delta from O and dO
    delta = tl.sum(do.to(tl.float32) * o.to(tl.float32), axis=1)

    dq = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    offs_d = tl.arange(0, BLOCK_D)

    n_tiles = tl.cdiv(S, BLOCK_N)
    # Stagger starting tile to spread out atomic accumulation
    start_n_idx = pid_m % n_tiles

    for i in range(n_tiles):
        n0 = (start_n_idx + i) % n_tiles
        start_n = n0 * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S
        
        k = tl.load(k_desc, [start_n, 0])
        v = tl.load(v_desc, [start_n, 0])
        
        scores = tl.dot(q, k.T) * scale
        
        mask_mn = mask_m[:, None] & mask_n[None, :]
        scores = tl.where(mask_mn, scores, float("-inf"))
        
        # Natural-log LSE decoding
        p = tl.exp(scores - l[:, None])
        p = tl.where(mask_mn, p, 0.0)
        
        dp = tl.dot(do, v.T)
        
        ds = p * (dp - delta[:, None]) * scale
        ds = tl.where(mask_mn, ds, 0.0)
        
        dq += tl.dot(ds.to(q.dtype), k)
        
        dk_partial = tl.dot(ds.T.to(q.dtype), q)
        dv_partial = tl.dot(p.T.to(q.dtype), do)
        
        dk_partial = dk_partial.to(k.dtype)
        dv_partial = dv_partial.to(v.dtype)
        
        dk_ptrs = dk_ptr + offs_n[:, None] * stride_dks + offs_d[None, :] * stride_dkd
        dv_ptrs = dv_ptr + offs_n[:, None] * stride_dvs + offs_d[None, :] * stride_dvd
        
        mask_nd = mask_n[:, None] & (offs_d[None, :] < BLOCK_D)
        tl.atomic_add(dk_ptrs, dk_partial, mask=mask_nd, sem="relaxed")
        tl.atomic_add(dv_ptrs, dv_partial, mask=mask_nd, sem="relaxed")

    dq_desc = tl.make_tensor_descriptor(
        dq_ptr, shape=[S, BLOCK_D], strides=[stride_dqs, stride_dqd],
        block_shape=[BLOCK_M, BLOCK_D], padding_option="zero"
    )
    tl.store(dq_desc, [pid_m * BLOCK_M, 0], dq.to(q.dtype))

def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes SDPA backward pass using standard single-owner pattern.
    
    Warning: Output dK and dV gradients are accumulated concurrently via atomics,
    which introduces nondeterminism from floating point execution ordering.
    """
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    
    total_elements = B * H * S * d
    block_size = 1024
    grid_zero = (triton.cdiv(total_elements, block_size),)
    
    zero_tensor_kernel[grid_zero](
        dK,
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        B, H, S, d,
        BLOCK=block_size
    )
    zero_tensor_kernel[grid_zero](
        dV,
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        B, H, S, d,
        BLOCK=block_size
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
        BLOCK_D=d
    )