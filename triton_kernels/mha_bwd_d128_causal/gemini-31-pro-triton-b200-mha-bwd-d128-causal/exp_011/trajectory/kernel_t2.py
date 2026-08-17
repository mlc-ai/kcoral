import torch
import triton
import triton.language as tl

# Define the descriptor allocator for device-side descriptor creation
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

@triton.jit
def _zero_dk_dv_kernel(
    dk_ptr, dv_ptr,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    S, H,
    BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid_s = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    batch_idx = pid_bh // H
    head_idx = pid_bh % H
    
    start_s = pid_s * BLOCK_S
    offs_s = start_s + tl.arange(0, BLOCK_S)
    offs_d = tl.arange(0, BLOCK_D)
    
    mask = offs_s < S
    
    dk = dk_ptr + batch_idx * stride_kb + head_idx * stride_kh + offs_s[:, None] * stride_ks + offs_d[None, :] * stride_kd
    dv = dv_ptr + batch_idx * stride_vb + head_idx * stride_vh + offs_s[:, None] * stride_vs + offs_d[None, :] * stride_vd
    
    tl.store(dk, 0.0, mask=mask[:, None])
    tl.store(dv, 0.0, mask=mask[:, None])

def get_autotune_configs():
    return [
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
    ]

@triton.autotune(
    configs=get_autotune_configs(),
    key=['S']
)
@triton.jit
def _bwd_kernel(
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
    S, sm_scale, H,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    # This kernel iterates over Q tiles (M) in the outer loop, loading each once.
    # It streams over corresponding K/V tiles (N). dK and dV components are atomically added.
    # Note: Atomic floating point additions can cause slight non-determinism in dK and dV outputs.
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    start_m = pid_m * BLOCK_M
    if start_m >= S:
        return

    batch_idx = pid_bh // H
    head_idx = pid_bh % H

    q_ptr = Q + batch_idx * stride_qb + head_idx * stride_qh
    k_ptr = K + batch_idx * stride_kb + head_idx * stride_kh
    v_ptr = V + batch_idx * stride_vb + head_idx * stride_vh
    o_ptr = O + batch_idx * stride_ob + head_idx * stride_oh
    do_ptr = dO + batch_idx * stride_dob + head_idx * stride_doh
    l_ptr = L + batch_idx * stride_lb + head_idx * stride_lh
    
    dq_ptr = dQ + batch_idx * stride_dqb + head_idx * stride_dqh
    dk_ptr = dK + batch_idx * stride_dkb + head_idx * stride_dkh
    dv_ptr = dV + batch_idx * stride_dvb + head_idx * stride_dvh

    q_desc = tl.make_tensor_descriptor(q_ptr, shape=[S, BLOCK_D], strides=[stride_qs, stride_qd], block_shape=[BLOCK_M, BLOCK_D], padding_option="zero")
    o_desc = tl.make_tensor_descriptor(o_ptr, shape=[S, BLOCK_D], strides=[stride_os, stride_od], block_shape=[BLOCK_M, BLOCK_D], padding_option="zero")
    do_desc = tl.make_tensor_descriptor(do_ptr, shape=[S, BLOCK_D], strides=[stride_dos, stride_dod], block_shape=[BLOCK_M, BLOCK_D], padding_option="zero")
    dq_desc = tl.make_tensor_descriptor(dq_ptr, shape=[S, BLOCK_D], strides=[stride_dqs, stride_dqd], block_shape=[BLOCK_M, BLOCK_D], padding_option="zero")
    
    k_desc = tl.make_tensor_descriptor(k_ptr, shape=[S, BLOCK_D], strides=[stride_ks, stride_kd], block_shape=[BLOCK_N, BLOCK_D], padding_option="zero")
    v_desc = tl.make_tensor_descriptor(v_ptr, shape=[S, BLOCK_D], strides=[stride_vs, stride_vd], block_shape=[BLOCK_N, BLOCK_D], padding_option="zero")

    q = q_desc.load([start_m, 0])
    o = o_desc.load([start_m, 0])
    do = do_desc.load([start_m, 0])

    offs_m = start_m + tl.arange(0, BLOCK_M)
    mask_m = offs_m < S
    l = tl.load(l_ptr + offs_m * stride_ls, mask=mask_m, other=0.0)

    delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
    dq_acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    max_n = tl.minimum(S, start_m + BLOCK_M)
    n_steps = tl.cdiv(max_n, BLOCK_N)
    
    offs_d = tl.arange(0, BLOCK_D)

    for n in range(n_steps):
        start_n = n * BLOCK_N
        
        k = k_desc.load([start_n, 0])
        v = v_desc.load([start_n, 0])

        qk = tl.dot(q, k.T) * sm_scale

        offs_n = start_n + tl.arange(0, BLOCK_N)
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        mask_n = offs_n < S
        valid_mask = mask_m[:, None] & mask_n[None, :] & causal_mask
        
        qk = tl.where(valid_mask, qk, float("-inf"))
        p = tl.math.exp2((qk - l[:, None]) * 1.4426950408889634)
        p = tl.where(valid_mask, p, 0.0)

        dp = tl.dot(do, v.T)

        ds = p * (dp - delta[:, None]) * sm_scale
        ds = tl.where(valid_mask, ds, 0.0)

        dq_acc = tl.dot(ds.to(q.dtype), k, acc=dq_acc)

        dk_partial = tl.dot(ds.T.to(q.dtype), q)
        dv_partial = tl.dot(p.T.to(q.dtype), do)

        dk_ptrs = dk_ptr + offs_n[:, None] * stride_dks + offs_d[None, :] * stride_dkd
        dv_ptrs = dv_ptr + offs_n[:, None] * stride_dvs + offs_d[None, :] * stride_dvd
        
        mask_2d = mask_n[:, None]
        tl.atomic_add(dk_ptrs, dk_partial.to(dk_ptr.dtype.element_ty), mask=mask_2d)
        tl.atomic_add(dv_ptrs, dv_partial.to(dv_ptr.dtype.element_ty), mask=mask_2d)

    dq_desc.store([start_m, 0], dq_acc.to(q.dtype))

def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes the backward pass for causal multi-head attention.
    Each result is written into the supplied output tensors without reallocation.
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, d = Q.shape
    sm_scale = 1.0 / (d ** 0.5)

    # Initialize accumulation tensors for dK and dV natively before computation
    grid_zero = (triton.cdiv(S, 128), B * H)
    _zero_dk_dv_kernel[grid_zero](
        dK, dV,
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        S, H,
        BLOCK_S=128, BLOCK_D=128
    )

    # Single-owner Q-tile mapping backward path
    grid = lambda META: (triton.cdiv(S, META['BLOCK_M']), B * H)
    _bwd_kernel[grid](
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
        S, sm_scale, H,
        BLOCK_D=128
    )