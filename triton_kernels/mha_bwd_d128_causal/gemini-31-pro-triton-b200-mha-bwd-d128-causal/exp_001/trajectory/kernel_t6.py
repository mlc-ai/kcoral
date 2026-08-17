import torch
import triton
import triton.language as tl

# Triton descriptor allocator for device-side TMA
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'WARP_SPEC': False}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'WARP_SPEC': True}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'WARP_SPEC': False}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'WARP_SPEC': False}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'WARP_SPEC': False}, num_warps=4, num_stages=3),
    ],
    key=['S']
)
@triton.jit
def bwd_dq_kernel(
    Q, K, V, O, dO, L, dQ,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    S,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, d: tl.constexpr, WARP_SPEC: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)

    start_m = pid_m * BLOCK_M

    q_base = Q + pid_b * stride_qb + pid_h * stride_qh
    q_desc = tl.make_tensor_descriptor(
        q_base, shape=[S, d], strides=[stride_qs, stride_qd],
        block_shape=[BLOCK_M, d], padding_option="zero"
    )
    q = q_desc.load([start_m, 0])

    o_base = O + pid_b * stride_ob + pid_h * stride_oh
    o_desc = tl.make_tensor_descriptor(
        o_base, shape=[S, d], strides=[stride_os, stride_od],
        block_shape=[BLOCK_M, d], padding_option="zero"
    )
    o = o_desc.load([start_m, 0])

    do_base = dO + pid_b * stride_dob + pid_h * stride_doh
    do_desc = tl.make_tensor_descriptor(
        do_base, shape=[S, d], strides=[stride_dos, stride_dod],
        block_shape=[BLOCK_M, d], padding_option="zero"
    )
    do = do_desc.load([start_m, 0])

    offs_m = start_m + tl.arange(0, BLOCK_M)
    l_ptrs = L + pid_b * stride_lb + pid_h * stride_lh + offs_m * stride_ls
    l = tl.load(l_ptrs, mask=offs_m < S, other=0.0)

    Di = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
    dq = tl.zeros((BLOCK_M, d), dtype=tl.float32)
    scale = 0.08838834764831845  # 1.0 / sqrt(128)

    max_n = start_m + BLOCK_M
    if max_n > S:
        max_n = S

    limit_n = (start_m // BLOCK_N) * BLOCK_N
    if limit_n > max_n:
        limit_n = max_n

    k_base = K + pid_b * stride_kb + pid_h * stride_kh
    k_desc = tl.make_tensor_descriptor(
        k_base, shape=[S, d], strides=[stride_ks, stride_kd],
        block_shape=[BLOCK_N, d], padding_option="zero"
    )
    
    v_base = V + pid_b * stride_vb + pid_h * stride_vh
    v_desc = tl.make_tensor_descriptor(
        v_base, shape=[S, d], strides=[stride_vs, stride_vd],
        block_shape=[BLOCK_N, d], padding_option="zero"
    )

    # 1. Off-diagonal K/V blocks
    for start_n in tl.range(0, limit_n, BLOCK_N, num_stages=3, warp_specialize=WARP_SPEC):
        k = k_desc.load([start_n, 0])
        v = v_desc.load([start_n, 0])

        s = tl.dot(q, k.T, out_dtype=tl.float32) * scale
        
        mask = (offs_m[:, None] < S) & ((start_n + tl.arange(0, BLOCK_N))[None, :] < S)
        s = tl.where(mask, s, float('-inf'))
        p = tl.exp(s - l[:, None])

        ds = tl.dot(do, v.T, out_dtype=tl.float32)
        dp = p * (ds - Di[:, None]) * scale

        dq += tl.dot(dp.to(q.dtype), k, out_dtype=tl.float32)

    # 2. Diagonal K/V blocks
    for start_n in range(limit_n, max_n, BLOCK_N):
        k = k_desc.load([start_n, 0])
        v = v_desc.load([start_n, 0])

        s = tl.dot(q, k.T, out_dtype=tl.float32) * scale
        
        causal_mask = offs_m[:, None] >= (start_n + tl.arange(0, BLOCK_N))[None, :]
        mask = causal_mask & (offs_m[:, None] < S) & ((start_n + tl.arange(0, BLOCK_N))[None, :] < S)
        s = tl.where(mask, s, float('-inf'))

        p = tl.exp(s - l[:, None])

        ds = tl.dot(do, v.T, out_dtype=tl.float32)
        dp = p * (ds - Di[:, None]) * scale

        dq += tl.dot(dp.to(q.dtype), k, out_dtype=tl.float32)

    dq_base = dQ + pid_b * stride_dqb + pid_h * stride_dqh
    dq_desc = tl.make_tensor_descriptor(
        dq_base, shape=[S, d], strides=[stride_dqs, stride_dqd],
        block_shape=[BLOCK_M, d], padding_option="zero"
    )
    dq_desc.store([start_m, 0], dq.to(q.dtype))


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'WARP_SPEC': False}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'WARP_SPEC': True}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'WARP_SPEC': False}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'WARP_SPEC': False}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'WARP_SPEC': False}, num_warps=4, num_stages=3),
    ],
    key=['S']
)
@triton.jit
def bwd_dk_dv_kernel(
    Q, K, V, O, dO, L, dK, dV,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    S,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, d: tl.constexpr, WARP_SPEC: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)

    start_n = pid_n * BLOCK_N
    
    k_base = K + pid_b * stride_kb + pid_h * stride_kh
    k_desc = tl.make_tensor_descriptor(
        k_base, shape=[S, d], strides=[stride_ks, stride_kd],
        block_shape=[BLOCK_N, d], padding_option="zero"
    )
    k = k_desc.load([start_n, 0])

    v_base = V + pid_b * stride_vb + pid_h * stride_vh
    v_desc = tl.make_tensor_descriptor(
        v_base, shape=[S, d], strides=[stride_vs, stride_vd],
        block_shape=[BLOCK_N, d], padding_option="zero"
    )
    v = v_desc.load([start_n, 0])

    dk = tl.zeros((BLOCK_N, d), dtype=tl.float32)
    dv = tl.zeros((BLOCK_N, d), dtype=tl.float32)
    scale = 0.08838834764831845  # 1.0 / sqrt(128)

    start_m_block = (start_n // BLOCK_M) * BLOCK_M
    if start_m_block < 0:
        start_m_block = 0
        
    end_m_block = (S + BLOCK_M - 1) // BLOCK_M * BLOCK_M
    limit_m = ((start_n + BLOCK_N + BLOCK_M - 1) // BLOCK_M) * BLOCK_M
    if limit_m > end_m_block:
        limit_m = end_m_block

    q_base = Q + pid_b * stride_qb + pid_h * stride_qh
    q_desc = tl.make_tensor_descriptor(
        q_base, shape=[S, d], strides=[stride_qs, stride_qd],
        block_shape=[BLOCK_M, d], padding_option="zero"
    )
    
    o_base = O + pid_b * stride_ob + pid_h * stride_oh
    o_desc = tl.make_tensor_descriptor(
        o_base, shape=[S, d], strides=[stride_os, stride_od],
        block_shape=[BLOCK_M, d], padding_option="zero"
    )
    
    do_base = dO + pid_b * stride_dob + pid_h * stride_doh
    do_desc = tl.make_tensor_descriptor(
        do_base, shape=[S, d], strides=[stride_dos, stride_dod],
        block_shape=[BLOCK_M, d], padding_option="zero"
    )

    offs_n = start_n + tl.arange(0, BLOCK_N)

    # 1. Diagonal Q blocks
    for curr_m in range(start_m_block, limit_m, BLOCK_M):
        q = q_desc.load([curr_m, 0])
        o = o_desc.load([curr_m, 0])
        do = do_desc.load([curr_m, 0])
        
        offs_m = curr_m + tl.arange(0, BLOCK_M)
        l_ptrs = L + pid_b * stride_lb + pid_h * stride_lh + offs_m * stride_ls
        l_val = tl.load(l_ptrs, mask=offs_m < S, other=0.0)

        Di = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)

        s_T = tl.dot(k, q.T, out_dtype=tl.float32) * scale
        
        causal_mask = offs_m[None, :] >= offs_n[:, None]
        mask = causal_mask & (offs_n[:, None] < S) & (offs_m[None, :] < S)
        s_T = tl.where(mask, s_T, float('-inf'))

        p_T = tl.exp(s_T - l_val[None, :])

        dv += tl.dot(p_T.to(v.dtype), do, out_dtype=tl.float32)

        ds_T = tl.dot(v, do.T, out_dtype=tl.float32)
        dp_T = p_T * (ds_T - Di[None, :]) * scale

        dk += tl.dot(dp_T.to(k.dtype), q, out_dtype=tl.float32)

    # 2. Off-diagonal Q blocks
    for curr_m in tl.range(limit_m, end_m_block, BLOCK_M, num_stages=3, warp_specialize=WARP_SPEC):
        q = q_desc.load([curr_m, 0])
        o = o_desc.load([curr_m, 0])
        do = do_desc.load([curr_m, 0])
        
        offs_m = curr_m + tl.arange(0, BLOCK_M)
        l_ptrs = L + pid_b * stride_lb + pid_h * stride_lh + offs_m * stride_ls
        l_val = tl.load(l_ptrs, mask=offs_m < S, other=0.0)

        Di = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)

        s_T = tl.dot(k, q.T, out_dtype=tl.float32) * scale
        
        mask = (offs_n[:, None] < S) & (offs_m[None, :] < S)
        s_T = tl.where(mask, s_T, float('-inf'))

        p_T = tl.exp(s_T - l_val[None, :])

        dv += tl.dot(p_T.to(v.dtype), do, out_dtype=tl.float32)

        ds_T = tl.dot(v, do.T, out_dtype=tl.float32)
        dp_T = p_T * (ds_T - Di[None, :]) * scale

        dk += tl.dot(dp_T.to(k.dtype), q, out_dtype=tl.float32)

    dk_base = dK + pid_b * stride_dkb + pid_h * stride_dkh
    dk_desc = tl.make_tensor_descriptor(
        dk_base, shape=[S, d], strides=[stride_dks, stride_dkd],
        block_shape=[BLOCK_N, d], padding_option="zero"
    )
    dk_desc.store([start_n, 0], dk.to(k.dtype))

    dv_base = dV + pid_b * stride_dvb + pid_h * stride_dvh
    dv_desc = tl.make_tensor_descriptor(
        dv_base, shape=[S, d], strides=[stride_dvs, stride_dvd],
        block_shape=[BLOCK_N, d], padding_option="zero"
    )
    dv_desc.store([start_n, 0], dv.to(v.dtype))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes causal multi-head attention backward pass natively on NVIDIA Blackwell.
    Results are written into preallocated dQ, dK, dV output buffers using TMA natively.
    """
    with torch.cuda.device(Q.device):
        B, H, S, d = Q.shape
        lse = L.view(B, H, S)
        
        grid_dq = lambda META: (triton.cdiv(S, META['BLOCK_M']), H, B)
        bwd_dq_kernel[grid_dq](
            Q, K, V, O, dO, lse, dQ,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
            S, d=d
        )

        grid_dk_dv = lambda META: (triton.cdiv(S, META['BLOCK_N']), H, B)
        bwd_dk_dv_kernel[grid_dk_dv](
            Q, K, V, O, dO, lse, dK, dV,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
            dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
            S, d=d
        )
        
        return dQ, dK, dV