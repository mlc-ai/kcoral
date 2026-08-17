import math
import torch
import triton
import triton.language as tl

def get_dq_configs():
    return [
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=5),
    ]

def get_dkdv_configs():
    return [
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=5),
    ]

@triton.autotune(configs=get_dkdv_configs(), key=['S'])
@triton.jit
def bwd_kernel_dk_dv(
    Q, K, V, O, dO, L, dK, dV,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    B, H, S, sm_scale,
    d: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    start_n = pid_n * BLOCK_N
    if start_n >= S:
        return

    off_q = pid_b * stride_qb + pid_h * stride_qh
    off_k = pid_b * stride_kb + pid_h * stride_kh
    off_v = pid_b * stride_vb + pid_h * stride_vh
    off_o = pid_b * stride_ob + pid_h * stride_oh
    off_do = pid_b * stride_dob + pid_h * stride_doh
    off_dk = pid_b * stride_dkb + pid_h * stride_dkh
    off_dv = pid_b * stride_dvb + pid_h * stride_dvh
    off_l = pid_b * stride_lb + pid_h * stride_lh

    k_desc = tl.make_tensor_descriptor(
        K + off_k, shape=[S, d], strides=[stride_ks, 1],
        block_shape=[BLOCK_N, d], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        V + off_v, shape=[S, d], strides=[stride_vs, 1],
        block_shape=[BLOCK_N, d], padding_option="zero"
    )
    k_j = k_desc.load([start_n, 0])
    v_j = v_desc.load([start_n, 0])

    dk_acc = tl.zeros((BLOCK_N, d), dtype=tl.float32)
    dv_acc = tl.zeros((BLOCK_N, d), dtype=tl.float32)

    start_m_initial = (start_n // BLOCK_M) * BLOCK_M
    if start_m_initial < S:
        num_steps = tl.cdiv(S - start_m_initial, BLOCK_M)
    else:
        num_steps = 0

    q_desc = tl.make_tensor_descriptor(
        Q + off_q, shape=[S, d], strides=[stride_qs, 1],
        block_shape=[BLOCK_M, d], padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        O + off_o, shape=[S, d], strides=[stride_os, 1],
        block_shape=[BLOCK_M, d], padding_option="zero"
    )
    do_desc = tl.make_tensor_descriptor(
        dO + off_do, shape=[S, d], strides=[stride_dos, 1],
        block_shape=[BLOCK_M, d], padding_option="zero"
    )

    offs_n = start_n + tl.arange(0, BLOCK_N)

    for step in range(num_steps):
        start_m = start_m_initial + step * BLOCK_M
        
        q_i = q_desc.load([start_m, 0])
        o_i = o_desc.load([start_m, 0])
        do_i = do_desc.load([start_m, 0])

        offs_m = start_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S
        
        l_ptrs = L + off_l + offs_m * stride_ls
        l_i = tl.load(l_ptrs, mask=mask_m, other=0.0)

        # Compute rowsum(dO_i * O_i) on the fly
        d_i = tl.sum(do_i.to(tl.float32) * o_i.to(tl.float32), axis=1)

        s_ij = tl.dot(q_i, tl.trans(k_j), out_dtype=tl.float32) * sm_scale

        valid = (offs_m[:, None] < S) & (offs_n[None, :] < S)
        if start_m < start_n + BLOCK_N:
            valid = valid & (offs_m[:, None] >= offs_n[None, :])
        
        s_ij = tl.where(valid, s_ij, float('-inf'))
        p_ij = tl.exp(s_ij - l_i[:, None])

        dp_ij = tl.dot(do_i, tl.trans(v_j), out_dtype=tl.float32)
        ds_ij = p_ij * (dp_ij - d_i[:, None]) * sm_scale

        dv_acc = tl.dot(tl.trans(p_ij.to(tl.bfloat16)), do_i, acc=dv_acc, out_dtype=tl.float32)
        dk_acc = tl.dot(tl.trans(ds_ij.to(tl.bfloat16)), q_i, acc=dk_acc, out_dtype=tl.float32)

    dk_desc = tl.make_tensor_descriptor(
        dK + off_dk, shape=[S, d], strides=[stride_dks, 1],
        block_shape=[BLOCK_N, d]
    )
    dv_desc = tl.make_tensor_descriptor(
        dV + off_dv, shape=[S, d], strides=[stride_dvs, 1],
        block_shape=[BLOCK_N, d]
    )

    dk_desc.store([start_n, 0], dk_acc.to(dK.dtype.element_ty))
    dv_desc.store([start_n, 0], dv_acc.to(dV.dtype.element_ty))


@triton.autotune(configs=get_dq_configs(), key=['S'])
@triton.jit
def bwd_kernel_dq(
    Q, K, V, O, dO, L, dQ,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    B, H, S, sm_scale,
    d: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    start_m = pid_m * BLOCK_M
    if start_m >= S:
        return

    off_q = pid_b * stride_qb + pid_h * stride_qh
    off_k = pid_b * stride_kb + pid_h * stride_kh
    off_v = pid_b * stride_vb + pid_h * stride_vh
    off_o = pid_b * stride_ob + pid_h * stride_oh
    off_do = pid_b * stride_dob + pid_h * stride_doh
    off_dq = pid_b * stride_dqb + pid_h * stride_dqh
    off_l = pid_b * stride_lb + pid_h * stride_lh

    q_desc = tl.make_tensor_descriptor(
        Q + off_q, shape=[S, d], strides=[stride_qs, 1],
        block_shape=[BLOCK_M, d], padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        O + off_o, shape=[S, d], strides=[stride_os, 1],
        block_shape=[BLOCK_M, d], padding_option="zero"
    )
    do_desc = tl.make_tensor_descriptor(
        dO + off_do, shape=[S, d], strides=[stride_dos, 1],
        block_shape=[BLOCK_M, d], padding_option="zero"
    )

    q_i = q_desc.load([start_m, 0])
    o_i = o_desc.load([start_m, 0])
    do_i = do_desc.load([start_m, 0])

    offs_m = start_m + tl.arange(0, BLOCK_M)
    mask_m = offs_m < S
    
    l_ptrs = L + off_l + offs_m * stride_ls
    l_i = tl.load(l_ptrs, mask=mask_m, other=0.0)

    # Compute rowsum(dO_i * O_i) on the fly
    d_i = tl.sum(do_i.to(tl.float32) * o_i.to(tl.float32), axis=1)

    k_desc = tl.make_tensor_descriptor(
        K + off_k, shape=[S, d], strides=[stride_ks, 1],
        block_shape=[BLOCK_N, d], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        V + off_v, shape=[S, d], strides=[stride_vs, 1],
        block_shape=[BLOCK_N, d], padding_option="zero"
    )

    dq_acc = tl.zeros((BLOCK_M, d), dtype=tl.float32)

    max_n = tl.minimum(S, start_m + BLOCK_M)
    num_steps = tl.cdiv(max_n, BLOCK_N)

    for step in range(num_steps):
        start_n = step * BLOCK_N
        
        k_j = k_desc.load([start_n, 0])
        v_j = v_desc.load([start_n, 0])

        s_ij = tl.dot(q_i, tl.trans(k_j), out_dtype=tl.float32) * sm_scale

        offs_n = start_n + tl.arange(0, BLOCK_N)
        valid = (offs_m[:, None] < S) & (offs_n[None, :] < S)
        
        if start_n + BLOCK_N > start_m:
            valid = valid & (offs_m[:, None] >= offs_n[None, :])
            
        s_ij = tl.where(valid, s_ij, float('-inf'))
        p_ij = tl.exp(s_ij - l_i[:, None])
        
        dp_ij = tl.dot(do_i, tl.trans(v_j), out_dtype=tl.float32)
        ds_ij = p_ij * (dp_ij - d_i[:, None]) * sm_scale

        dq_acc = tl.dot(ds_ij.to(tl.bfloat16), k_j, acc=dq_acc, out_dtype=tl.float32)

    dq_desc = tl.make_tensor_descriptor(
        dQ + off_dq, shape=[S, d], strides=[stride_dqs, 1],
        block_shape=[BLOCK_M, d]
    )
    dq_desc.store([start_m, 0], dq_acc.to(dQ.dtype.element_ty))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes the backward pass of causal multi-head attention.
    Receives all input tensors in definition order, followed by preallocated output tensors.
    Utilizes Hopper TMA descriptors created device-side for maximum bandwidth and WGMMA performance.
    """
    torch.cuda.set_device(Q.device)
    
    # Provide the standard descriptor allocator needed for tl.make_tensor_descriptor
    def alloc_fn(size: int, alignment: int, stream):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    B, H, S, d = Q.shape
    sm_scale = 1.0 / math.sqrt(d)
    
    grid_dk_dv = lambda META: (triton.cdiv(S, META['BLOCK_N']), B, H)
    bwd_kernel_dk_dv[grid_dk_dv](
        Q, K, V, O, dO, L, dK, dV,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        B, H, S, sm_scale,
        d=d
    )
    
    grid_dq = lambda META: (triton.cdiv(S, META['BLOCK_M']), B, H)
    bwd_kernel_dq[grid_dq](
        Q, K, V, O, dO, L, dQ,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        B, H, S, sm_scale,
        d=d
    )