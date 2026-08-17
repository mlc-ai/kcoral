import math
import torch
import triton
import triton.language as tl

def get_dq_configs():
    return [
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
    ]

def get_dkdv_configs():
    return [
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
    ]

@triton.jit
def precompute_D_kernel(
    O, dO, dQ,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    S, d: tl.constexpr, BLOCK_M: tl.constexpr
):
    """
    Precomputes D_i = sum(dO_i * O_i) to avoid O(N^2) HBM reads of O_i in the dk_dv kernel.
    Safely stages the float32 result into the first elements of the uninitialized bfloat16 dQ tensor,
    which bwd_kernel_dk_dv will read, before bwd_kernel_dq overwrites dQ completely.
    """
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    start_m = pid_m * BLOCK_M
    if start_m >= S:
        return

    off_o = pid_b * stride_ob + pid_h * stride_oh
    off_do = pid_b * stride_dob + pid_h * stride_doh
    off_dq = pid_b * stride_dqb + pid_h * stride_dqh

    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, d)
    mask_m = offs_m < S

    o_ptrs = O + off_o + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    do_ptrs = dO + off_do + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod

    o_i = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
    do_i = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)

    # Compute row sum dynamically for the block
    d_i = tl.sum(do_i.to(tl.float32) * o_i.to(tl.float32), axis=1)

    dq_ptrs = dQ + off_dq + offs_m * stride_dqs
    # Stage into the 4 bytes available per bfloat16 row head (since d=128, this fits safely)
    dq_ptrs_f32 = dq_ptrs.to(tl.pointer_type(tl.float32))
    tl.store(dq_ptrs_f32, d_i, mask=mask_m)


@triton.autotune(configs=get_dkdv_configs(), key=['S'])
@triton.jit
def bwd_kernel_dk_dv(
    Q, K, V, dO, L, dQ_for_D, dK, dV,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    S, sm_scale,
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
    off_do = pid_b * stride_dob + pid_h * stride_doh
    off_dq = pid_b * stride_dqb + pid_h * stride_dqh
    off_l = pid_b * stride_lb + pid_h * stride_lh
    off_dk = pid_b * stride_dkb + pid_h * stride_dkh
    off_dv = pid_b * stride_dvb + pid_h * stride_dvh

    # Create TMA descriptors for memory loads outside the inner loop
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

    # Causal logic: we only process Q blocks that start at or before n's column
    start_m_initial = (start_n // BLOCK_M) * BLOCK_M
    num_steps = tl.cdiv(S - start_m_initial, BLOCK_M)

    q_desc = tl.make_tensor_descriptor(
        Q + off_q, shape=[S, d], strides=[stride_qs, 1],
        block_shape=[BLOCK_M, d], padding_option="zero"
    )
    do_desc = tl.make_tensor_descriptor(
        dO + off_do, shape=[S, d], strides=[stride_dos, 1],
        block_shape=[BLOCK_M, d], padding_option="zero"
    )

    offs_n = start_n + tl.arange(0, BLOCK_N)

    # Core loop. Triton automatically pipelines TMA loads of q_i and do_i over num_stages
    for step in range(num_steps):
        start_m = start_m_initial + step * BLOCK_M
        
        q_i = q_desc.load([start_m, 0])
        do_i = do_desc.load([start_m, 0])

        offs_m = start_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S
        
        l_ptrs = L + off_l + offs_m * stride_ls
        l_i = tl.load(l_ptrs, mask=mask_m, other=0.0)

        # Load safely precomputed D_i
        dq_ptrs = dQ_for_D + off_dq + offs_m * stride_dqs
        d_i = tl.load(dq_ptrs.to(tl.pointer_type(tl.float32)), mask=mask_m, other=0.0)

        s_ij = tl.dot(q_i, tl.trans(k_j), out_dtype=tl.float32) * sm_scale

        valid = (offs_m[:, None] < S) & (offs_n[None, :] < S)
        if start_m < start_n + BLOCK_N:
            valid = valid & (offs_m[:, None] >= offs_n[None, :])
            
        s_ij = tl.where(valid, s_ij, float('-inf'))
        p_ij = tl.exp(s_ij - l_i[:, None])

        dp_ij = tl.dot(do_i, tl.trans(v_j), out_dtype=tl.float32)
        ds_ij = p_ij * (dp_ij - d_i[:, None]) * sm_scale

        # Hopper WGMMA effectively consumes fp16/bf16 operations; explicitly convert scales prior to execution
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
    Q, K, V, dO, L, dQ_for_D, dQ,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    S, sm_scale,
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
    off_do = pid_b * stride_dob + pid_h * stride_doh
    off_dq = pid_b * stride_dqb + pid_h * stride_dqh
    off_l = pid_b * stride_lb + pid_h * stride_lh

    q_desc = tl.make_tensor_descriptor(
        Q + off_q, shape=[S, d], strides=[stride_qs, 1],
        block_shape=[BLOCK_M, d], padding_option="zero"
    )
    do_desc = tl.make_tensor_descriptor(
        dO + off_do, shape=[S, d], strides=[stride_dos, 1],
        block_shape=[BLOCK_M, d], padding_option="zero"
    )

    q_i = q_desc.load([start_m, 0])
    do_i = do_desc.load([start_m, 0])

    offs_m = start_m + tl.arange(0, BLOCK_M)
    mask_m = offs_m < S
    
    l_ptrs = L + off_l + offs_m * stride_ls
    l_i = tl.load(l_ptrs, mask=mask_m, other=0.0)

    dq_ptrs = dQ_for_D + off_dq + offs_m * stride_dqs
    d_i = tl.load(dq_ptrs.to(tl.pointer_type(tl.float32)), mask=mask_m, other=0.0)

    dq_acc = tl.zeros((BLOCK_M, d), dtype=tl.float32)

    k_desc = tl.make_tensor_descriptor(
        K + off_k, shape=[S, d], strides=[stride_ks, 1],
        block_shape=[BLOCK_N, d], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        V + off_v, shape=[S, d], strides=[stride_vs, 1],
        block_shape=[BLOCK_N, d], padding_option="zero"
    )

    # Causal logic: we only process K blocks that end at or before m's row
    max_n = tl.minimum(S, start_m + BLOCK_M)
    num_steps = tl.cdiv(max_n, BLOCK_N)

    offs_n = tl.arange(0, BLOCK_N)

    # Core pipelined inner loop iterating across K, V descriptors
    for step in range(num_steps):
        start_n = step * BLOCK_N
        
        k_j = k_desc.load([start_n, 0])
        v_j = v_desc.load([start_n, 0])

        s_ij = tl.dot(q_i, tl.trans(k_j), out_dtype=tl.float32) * sm_scale

        offs_n_cur = start_n + offs_n
        valid = (offs_m[:, None] < S) & (offs_n_cur[None, :] < S)
        
        if start_n + BLOCK_N > start_m:
            valid = valid & (offs_m[:, None] >= offs_n_cur[None, :])
            
        s_ij = tl.where(valid, s_ij, float('-inf'))
        p_ij = tl.exp(s_ij - l_i[:, None])
        
        dp_ij = tl.dot(do_i, tl.trans(v_j), out_dtype=tl.float32)
        ds_ij = p_ij * (dp_ij - d_i[:, None]) * sm_scale

        dq_acc = tl.dot(ds_ij.to(tl.bfloat16), k_j, acc=dq_acc, out_dtype=tl.float32)

    # We now safely overwrite the earlier staging outputs with final evaluations
    dq_desc = tl.make_tensor_descriptor(
        dQ + off_dq, shape=[S, d], strides=[stride_dqs, 1],
        block_shape=[BLOCK_M, d]
    )
    dq_desc.store([start_m, 0], dq_acc.to(dQ.dtype.element_ty))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes the backward pass of causal multi-head attention leveraging Hopper TMA and 
    Tensor Cores pipelined by Triton's JIT. The O(N^2) load overhead of the original 
    softmax-vector pre-accumulation D is avoided by precomputing it and staging it in `dQ`.
    """
    torch.cuda.set_device(Q.device)

    # Create a Python list reference that strictly maintains the allocation scopes of all TMA 
    # descriptors on the device across kernels preventing memcheck allocation leaks.
    workspace = []
    
    def alloc_fn(size: int, alignment: int, stream):
        t = torch.empty(size, device="cuda", dtype=torch.int8)
        workspace.append(t)
        return t

    triton.set_allocator(alloc_fn)

    B, H, S, d = Q.shape
    sm_scale = float(1.0 / math.sqrt(d))
    
    # 1. Precompute `D_i = rowsum(dO * O)` and stage it safely in the memory
    # space of the dQ tensor, to be consumed by the inner loop of bwd_kernel_dk_dv.
    # We do this to avoid an O(N^2) memory read for O in the dk_dv kernel.
    grid_precompute = (triton.cdiv(S, 128), B, H)
    precompute_D_kernel[grid_precompute](
        O, dO, dQ,
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        S, d=d, BLOCK_M=128
    )
    
    # 2. Reconstruct `dK` and `dV`. Sequential CUDA stream execution safely prevents
    # reading the partially written dQ prior to its final overwriting.
    grid_dk_dv = lambda META: (triton.cdiv(S, META['BLOCK_N']), B, H)
    bwd_kernel_dk_dv[grid_dk_dv](
        Q, K, V, dO, L, dQ, dK, dV,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        S, sm_scale, d=d
    )
    
    # 3. Compute `dQ` overriding the same elements loaded temporarily during execution.
    grid_dq = lambda META: (triton.cdiv(S, META['BLOCK_M']), B, H)
    bwd_kernel_dq[grid_dq](
        Q, K, V, dO, L, dQ, dQ,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        S, sm_scale, d=d
    )