import math
import torch
import triton
import triton.language as tl

# Install Triton's descriptor allocator for device-side TMA descriptors
def _alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(_alloc_fn)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=3, num_warps=4),
    ],
    key=['seq_len']
)
@triton.jit
def bwd_kernel_dq(
    Q, K, V, O, sm_scale, dO, dQ, L,
    seq_len,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    stride_lb, stride_lh, stride_ls,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D_HEAD: tl.constexpr
):
    pid_m = tl.program_id(0)
    start_m = pid_m * BLOCK_M
    
    if start_m >= seq_len:
        return
        
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    # Base pointers utilizing TMA capabilities natively bounds-checked and padded
    q_ptr = Q + pid_b * stride_qb + pid_h * stride_qh
    q_desc = tl.make_tensor_descriptor(
        q_ptr, shape=[seq_len, D_HEAD], strides=[stride_qs, stride_qd],
        block_shape=[BLOCK_M, D_HEAD], padding_option="zero"
    )
    q = q_desc.load([start_m, 0])
    
    do_ptr = dO + pid_b * stride_dob + pid_h * stride_doh
    do_desc = tl.make_tensor_descriptor(
        do_ptr, shape=[seq_len, D_HEAD], strides=[stride_dos, stride_dod],
        block_shape=[BLOCK_M, D_HEAD], padding_option="zero"
    )
    do = do_desc.load([start_m, 0])
    
    o_ptr = O + pid_b * stride_ob + pid_h * stride_oh
    o_desc = tl.make_tensor_descriptor(
        o_ptr, shape=[seq_len, D_HEAD], strides=[stride_os, stride_od],
        block_shape=[BLOCK_M, D_HEAD], padding_option="zero"
    )
    o = o_desc.load([start_m, 0])
    
    offs_m = start_m + tl.arange(0, BLOCK_M)
    mask_m = offs_m < seq_len
    l_ptrs = L + pid_b * stride_lb + pid_h * stride_lh + offs_m * stride_ls
    l = tl.load(l_ptrs, mask=mask_m, other=0.0)
    
    # Precompute local row sum of (O * dO) for the current Q block (outside the j loop)
    d_val = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
    
    k_ptr = K + pid_b * stride_kb + pid_h * stride_kh
    k_desc = tl.make_tensor_descriptor(
        k_ptr, shape=[seq_len, D_HEAD], strides=[stride_ks, stride_kd],
        block_shape=[BLOCK_N, D_HEAD], padding_option="zero"
    )
    
    v_ptr = V + pid_b * stride_vb + pid_h * stride_vh
    v_desc = tl.make_tensor_descriptor(
        v_ptr, shape=[seq_len, D_HEAD], strides=[stride_vs, stride_vd],
        block_shape=[BLOCK_N, D_HEAD], padding_option="zero"
    )
    
    dq = tl.zeros([BLOCK_M, D_HEAD], dtype=tl.float32)
    
    # 1. Unmasked steps: Fully out of range of any causal overlap
    num_unmasked_n = start_m // BLOCK_N
    for step in range(num_unmasked_n):
        start_n = step * BLOCK_N
        k = k_desc.load([start_n, 0])
        v = v_desc.load([start_n, 0])
        
        qk = tl.dot(q, k.T)
        qk = qk * sm_scale
        
        qk = tl.where(mask_m[:, None], qk, float("-inf"))
        p = tl.exp(qk - l[:, None])
        p = tl.where(mask_m[:, None], p, 0.0)
        
        dp = tl.dot(do, v.T)
        ds = p * (dp - d_val[:, None]) * sm_scale
        
        ds_cast = tl.cast(ds, q.dtype)
        dq = tl.dot(ds_cast, k, acc=dq)
        
    # 2. Causal steps: Unrolled bound checked segments ensuring minimal divergence overhead
    start_n_initial = num_unmasked_n * BLOCK_N
    num_causal_n = tl.constexpr(max(1, BLOCK_M // BLOCK_N))
    
    for step in tl.range(0, num_causal_n):
        start_n = start_n_initial + step * BLOCK_N
        k = k_desc.load([start_n, 0])
        v = v_desc.load([start_n, 0])
        
        qk = tl.dot(q, k.T)
        qk = qk * sm_scale
        
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_causal = (offs_m[:, None] >= offs_n[None, :]) & mask_m[:, None]
        qk = tl.where(mask_causal, qk, float("-inf"))
        
        p = tl.exp(qk - l[:, None])
        p = tl.where(mask_causal, p, 0.0)
        
        dp = tl.dot(do, v.T)
        ds = p * (dp - d_val[:, None]) * sm_scale
        
        ds_cast = tl.cast(ds, q.dtype)
        dq = tl.dot(ds_cast, k, acc=dq)
        
    dq_ptr = dQ + pid_b * stride_dqb + pid_h * stride_dqh
    dq_desc = tl.make_tensor_descriptor(
        dq_ptr, shape=[seq_len, D_HEAD], strides=[stride_dqs, stride_dqd],
        block_shape=[BLOCK_M, D_HEAD]
    )
    dq_desc.store([start_m, 0], tl.cast(dq, dQ.dtype.element_ty))


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 128, 'BLOCK_M': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_N': 128, 'BLOCK_M': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_N': 64, 'BLOCK_M': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_N': 64, 'BLOCK_M': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_N': 64, 'BLOCK_M': 64}, num_stages=3, num_warps=4),
    ],
    key=['seq_len']
)
@triton.jit
def bwd_kernel_dk_dv(
    Q, K, V, O, sm_scale, dO, dK, dV, L,
    seq_len,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    stride_lb, stride_lh, stride_ls,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D_HEAD: tl.constexpr
):
    pid_n = tl.program_id(0)
    start_n = pid_n * BLOCK_N
    
    if start_n >= seq_len:
        return
        
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    k_ptr = K + pid_b * stride_kb + pid_h * stride_kh
    k_desc = tl.make_tensor_descriptor(
        k_ptr, shape=[seq_len, D_HEAD], strides=[stride_ks, stride_kd],
        block_shape=[BLOCK_N, D_HEAD], padding_option="zero"
    )
    k = k_desc.load([start_n, 0])
    
    v_ptr = V + pid_b * stride_vb + pid_h * stride_vh
    v_desc = tl.make_tensor_descriptor(
        v_ptr, shape=[seq_len, D_HEAD], strides=[stride_vs, stride_vd],
        block_shape=[BLOCK_N, D_HEAD], padding_option="zero"
    )
    v = v_desc.load([start_n, 0])
    
    q_ptr = Q + pid_b * stride_qb + pid_h * stride_qh
    q_desc = tl.make_tensor_descriptor(
        q_ptr, shape=[seq_len, D_HEAD], strides=[stride_qs, stride_qd],
        block_shape=[BLOCK_M, D_HEAD], padding_option="zero"
    )
    
    do_ptr = dO + pid_b * stride_dob + pid_h * stride_doh
    do_desc = tl.make_tensor_descriptor(
        do_ptr, shape=[seq_len, D_HEAD], strides=[stride_dos, stride_dod],
        block_shape=[BLOCK_M, D_HEAD], padding_option="zero"
    )
    
    o_ptr = O + pid_b * stride_ob + pid_h * stride_oh
    o_desc = tl.make_tensor_descriptor(
        o_ptr, shape=[seq_len, D_HEAD], strides=[stride_os, stride_od],
        block_shape=[BLOCK_M, D_HEAD], padding_option="zero"
    )
    
    l_ptr_base = L + pid_b * stride_lb + pid_h * stride_lh
    
    dk = tl.zeros([BLOCK_N, D_HEAD], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, D_HEAD], dtype=tl.float32)
    
    offs_n = start_n + tl.arange(0, BLOCK_N)
    
    # 1. Causal steps handling diagonal sequence interactions inherently
    start_m_initial = (start_n // BLOCK_M) * BLOCK_M
    num_causal_m = tl.constexpr(max(1, BLOCK_N // BLOCK_M))
    
    for step in tl.range(0, num_causal_m):
        start_m = start_m_initial + step * BLOCK_M
        
        q = q_desc.load([start_m, 0])
        do = do_desc.load([start_m, 0])
        o = o_desc.load([start_m, 0])
        
        offs_m = start_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < seq_len
        l = tl.load(l_ptr_base + offs_m * stride_ls, mask=mask_m, other=0.0)
        
        d_val = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
        
        qk = tl.dot(q, k.T)
        qk = qk * sm_scale
        
        mask_causal = (offs_m[:, None] >= offs_n[None, :]) & mask_m[:, None]
        qk = tl.where(mask_causal, qk, float("-inf"))
        
        p = tl.exp(qk - l[:, None])
        p = tl.where(mask_causal, p, 0.0)
        
        dp = tl.dot(do, v.T)
        ds = p * (dp - d_val[:, None]) * sm_scale
        
        ds_cast = tl.cast(ds, q.dtype)
        p_cast = tl.cast(p, q.dtype)
        
        # Maps robustly into Register-Shared configuration (RS-GEMM paths) without warp spilling
        dv = tl.dot(p_cast.T, do, acc=dv)
        dk = tl.dot(ds_cast.T, q, acc=dk)
        
    # 2. Unmasked steps structurally relying on WGMMA Tensor Core throughput (clean sequence loops)
    start_m_unmasked = start_m_initial + num_causal_m * BLOCK_M
    start_q_block = start_m_unmasked // BLOCK_M
    total_q_blocks = tl.cdiv(seq_len, BLOCK_M)
    
    for step in range(start_q_block, total_q_blocks):
        start_m = step * BLOCK_M
        
        q = q_desc.load([start_m, 0])
        do = do_desc.load([start_m, 0])
        o = o_desc.load([start_m, 0])
        
        offs_m = start_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < seq_len
        l = tl.load(l_ptr_base + offs_m * stride_ls, mask=mask_m, other=0.0)
        
        d_val = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
        
        qk = tl.dot(q, k.T)
        qk = qk * sm_scale
        
        p = tl.exp(qk - l[:, None])
        
        dp = tl.dot(do, v.T)
        ds = p * (dp - d_val[:, None]) * sm_scale
        
        ds_cast = tl.cast(ds, q.dtype)
        p_cast = tl.cast(p, q.dtype)
        
        dv = tl.dot(p_cast.T, do, acc=dv)
        dk = tl.dot(ds_cast.T, q, acc=dk)
        
    dk_ptr = dK + pid_b * stride_dkb + pid_h * stride_dkh
    dk_desc = tl.make_tensor_descriptor(
        dk_ptr, shape=[seq_len, D_HEAD], strides=[stride_dks, stride_dkd],
        block_shape=[BLOCK_N, D_HEAD]
    )
    dk_desc.store([start_n, 0], tl.cast(dk, dK.dtype.element_ty))
    
    dv_ptr = dV + pid_b * stride_dvb + pid_h * stride_dvh
    dv_desc = tl.make_tensor_descriptor(
        dv_ptr, shape=[seq_len, D_HEAD], strides=[stride_dvs, stride_dvd],
        block_shape=[BLOCK_N, D_HEAD]
    )
    dv_desc.store([start_n, 0], tl.cast(dv, dV.dtype.element_ty))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes causal Multi-Head Attention backward pass utilizing optimal Standard Triton primitives.
    Results are seamlessly written inplace into destination tensors: `dQ`, `dK`, `dV`.
    
    Leverages NVIDIA Hopper capabilities directly such as robust TMA access bounds, automatic sequence 
    length padded zeros masking within Tensor Core accumulations and unrolled Register-Shared RS-GEMMs.
    """
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    sm_scale = 1.0 / math.sqrt(d)
    
    # Unifying the Logsumexp shape dimensionality explicitly addressing edge PyTorch backward paths natively
    L_sq = L.squeeze(-1) if L.dim() == 4 else L
    
    grid_dq = lambda META: (
        triton.cdiv(S, META['BLOCK_M']),
        B,
        H
    )
    
    bwd_kernel_dq[grid_dq](
        Q, K, V, O, sm_scale, dO, dQ, L_sq,
        S,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        L_sq.stride(0), L_sq.stride(1), L_sq.stride(2),
        D_HEAD=d
    )

    grid_dkdv = lambda META: (
        triton.cdiv(S, META['BLOCK_N']),
        B,
        H
    )
    
    bwd_kernel_dk_dv[grid_dkdv](
        Q, K, V, O, sm_scale, dO, dK, dV, L_sq,
        S,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        L_sq.stride(0), L_sq.stride(1), L_sq.stride(2),
        D_HEAD=d
    )