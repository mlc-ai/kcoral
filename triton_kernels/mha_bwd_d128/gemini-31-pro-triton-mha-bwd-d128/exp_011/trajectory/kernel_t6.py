import math
import torch
import triton
import triton.language as tl

# Provide infrastructure allocation strictly for Triton's device-side descriptors
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

@triton.jit
def _bwd_preprocess_kernel(
    O, dO, D,
    seqlen,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_db, stride_dh, stride_ds,
    BLOCK_M: tl.constexpr, D_HEAD: tl.constexpr
):
    """
    Precomputes the row-wise dot product D = sum(O * dO, axis=-1).
    Extracting this avoids O(N^2) memory reads of O in the main backward kernels.
    """
    pid_m = tl.program_id(0)
    b_id = tl.program_id(1)
    h_id = tl.program_id(2)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D_HEAD)
    mask_m = offs_m < seqlen
    
    o_ptrs = O + b_id * stride_ob + h_id * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    do_ptrs = dO + b_id * stride_dob + h_id * stride_doh + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod
    
    o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
    do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
    
    d = tl.sum(o * do, axis=1)
    
    d_ptrs = D + b_id * stride_db + h_id * stride_dh + offs_m * stride_ds
    tl.store(d_ptrs, d, mask=mask_m)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
    ],
    key=['seqlen']
)
@triton.jit
def _bwd_dq_kernel(
    Q, K, V, dO, dQ, L, D,
    seqlen, scale,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    stride_lb, stride_lh, stride_ls,
    stride_db, stride_dh, stride_ds,
    H,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    D_HEAD: tl.constexpr,
):
    pid_m = tl.program_id(0)
    bh_id = tl.program_id(1)
    b_id = bh_id // H
    h_id = bh_id % H
    
    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    mask_m = offs_m < seqlen
    offs_d = tl.arange(0, D_HEAD)
    
    # Outer-loop variants (Q, dO) loaded via pointers directly into registers
    q_ptrs = Q + b_id * stride_qb + h_id * stride_qh + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    do_ptrs = dO + b_id * stride_dob + h_id * stride_doh + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod
    l_ptrs = L + b_id * stride_lb + h_id * stride_lh + offs_m * stride_ls
    d_ptrs = D + b_id * stride_db + h_id * stride_dh + offs_m * stride_ds
    
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
    l = tl.load(l_ptrs, mask=mask_m, other=0.0)
    d = tl.load(d_ptrs, mask=mask_m, other=0.0)
    
    q_scaled = (q * scale).to(tl.bfloat16)
    dq = tl.zeros([BLOCK_M, D_HEAD], dtype=tl.float32)
    
    k_base = K + b_id * stride_kb + h_id * stride_kh
    v_base = V + b_id * stride_vb + h_id * stride_vh
    
    # Establish device TMA tensor descriptors for pipelined asynchronous SMEM loads
    k_desc = tl.make_tensor_descriptor(k_base, shape=[seqlen, D_HEAD], strides=[stride_ks, stride_kd], block_shape=[BLOCK_N, D_HEAD], padding_option="zero")
    v_desc = tl.make_tensor_descriptor(v_base, shape=[seqlen, D_HEAD], strides=[stride_vs, stride_vd], block_shape=[BLOCK_N, D_HEAD], padding_option="zero")
    
    # Range loop pipelines natively with Hopper SMEM via TMA
    for start_n in range(0, seqlen, BLOCK_N):
        k = k_desc.load([start_n, 0])
        v = v_desc.load([start_n, 0])
        
        # WGMMA bindings natively emit since A is Reg and B is SMEM 
        qk = tl.dot(q_scaled, k.T, out_dtype=tl.float32)
        
        p = tl.exp(qk - l[:, None])
        
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < seqlen
        p = tl.where((mask_m[:, None]) & (mask_n[None, :]), p, 0.0)
        
        dp = tl.dot(do, v.T, out_dtype=tl.float32)
        
        ds = p * (dp - d[:, None]) * scale
        
        dq += tl.dot(ds.to(tl.bfloat16), k, out_dtype=tl.float32)
        
    dq_ptrs = dQ + b_id * stride_dqb + h_id * stride_dqh + offs_m[:, None] * stride_dqs + offs_d[None, :] * stride_dqd
    tl.store(dq_ptrs, dq.to(tl.bfloat16), mask=mask_m[:, None])


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
    ],
    key=['seqlen']
)
@triton.jit
def _bwd_dk_dv_kernel(
    Q, K, V, dO, dK, dV, L, D,
    seqlen, scale,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    stride_lb, stride_lh, stride_ls,
    stride_db, stride_dh, stride_ds,
    H,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    D_HEAD: tl.constexpr,
):
    pid_n = tl.program_id(0)
    bh_id = tl.program_id(1)
    b_id = bh_id // H
    h_id = bh_id % H
    
    start_n = pid_n * BLOCK_N
    offs_n = start_n + tl.arange(0, BLOCK_N)
    mask_n = offs_n < seqlen
    offs_d = tl.arange(0, D_HEAD)
    
    # Outer-loop K and V loaded once into Reg
    k_ptrs = K + b_id * stride_kb + h_id * stride_kh + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = V + b_id * stride_vb + h_id * stride_vh + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
    
    k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
    
    k_scaled = (k * scale).to(tl.bfloat16)
    
    dk = tl.zeros([BLOCK_N, D_HEAD], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, D_HEAD], dtype=tl.float32)
    
    q_base = Q + b_id * stride_qb + h_id * stride_qh
    do_base = dO + b_id * stride_dob + h_id * stride_doh
    
    # Inner-loop variants are fed via TMA descriptors to stream as Operand B seamlessly
    q_desc = tl.make_tensor_descriptor(q_base, shape=[seqlen, D_HEAD], strides=[stride_qs, stride_qd], block_shape=[BLOCK_M, D_HEAD], padding_option="zero")
    do_desc = tl.make_tensor_descriptor(do_base, shape=[seqlen, D_HEAD], strides=[stride_dos, stride_dod], block_shape=[BLOCK_M, D_HEAD], padding_option="zero")
    
    for start_m in range(0, seqlen, BLOCK_M):
        q = q_desc.load([start_m, 0])
        do = do_desc.load([start_m, 0])
        
        offs_m = start_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < seqlen
        
        l_ptrs = L + b_id * stride_lb + h_id * stride_lh + offs_m * stride_ls
        d_ptrs = D + b_id * stride_db + h_id * stride_dh + offs_m * stride_ds
        
        l = tl.load(l_ptrs, mask=mask_m, other=0.0)
        d = tl.load(d_ptrs, mask=mask_m, other=0.0)
        
        kq = tl.dot(k_scaled, q.T, out_dtype=tl.float32)
        
        p_T = tl.exp(kq - l[None, :])
        p_T = tl.where((mask_n[:, None]) & (mask_m[None, :]), p_T, 0.0)
        
        dp_T = tl.dot(v, do.T, out_dtype=tl.float32)
        
        ds_T = p_T * (dp_T - d[None, :]) * scale
        
        dv += tl.dot(p_T.to(tl.bfloat16), do, out_dtype=tl.float32)
        dk += tl.dot(ds_T.to(tl.bfloat16), q, out_dtype=tl.float32)
        
    dk_ptrs = dK + b_id * stride_dkb + h_id * stride_dkh + offs_n[:, None] * stride_dks + offs_d[None, :] * stride_dkd
    dv_ptrs = dV + b_id * stride_dvb + h_id * stride_dvh + offs_n[:, None] * stride_dvs + offs_d[None, :] * stride_dvd
    
    tl.store(dk_ptrs, dk.to(tl.bfloat16), mask=mask_n[:, None])
    tl.store(dv_ptrs, dv.to(tl.bfloat16), mask=mask_n[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes the backward pass for multi-head attention (non-causal).
    Writes output into preallocated dQ, dK, dV tensors using WGMMA and TMA.
    """
    with torch.cuda.device(Q.device):
        B, H_dim, seqlen, D_HEAD = Q.shape
        scale = 1.0 / math.sqrt(D_HEAD)
        
        # O(N) pre-scaling accumulation of dO ◦ O to circumvent O(N^2) inner-loop memory reads
        D = torch.empty((B, H_dim, seqlen), device=Q.device, dtype=torch.float32)
        grid_prep = (triton.cdiv(seqlen, 128), B, H_dim)
        
        _bwd_preprocess_kernel[grid_prep](
            O, dO, D,
            seqlen,
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
            D.stride(0), D.stride(1), D.stride(2),
            BLOCK_M=128, D_HEAD=D_HEAD
        )
        
        # Parallel CTA grids designed structurally to avoid atomic adds and lock contention 
        grid_dq = lambda META: (triton.cdiv(seqlen, META['BLOCK_M']), B * H_dim)
        grid_dk_dv = lambda META: (triton.cdiv(seqlen, META['BLOCK_N']), B * H_dim)
        
        if L.dim() == 4:
            stride_lb, stride_lh, stride_ls, _ = L.stride()
        else:
            stride_lb, stride_lh, stride_ls = L.stride()
        
        _bwd_dq_kernel[grid_dq](
            Q, K, V, dO, dQ, L, D,
            seqlen, scale,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
            dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
            stride_lb, stride_lh, stride_ls,
            D.stride(0), D.stride(1), D.stride(2),
            H_dim,
            D_HEAD=D_HEAD,
        )
        
        _bwd_dk_dv_kernel[grid_dk_dv](
            Q, K, V, dO, dK, dV, L, D,
            seqlen, scale,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
            dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
            dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
            stride_lb, stride_lh, stride_ls,
            D.stride(0), D.stride(1), D.stride(2),
            H_dim,
            D_HEAD=D_HEAD,
        )