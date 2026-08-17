import math
import torch
import triton
import triton.language as tl

# Provide infrastructure allocation for Triton's device-side descriptors
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
    ],
    key=['seqlen']
)
@triton.jit
def _bwd_dq_kernel(
    Q, K, V, O, dO, dQ, L,
    seqlen, scale,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    stride_lb, stride_lh, stride_ls,
    H,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    D_HEAD: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    b_id = pid_bh // H
    h_id = pid_bh % H
    
    # Calculate 2D tensor base pointers for the active batch and head
    q_base = Q + b_id * stride_qb + h_id * stride_qh
    k_base = K + b_id * stride_kb + h_id * stride_kh
    v_base = V + b_id * stride_vb + h_id * stride_vh
    o_base = O + b_id * stride_ob + h_id * stride_oh
    do_base = dO + b_id * stride_dob + h_id * stride_doh
    dq_base = dQ + b_id * stride_dqb + h_id * stride_dqh
    
    # Establish device TMA tensor descriptors
    q_desc = tl.make_tensor_descriptor(q_base, shape=[seqlen, D_HEAD], strides=[stride_qs, stride_qd], block_shape=[BLOCK_M, D_HEAD], padding_option="zero")
    o_desc = tl.make_tensor_descriptor(o_base, shape=[seqlen, D_HEAD], strides=[stride_os, stride_od], block_shape=[BLOCK_M, D_HEAD], padding_option="zero")
    do_desc = tl.make_tensor_descriptor(do_base, shape=[seqlen, D_HEAD], strides=[stride_dos, stride_dod], block_shape=[BLOCK_M, D_HEAD], padding_option="zero")
    dq_desc = tl.make_tensor_descriptor(dq_base, shape=[seqlen, D_HEAD], strides=[stride_dqs, stride_dqd], block_shape=[BLOCK_M, D_HEAD])
    
    k_desc = tl.make_tensor_descriptor(k_base, shape=[seqlen, D_HEAD], strides=[stride_ks, stride_kd], block_shape=[BLOCK_N, D_HEAD], padding_option="zero")
    v_desc = tl.make_tensor_descriptor(v_base, shape=[seqlen, D_HEAD], strides=[stride_vs, stride_vd], block_shape=[BLOCK_N, D_HEAD], padding_option="zero")
    
    start_m = pid_m * BLOCK_M
    
    q = q_desc.load([start_m, 0])
    o = o_desc.load([start_m, 0])
    do = do_desc.load([start_m, 0])
    
    offs_m = start_m + tl.arange(0, BLOCK_M)
    mask_m = offs_m < seqlen
    l_ptrs = L + b_id * stride_lb + h_id * stride_lh + offs_m * stride_ls
    l = tl.load(l_ptrs, mask=mask_m, other=0.0)
    
    # Scale calculation for backwards gradient alignment
    d = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
    
    dq = tl.zeros([BLOCK_M, D_HEAD], dtype=tl.float32)
    
    # Accumulate dQ components through Key and Value iterations, efficiently buffered 
    # to HBM using Hopper TMA 
    for start_n in tl.range(0, seqlen, BLOCK_N, num_stages=3):
        k = k_desc.load([start_n, 0])
        v = v_desc.load([start_n, 0])
        
        qk = tl.dot(q, k.T) * scale
        
        p = tl.exp(qk - l[:, None])
        
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < seqlen
        p = tl.where((mask_m[:, None]) & (mask_n[None, :]), p, 0.0)
        
        dp = tl.dot(do, v.T)
        
        ds = p * (dp - d[:, None]) * scale
        
        # Operand A in Register, Operand B generated from SMEM mapping (from TMA load)
        dq += tl.dot(ds.to(tl.bfloat16), k)
        
    dq_desc.store([start_m, 0], dq.to(tl.bfloat16))


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
    ],
    key=['seqlen']
)
@triton.jit
def _bwd_dk_dv_kernel(
    Q, K, V, O, dO, dK, dV, L,
    seqlen, scale,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    stride_lb, stride_lh, stride_ls,
    H,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    D_HEAD: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    b_id = pid_bh // H
    h_id = pid_bh % H
    
    q_base = Q + b_id * stride_qb + h_id * stride_qh
    k_base = K + b_id * stride_kb + h_id * stride_kh
    v_base = V + b_id * stride_vb + h_id * stride_vh
    o_base = O + b_id * stride_ob + h_id * stride_oh
    do_base = dO + b_id * stride_dob + h_id * stride_doh
    dk_base = dK + b_id * stride_dkb + h_id * stride_dkh
    dv_base = dV + b_id * stride_dvb + h_id * stride_dvh
    
    q_desc = tl.make_tensor_descriptor(q_base, shape=[seqlen, D_HEAD], strides=[stride_qs, stride_qd], block_shape=[BLOCK_M, D_HEAD], padding_option="zero")
    o_desc = tl.make_tensor_descriptor(o_base, shape=[seqlen, D_HEAD], strides=[stride_os, stride_od], block_shape=[BLOCK_M, D_HEAD], padding_option="zero")
    do_desc = tl.make_tensor_descriptor(do_base, shape=[seqlen, D_HEAD], strides=[stride_dos, stride_dod], block_shape=[BLOCK_M, D_HEAD], padding_option="zero")
    
    k_desc = tl.make_tensor_descriptor(k_base, shape=[seqlen, D_HEAD], strides=[stride_ks, stride_kd], block_shape=[BLOCK_N, D_HEAD], padding_option="zero")
    v_desc = tl.make_tensor_descriptor(v_base, shape=[seqlen, D_HEAD], strides=[stride_vs, stride_vd], block_shape=[BLOCK_N, D_HEAD], padding_option="zero")
    dk_desc = tl.make_tensor_descriptor(dk_base, shape=[seqlen, D_HEAD], strides=[stride_dks, stride_dkd], block_shape=[BLOCK_N, D_HEAD])
    dv_desc = tl.make_tensor_descriptor(dv_base, shape=[seqlen, D_HEAD], strides=[stride_dvs, stride_dvd], block_shape=[BLOCK_N, D_HEAD])
    
    start_n = pid_n * BLOCK_N
    
    k = k_desc.load([start_n, 0])
    v = v_desc.load([start_n, 0])
    
    offs_n = start_n + tl.arange(0, BLOCK_N)
    mask_n = offs_n < seqlen
    
    dk = tl.zeros([BLOCK_N, D_HEAD], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, D_HEAD], dtype=tl.float32)
    
    for start_m in tl.range(0, seqlen, BLOCK_M, num_stages=3):
        q = q_desc.load([start_m, 0])
        o = o_desc.load([start_m, 0])
        do = do_desc.load([start_m, 0])
        
        offs_m = start_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < seqlen
        l_ptrs = L + b_id * stride_lb + h_id * stride_lh + offs_m * stride_ls
        l = tl.load(l_ptrs, mask=mask_m, other=0.0)
        
        d = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
        
        qk = tl.dot(q, k.T) * scale
        
        p = tl.exp(qk - l[:, None])
        p = tl.where((mask_m[:, None]) & (mask_n[None, :]), p, 0.0)
        
        dp = tl.dot(do, v.T)
        
        ds = p * (dp - d[:, None]) * scale
        
        # Operand A locally computed in WGMMA Reg-space, Operand B cleanly piped through TMA to SMEM 
        p_t = p.T.to(tl.bfloat16)
        dv += tl.dot(p_t, do)
        
        ds_t = ds.T.to(tl.bfloat16)
        dk += tl.dot(ds_t, q)
        
    dk_desc.store([start_n, 0], dk.to(tl.bfloat16))
    dv_desc.store([start_n, 0], dv.to(tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes the backward pass for multi-head attention (non-causal).
    Input tensors are (B, H, S, d) and L is (B, H, S) or (B, H, S, 1).
    Writes output into preallocated dQ, dK, dV tensors.
    """
    with torch.cuda.device(Q.device):
        B, H_dim, seqlen, D_HEAD = Q.shape
        scale = 1.0 / math.sqrt(D_HEAD)
        
        # Dynamic CTA bounds setup respecting memory alignments  
        grid_dq = lambda META: (triton.cdiv(seqlen, META['BLOCK_M']), B * H_dim)
        grid_dk_dv = lambda META: (triton.cdiv(seqlen, META['BLOCK_N']), B * H_dim)
        
        # Handle L tensor rank correctly mapping internally applied unsqueeze logic 
        if L.dim() == 4:
            stride_lb, stride_lh, stride_ls, _ = L.stride()
        else:
            stride_lb, stride_lh, stride_ls = L.stride()
        
        _bwd_dq_kernel[grid_dq](
            Q, K, V, O, dO, dQ, L,
            seqlen, scale,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
            dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
            stride_lb, stride_lh, stride_ls,
            H_dim,
            D_HEAD=D_HEAD,
        )
        
        _bwd_dk_dv_kernel[grid_dk_dv](
            Q, K, V, O, dO, dK, dV, L,
            seqlen, scale,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
            dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
            dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
            stride_lb, stride_lh, stride_ls,
            H_dim,
            D_HEAD=D_HEAD,
        )