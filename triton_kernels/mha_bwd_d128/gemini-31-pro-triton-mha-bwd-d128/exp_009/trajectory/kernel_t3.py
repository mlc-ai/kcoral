import torch
import triton
import triton.language as tl

# Set up allocator for Triton device-created tensor descriptors
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
    ],
    key=['seqlen_q']
)
@triton.jit
def bwd_kernel_dq(
    Q, K, V, L, O, dO, dQ,
    sm_scale, seqlen_q,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    stride_lb, stride_lh, stride_ls,
    H,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D_HEAD: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    b_idx = pid_bh // H
    h_idx = pid_bh % H
    
    q_base = Q + b_idx * stride_qb + h_idx * stride_qh
    k_base = K + b_idx * stride_kb + h_idx * stride_kh
    v_base = V + b_idx * stride_vb + h_idx * stride_vh
    o_base = O + b_idx * stride_ob + h_idx * stride_oh
    do_base = dO + b_idx * stride_dob + h_idx * stride_doh
    dq_base = dQ + b_idx * stride_dqb + h_idx * stride_dqh
    
    # Device descriptors for TMA
    q_desc = tl.make_tensor_descriptor(
        q_base, shape=[seqlen_q, D_HEAD], strides=[stride_qs, stride_qd],
        block_shape=[BLOCK_M, D_HEAD], padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        o_base, shape=[seqlen_q, D_HEAD], strides=[stride_os, stride_od],
        block_shape=[BLOCK_M, D_HEAD], padding_option="zero"
    )
    do_desc = tl.make_tensor_descriptor(
        do_base, shape=[seqlen_q, D_HEAD], strides=[stride_dos, stride_dod],
        block_shape=[BLOCK_M, D_HEAD], padding_option="zero"
    )
    k_desc = tl.make_tensor_descriptor(
        k_base, shape=[seqlen_q, D_HEAD], strides=[stride_ks, stride_kd],
        block_shape=[BLOCK_N, D_HEAD], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        v_base, shape=[seqlen_q, D_HEAD], strides=[stride_vs, stride_vd],
        block_shape=[BLOCK_N, D_HEAD], padding_option="zero"
    )
    dq_desc = tl.make_tensor_descriptor(
        dq_base, shape=[seqlen_q, D_HEAD], strides=[stride_dqs, stride_dqd],
        block_shape=[BLOCK_M, D_HEAD]
    )
    
    offset_m = pid_m * BLOCK_M
    
    # Load Q, O, dO outside the loop
    q = q_desc.load([offset_m, 0])
    o = o_desc.load([offset_m, 0])
    do = do_desc.load([offset_m, 0])
    
    # L is loaded using traditional pointers
    offs_m = offset_m + tl.arange(0, BLOCK_M)
    mask_m = offs_m < seqlen_q
    l_ptrs = L + b_idx * stride_lb + h_idx * stride_lh + offs_m * stride_ls
    l_i = tl.load(l_ptrs, mask=mask_m, other=0.0)
    
    # D_i = sum(O_i * dO_i, dim=-1) computed once per M block
    D_i = tl.sum(tl.cast(o, tl.float32) * tl.cast(do, tl.float32), axis=1)
    
    dq = tl.zeros([BLOCK_M, D_HEAD], dtype=tl.float32)
    
    offs_n = tl.arange(0, BLOCK_N)
    
    # Loop over N blocks (K and V)
    for offset_n in tl.range(0, seqlen_q, BLOCK_N):
        mask_n = (offset_n + offs_n) < seqlen_q
        
        k = k_desc.load([offset_n, 0])
        v = v_desc.load([offset_n, 0])
        
        qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * sm_scale
        p = tl.exp(qk - l_i[:, None])
        mask_mn = mask_m[:, None] & mask_n[None, :]
        p = tl.where(mask_mn, p, 0.0)
        
        dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32)
        ds = p * (dp - D_i[:, None]) * sm_scale
        ds = tl.where(mask_mn, ds, 0.0)
        
        dq += tl.dot(ds.to(tl.bfloat16), k, out_dtype=tl.float32)
        
    dq_desc.store([offset_m, 0], dq.to(tl.bfloat16))


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 128, 'BLOCK_M': 128}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_N': 64, 'BLOCK_M': 128}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_N': 128, 'BLOCK_M': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_N': 64, 'BLOCK_M': 64}, num_stages=3, num_warps=4),
    ],
    key=['seqlen_q']
)
@triton.jit
def bwd_kernel_dk_dv(
    Q, K, V, L, O, dO, dK, dV,
    sm_scale, seqlen_q,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    stride_lb, stride_lh, stride_ls,
    H,
    BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr, D_HEAD: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    b_idx = pid_bh // H
    h_idx = pid_bh % H
    
    q_base = Q + b_idx * stride_qb + h_idx * stride_qh
    k_base = K + b_idx * stride_kb + h_idx * stride_kh
    v_base = V + b_idx * stride_vb + h_idx * stride_vh
    o_base = O + b_idx * stride_ob + h_idx * stride_oh
    do_base = dO + b_idx * stride_dob + h_idx * stride_doh
    dk_base = dK + b_idx * stride_dkb + h_idx * stride_dkh
    dv_base = dV + b_idx * stride_dvb + h_idx * stride_dvh
    
    # Device descriptors for TMA
    q_desc = tl.make_tensor_descriptor(
        q_base, shape=[seqlen_q, D_HEAD], strides=[stride_qs, stride_qd],
        block_shape=[BLOCK_M, D_HEAD], padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        o_base, shape=[seqlen_q, D_HEAD], strides=[stride_os, stride_od],
        block_shape=[BLOCK_M, D_HEAD], padding_option="zero"
    )
    do_desc = tl.make_tensor_descriptor(
        do_base, shape=[seqlen_q, D_HEAD], strides=[stride_dos, stride_dod],
        block_shape=[BLOCK_M, D_HEAD], padding_option="zero"
    )
    k_desc = tl.make_tensor_descriptor(
        k_base, shape=[seqlen_q, D_HEAD], strides=[stride_ks, stride_kd],
        block_shape=[BLOCK_N, D_HEAD], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        v_base, shape=[seqlen_q, D_HEAD], strides=[stride_vs, stride_vd],
        block_shape=[BLOCK_N, D_HEAD], padding_option="zero"
    )
    dk_desc = tl.make_tensor_descriptor(
        dk_base, shape=[seqlen_q, D_HEAD], strides=[stride_dks, stride_dkd],
        block_shape=[BLOCK_N, D_HEAD]
    )
    dv_desc = tl.make_tensor_descriptor(
        dv_base, shape=[seqlen_q, D_HEAD], strides=[stride_dvs, stride_dvd],
        block_shape=[BLOCK_N, D_HEAD]
    )
    
    offset_n = pid_n * BLOCK_N
    
    # Load K and V outside the loop
    k = k_desc.load([offset_n, 0])
    v = v_desc.load([offset_n, 0])
    
    offs_n = offset_n + tl.arange(0, BLOCK_N)
    mask_n = offs_n < seqlen_q
    
    dk = tl.zeros([BLOCK_N, D_HEAD], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, D_HEAD], dtype=tl.float32)
    
    offs_m = tl.arange(0, BLOCK_M)
    l_base_ptrs = L + b_idx * stride_lb + h_idx * stride_lh
    
    # Loop over M blocks (Q, dO, O)
    for offset_m in tl.range(0, seqlen_q, BLOCK_M):
        mask_m = (offset_m + offs_m) < seqlen_q
        
        q = q_desc.load([offset_m, 0])
        o = o_desc.load([offset_m, 0])
        do = do_desc.load([offset_m, 0])
        
        l_ptrs = l_base_ptrs + (offset_m + offs_m) * stride_ls
        l_i = tl.load(l_ptrs, mask=mask_m, other=0.0)
        
        D_i = tl.sum(tl.cast(o, tl.float32) * tl.cast(do, tl.float32), axis=1)
        
        # Calculate in functionally transposed orientation to avoid register transposes
        qk_T = tl.dot(k, tl.trans(q), out_dtype=tl.float32) * sm_scale
        
        p_T = tl.exp(qk_T - l_i[None, :])
        mask_nm = mask_n[:, None] & mask_m[None, :]
        p_T = tl.where(mask_nm, p_T, 0.0)
        
        dp_T = tl.dot(v, tl.trans(do), out_dtype=tl.float32)
        
        ds_T = p_T * (dp_T - D_i[None, :]) * sm_scale
        ds_T = tl.where(mask_nm, ds_T, 0.0)
        
        dv += tl.dot(p_T.to(tl.bfloat16), do, out_dtype=tl.float32)
        dk += tl.dot(ds_T.to(tl.bfloat16), q, out_dtype=tl.float32)
        
    dk_desc.store([offset_n, 0], dk.to(tl.bfloat16))
    dv_desc.store([offset_n, 0], dv.to(tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes the backward pass for multi-head attention.
    Destination-passing semantics: overwrites pre-allocated dQ, dK, dV tensors in-place.
    """
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape

    if S == 0:
        return

    sm_scale = 1.0 / (d ** 0.5)

    stride_lb = L.stride(0)
    stride_lh = L.stride(1)
    stride_ls = L.stride(2)

    # 1) Compute dQ (iterating over N blocks)
    grid_dq = lambda META: (triton.cdiv(S, META['BLOCK_M']), B * H)
    bwd_kernel_dq[grid_dq](
        Q, K, V, L, O, dO, dQ,
        sm_scale, S,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        stride_lb, stride_lh, stride_ls,
        H,
        D_HEAD=d
    )

    # 2) Compute dK and dV (iterating over M blocks)
    grid_dkdv = lambda META: (triton.cdiv(S, META['BLOCK_N']), B * H)
    bwd_kernel_dk_dv[grid_dkdv](
        Q, K, V, L, O, dO, dK, dV,
        sm_scale, S,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        stride_lb, stride_lh, stride_ls,
        H,
        D_HEAD=d
    )