import torch
import triton
import triton.language as tl

def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=3, num_warps=4),
    ],
    key=['seqlen_q', 'seqlen_k']
)
@triton.jit
def bwd_kernel_dq(
    Q, K, V, L, O, dO, dQ,
    sm_scale,
    seqlen_q, seqlen_k,
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
    pid_i = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    b_idx = pid_bh // H
    h_idx = pid_bh % H
    
    q_base = Q + b_idx * stride_qb + h_idx * stride_qh
    do_base = dO + b_idx * stride_dob + h_idx * stride_doh
    o_base = O + b_idx * stride_ob + h_idx * stride_oh
    
    k_base = K + b_idx * stride_kb + h_idx * stride_kh
    v_base = V + b_idx * stride_vb + h_idx * stride_vh
    
    q_desc = tl.make_tensor_descriptor(
        q_base, shape=[seqlen_q, D_HEAD], strides=[stride_qs, stride_qd],
        block_shape=[BLOCK_M, D_HEAD], padding_option="zero"
    )
    do_desc = tl.make_tensor_descriptor(
        do_base, shape=[seqlen_q, D_HEAD], strides=[stride_dos, stride_dod],
        block_shape=[BLOCK_M, D_HEAD], padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        o_base, shape=[seqlen_q, D_HEAD], strides=[stride_os, stride_od],
        block_shape=[BLOCK_M, D_HEAD], padding_option="zero"
    )
    
    k_desc = tl.make_tensor_descriptor(
        k_base, shape=[seqlen_k, D_HEAD], strides=[stride_ks, stride_kd],
        block_shape=[BLOCK_N, D_HEAD], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        v_base, shape=[seqlen_k, D_HEAD], strides=[stride_vs, stride_vd],
        block_shape=[BLOCK_N, D_HEAD], padding_option="zero"
    )
    
    offset_m = pid_i * BLOCK_M
    
    q = q_desc.load([offset_m, 0])
    do = do_desc.load([offset_m, 0])
    o = o_desc.load([offset_m, 0])
    
    offs_m = offset_m + tl.arange(0, BLOCK_M)
    mask_m = offs_m < seqlen_q
    l_ptrs = L + b_idx * stride_lb + h_idx * stride_lh + offs_m * stride_ls
    l_i = tl.load(l_ptrs, mask=mask_m, other=0.0)
    
    D_i = tl.sum(tl.cast(o, tl.float32) * tl.cast(do, tl.float32), axis=1)
    
    dq = tl.zeros([BLOCK_M, D_HEAD], dtype=tl.float32)
    
    num_n_blocks = tl.cdiv(seqlen_k, BLOCK_N)
    
    for block_n in range(num_n_blocks):
        offset_n = block_n * BLOCK_N
        
        k = k_desc.load([offset_n, 0])
        v = v_desc.load([offset_n, 0])
        
        qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * sm_scale
        
        p = tl.exp(qk - l_i[:, None])
        
        offs_n = offset_n + tl.arange(0, BLOCK_N)
        mask_mn = mask_m[:, None] & (offs_n[None, :] < seqlen_k)
        p = tl.where(mask_mn, p, 0.0)
        
        dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32)
        
        ds = p * (dp - D_i[:, None]) * sm_scale
        
        dq = tl.dot(ds.to(Q.dtype.element_ty), k, acc=dq, out_dtype=tl.float32)
        
    dq_base = dQ + b_idx * stride_dqb + h_idx * stride_dqh
    dq_desc = tl.make_tensor_descriptor(
        dq_base, shape=[seqlen_q, D_HEAD], strides=[stride_dqs, stride_dqd],
        block_shape=[BLOCK_M, D_HEAD]
    )
    dq_desc.store([offset_m, 0], dq.to(dQ.dtype.element_ty))


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=3, num_warps=4),
    ],
    key=['seqlen_q', 'seqlen_k']
)
@triton.jit
def bwd_kernel_dk_dv(
    Q, K, V, L, O, dO, dK, dV,
    sm_scale,
    seqlen_q, seqlen_k,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    stride_lb, stride_lh, stride_ls,
    H,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D_HEAD: tl.constexpr
):
    pid_j = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    b_idx = pid_bh // H
    h_idx = pid_bh % H
    
    k_base = K + b_idx * stride_kb + h_idx * stride_kh
    v_base = V + b_idx * stride_vb + h_idx * stride_vh
    
    k_desc = tl.make_tensor_descriptor(
        k_base, shape=[seqlen_k, D_HEAD], strides=[stride_ks, stride_kd],
        block_shape=[BLOCK_N, D_HEAD], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        v_base, shape=[seqlen_k, D_HEAD], strides=[stride_vs, stride_vd],
        block_shape=[BLOCK_N, D_HEAD], padding_option="zero"
    )
    
    offset_n = pid_j * BLOCK_N
    k = k_desc.load([offset_n, 0])
    v = v_desc.load([offset_n, 0])
    
    dk = tl.zeros([BLOCK_N, D_HEAD], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, D_HEAD], dtype=tl.float32)
    
    q_base = Q + b_idx * stride_qb + h_idx * stride_qh
    do_base = dO + b_idx * stride_dob + h_idx * stride_doh
    o_base = O + b_idx * stride_ob + h_idx * stride_oh
    
    q_desc = tl.make_tensor_descriptor(
        q_base, shape=[seqlen_q, D_HEAD], strides=[stride_qs, stride_qd],
        block_shape=[BLOCK_M, D_HEAD], padding_option="zero"
    )
    do_desc = tl.make_tensor_descriptor(
        do_base, shape=[seqlen_q, D_HEAD], strides=[stride_dos, stride_dod],
        block_shape=[BLOCK_M, D_HEAD], padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        o_base, shape=[seqlen_q, D_HEAD], strides=[stride_os, stride_od],
        block_shape=[BLOCK_M, D_HEAD], padding_option="zero"
    )
    
    offs_n = offset_n + tl.arange(0, BLOCK_N)
    mask_n = offs_n < seqlen_k
    
    num_m_blocks = tl.cdiv(seqlen_q, BLOCK_M)
    
    for block_m in range(num_m_blocks):
        offset_m = block_m * BLOCK_M
        
        q = q_desc.load([offset_m, 0])
        do = do_desc.load([offset_m, 0])
        o = o_desc.load([offset_m, 0])
        
        offs_m = offset_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < seqlen_q
        l_ptrs = L + b_idx * stride_lb + h_idx * stride_lh + offs_m * stride_ls
        l_i = tl.load(l_ptrs, mask=mask_m, other=0.0)
        
        D_i = tl.sum(tl.cast(o, tl.float32) * tl.cast(do, tl.float32), axis=1)
        
        qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * sm_scale
        
        p = tl.exp(qk - l_i[:, None])
        mask_mn = mask_m[:, None] & mask_n[None, :]
        p = tl.where(mask_mn, p, 0.0)
        
        dv = tl.dot(tl.trans(p.to(Q.dtype.element_ty)), do, acc=dv, out_dtype=tl.float32)
        dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32)
        ds = p * (dp - D_i[:, None]) * sm_scale
        dk = tl.dot(tl.trans(ds.to(Q.dtype.element_ty)), q, acc=dk, out_dtype=tl.float32)
        
    dk_base = dK + b_idx * stride_dkb + h_idx * stride_dkh
    dv_base = dV + b_idx * stride_dvb + h_idx * stride_dvh
    
    dk_desc = tl.make_tensor_descriptor(
        dk_base, shape=[seqlen_k, D_HEAD], strides=[stride_dks, stride_dkd],
        block_shape=[BLOCK_N, D_HEAD]
    )
    dv_desc = tl.make_tensor_descriptor(
        dv_base, shape=[seqlen_k, D_HEAD], strides=[stride_dvs, stride_dvd],
        block_shape=[BLOCK_N, D_HEAD]
    )
    
    dk_desc.store([offset_n, 0], dk.to(dK.dtype.element_ty))
    dv_desc.store([offset_n, 0], dv.to(dV.dtype.element_ty))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape

    if S == 0:
        return

    sm_scale = 1.0 / (d ** 0.5)

    stride_lb = L.stride(0)
    stride_lh = L.stride(1)
    stride_ls = L.stride(2)

    grid_dq = lambda META: (triton.cdiv(S, META['BLOCK_M']), B * H)
    bwd_kernel_dq[grid_dq](
        Q, K, V, L, O, dO, dQ,
        sm_scale,
        S, S,
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

    grid_dkdv = lambda META: (triton.cdiv(S, META['BLOCK_N']), B * H)
    bwd_kernel_dk_dv[grid_dkdv](
        Q, K, V, L, O, dO, dK, dV,
        sm_scale,
        S, S,
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