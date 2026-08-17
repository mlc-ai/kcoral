import torch
import triton
import triton.language as tl

def _tma_alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(_tma_alloc_fn)

def get_autotune_configs_dk_dv():
    return [
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
    ]

@triton.autotune(
    configs=get_autotune_configs_dk_dv(),
    key=['seq_len'],
)
@triton.jit
def bwd_kernel_dk_dv(
    Q, K, V, O, dO, L,
    dK, dV,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    seq_len, sm_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    batch_idx = tl.program_id(1)
    head_idx = tl.program_id(2)
    start_n = tl.program_id(0) * BLOCK_N
    
    k_desc = tl.make_tensor_descriptor(
        K + batch_idx * stride_kb + head_idx * stride_kh,
        shape=[seq_len, BLOCK_D], strides=[stride_ks, 1],
        block_shape=[BLOCK_N, BLOCK_D], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        V + batch_idx * stride_vb + head_idx * stride_vh,
        shape=[seq_len, BLOCK_D], strides=[stride_vs, 1],
        block_shape=[BLOCK_N, BLOCK_D], padding_option="zero"
    )
    q_desc = tl.make_tensor_descriptor(
        Q + batch_idx * stride_qb + head_idx * stride_qh,
        shape=[seq_len, BLOCK_D], strides=[stride_qs, 1],
        block_shape=[BLOCK_M, BLOCK_D], padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        O + batch_idx * stride_ob + head_idx * stride_oh,
        shape=[seq_len, BLOCK_D], strides=[stride_os, 1],
        block_shape=[BLOCK_M, BLOCK_D], padding_option="zero"
    )
    do_desc = tl.make_tensor_descriptor(
        dO + batch_idx * stride_dob + head_idx * stride_doh,
        shape=[seq_len, BLOCK_D], strides=[stride_dos, 1],
        block_shape=[BLOCK_M, BLOCK_D], padding_option="zero"
    )

    k = k_desc.load([start_n, 0])
    v = v_desc.load([start_n, 0])
    
    dk = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    
    l_base = L + batch_idx * stride_lb + head_idx * stride_lh
    
    for start_m in range(0, seq_len, BLOCK_M):
        q = q_desc.load([start_m, 0])
        o = o_desc.load([start_m, 0])
        do = do_desc.load([start_m, 0])
        
        offs_m = start_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < seq_len
        l = tl.load(l_base + offs_m * stride_ls, mask=mask_m, other=0.0)
        
        di = tl.sum(do.to(tl.float32) * o.to(tl.float32), axis=1)
        
        kq = tl.dot(k, tl.trans(q))
        kq = kq * sm_scale
        
        p_t = tl.exp(kq - l[None, :])
        
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < seq_len
        p_t = tl.where(mask_n[:, None] & mask_m[None, :], p_t, 0.0)
        
        dv += tl.dot(p_t.to(tl.bfloat16), do)
        
        v_do_t = tl.dot(v, tl.trans(do))
        ds_t = p_t * (v_do_t - di[None, :])
        ds_t = ds_t * sm_scale
        
        dk += tl.dot(ds_t.to(tl.bfloat16), q)
        
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    mask_n = offs_n < seq_len
    
    dk_ptrs = dK + batch_idx * stride_dkb + head_idx * stride_dkh + offs_n[:, None] * stride_dks + offs_d[None, :] * stride_dkd
    dv_ptrs = dV + batch_idx * stride_dvb + head_idx * stride_dvh + offs_n[:, None] * stride_dvs + offs_d[None, :] * stride_dvd
    
    tl.store(dk_ptrs, dk.to(tl.bfloat16), mask=mask_n[:, None])
    tl.store(dv_ptrs, dv.to(tl.bfloat16), mask=mask_n[:, None])


def get_autotune_configs_dq():
    return [
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
    ]

@triton.autotune(
    configs=get_autotune_configs_dq(),
    key=['seq_len'],
)
@triton.jit
def bwd_kernel_dq(
    Q, K, V, O, dO, L,
    dQ,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    seq_len, sm_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    batch_idx = tl.program_id(1)
    head_idx = tl.program_id(2)
    start_m = tl.program_id(0) * BLOCK_M
    
    q_desc = tl.make_tensor_descriptor(
        Q + batch_idx * stride_qb + head_idx * stride_qh,
        shape=[seq_len, BLOCK_D], strides=[stride_qs, 1],
        block_shape=[BLOCK_M, BLOCK_D], padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        O + batch_idx * stride_ob + head_idx * stride_oh,
        shape=[seq_len, BLOCK_D], strides=[stride_os, 1],
        block_shape=[BLOCK_M, BLOCK_D], padding_option="zero"
    )
    do_desc = tl.make_tensor_descriptor(
        dO + batch_idx * stride_dob + head_idx * stride_doh,
        shape=[seq_len, BLOCK_D], strides=[stride_dos, 1],
        block_shape=[BLOCK_M, BLOCK_D], padding_option="zero"
    )
    k_desc = tl.make_tensor_descriptor(
        K + batch_idx * stride_kb + head_idx * stride_kh,
        shape=[seq_len, BLOCK_D], strides=[stride_ks, 1],
        block_shape=[BLOCK_N, BLOCK_D], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        V + batch_idx * stride_vb + head_idx * stride_vh,
        shape=[seq_len, BLOCK_D], strides=[stride_vs, 1],
        block_shape=[BLOCK_N, BLOCK_D], padding_option="zero"
    )

    q = q_desc.load([start_m, 0])
    o = o_desc.load([start_m, 0])
    do = do_desc.load([start_m, 0])
    
    offs_m = start_m + tl.arange(0, BLOCK_M)
    mask_m = offs_m < seq_len
    
    l_base = L + batch_idx * stride_lb + head_idx * stride_lh
    l = tl.load(l_base + offs_m * stride_ls, mask=mask_m, other=0.0)
    
    di = tl.sum(do.to(tl.float32) * o.to(tl.float32), axis=1)
    
    dq = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    for start_n in range(0, seq_len, BLOCK_N):
        k = k_desc.load([start_n, 0])
        v = v_desc.load([start_n, 0])
        
        qk = tl.dot(q, tl.trans(k))
        qk = qk * sm_scale
        
        p = tl.exp(qk - l[:, None])
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < seq_len
        p = tl.where(mask_m[:, None] & mask_n[None, :], p, 0.0)
        
        do_v_t = tl.dot(do, tl.trans(v))
        ds = p * (do_v_t - di[:, None])
        ds = ds * sm_scale
        
        dq += tl.dot(ds.to(tl.bfloat16), k)
        
    offs_d = tl.arange(0, BLOCK_D)
    dq_ptrs = dQ + batch_idx * stride_dqb + head_idx * stride_dqh + offs_m[:, None] * stride_dqs + offs_d[None, :] * stride_dqd
    tl.store(dq_ptrs, dq.to(tl.bfloat16), mask=mask_m[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    sm_scale = 1.0 / (d ** 0.5)

    grid_dk_dv = lambda META: (triton.cdiv(S, META['BLOCK_N']), B, H)
    bwd_kernel_dk_dv[grid_dk_dv](
        Q, K, V, O, dO, L,
        dK, dV,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        S, sm_scale, BLOCK_D=d
    )

    grid_dq = lambda META: (triton.cdiv(S, META['BLOCK_M']), B, H)
    bwd_kernel_dq[grid_dq](
        Q, K, V, O, dO, L,
        dQ,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        S, sm_scale, BLOCK_D=d
    )