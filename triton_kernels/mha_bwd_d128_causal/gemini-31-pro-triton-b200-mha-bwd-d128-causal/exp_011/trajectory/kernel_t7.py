import torch
import triton
import triton.language as tl

# Set up allocator for device-side TensorDescriptors mapping natively to TMAs
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

def get_autotune_configs_dq():
    return [
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
    ]

@triton.autotune(
    configs=get_autotune_configs_dq(),
    key=['S']
)
@triton.jit
def _bwd_dq_kernel(
    Q, K, V, O, dO, L, dQ,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    S, sm_scale, H,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    start_m = pid_m * BLOCK_M
    if start_m >= S:
        return

    batch_idx = pid_bh // H
    head_idx = pid_bh % H

    # Advance pointers for current batch/head
    q_ptr = Q + batch_idx * stride_qb + head_idx * stride_qh
    k_ptr = K + batch_idx * stride_kb + head_idx * stride_kh
    v_ptr = V + batch_idx * stride_vb + head_idx * stride_vh
    o_ptr = O + batch_idx * stride_ob + head_idx * stride_oh
    do_ptr = dO + batch_idx * stride_dob + head_idx * stride_doh
    l_ptr = L + batch_idx * stride_lb + head_idx * stride_lh
    dq_ptr = dQ + batch_idx * stride_dqb + head_idx * stride_dqh

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

    m_fully_valid = (start_m + BLOCK_M <= S)
    n_diag_start = start_m // BLOCK_N

    sm_scale_log2 = sm_scale * 1.4426950408889634
    l_log2 = l * 1.4426950408889634

    # 1. Fully valid causal K/V part 
    for n in range(n_diag_start):
        start_n = n * BLOCK_N
        k = k_desc.load([start_n, 0])
        v = v_desc.load([start_n, 0])
        
        qk_log2 = tl.dot(q, k.T) * sm_scale_log2
        p = tl.math.exp2(qk_log2 - l_log2[:, None])
        
        if not m_fully_valid:
            p = tl.where(mask_m[:, None], p, 0.0)
        
        dp = tl.dot(do, v.T)
        ds = p * (dp - delta[:, None])
        
        dq_acc = tl.dot(ds.to(q.dtype), k, acc=dq_acc)

    # 2. Diagonal and boundary part
    max_n = tl.minimum(S, start_m + BLOCK_M)
    n_steps_total = tl.cdiv(max_n, BLOCK_N)
    
    for n in range(n_diag_start, n_steps_total):
        start_n = n * BLOCK_N
        k = k_desc.load([start_n, 0])
        v = v_desc.load([start_n, 0])
        
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S
        n_fully_valid = (start_n + BLOCK_N <= S)
        
        qk_log2 = tl.dot(q, k.T) * sm_scale_log2
        
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        if m_fully_valid and n_fully_valid:
            valid_mask = causal_mask
        else:
            valid_mask = mask_m[:, None] & mask_n[None, :] & causal_mask
            
        qk_log2 = tl.where(valid_mask, qk_log2, float("-inf"))
        p = tl.math.exp2(qk_log2 - l_log2[:, None])
        p = tl.where(valid_mask, p, 0.0)
        
        dp = tl.dot(do, v.T)
        ds = p * (dp - delta[:, None])
        ds = tl.where(valid_mask, ds, 0.0)
        
        dq_acc = tl.dot(ds.to(q.dtype), k, acc=dq_acc)

    dq_acc = dq_acc * sm_scale
    dq_desc.store([start_m, 0], dq_acc.to(q.dtype))


def get_autotune_configs_dkv():
    return [
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
    ]

@triton.autotune(
    configs=get_autotune_configs_dkv(),
    key=['S']
)
@triton.jit
def _bwd_dkv_kernel(
    Q, K, V, O, dO, L, dK, dV,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    S, sm_scale, H,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)

    start_n = pid_n * BLOCK_N
    if start_n >= S:
        return

    batch_idx = pid_bh // H
    head_idx = pid_bh % H

    q_ptr = Q + batch_idx * stride_qb + head_idx * stride_qh
    k_ptr = K + batch_idx * stride_kb + head_idx * stride_kh
    v_ptr = V + batch_idx * stride_vb + head_idx * stride_vh
    o_ptr = O + batch_idx * stride_ob + head_idx * stride_oh
    do_ptr = dO + batch_idx * stride_dob + head_idx * stride_doh
    l_ptr = L + batch_idx * stride_lb + head_idx * stride_lh
    dk_ptr = dK + batch_idx * stride_dkb + head_idx * stride_dkh
    dv_ptr = dV + batch_idx * stride_dvb + head_idx * stride_dvh

    q_desc = tl.make_tensor_descriptor(q_ptr, shape=[S, BLOCK_D], strides=[stride_qs, stride_qd], block_shape=[BLOCK_M, BLOCK_D], padding_option="zero")
    o_desc = tl.make_tensor_descriptor(o_ptr, shape=[S, BLOCK_D], strides=[stride_os, stride_od], block_shape=[BLOCK_M, BLOCK_D], padding_option="zero")
    do_desc = tl.make_tensor_descriptor(do_ptr, shape=[S, BLOCK_D], strides=[stride_dos, stride_dod], block_shape=[BLOCK_M, BLOCK_D], padding_option="zero")
    
    k_desc = tl.make_tensor_descriptor(k_ptr, shape=[S, BLOCK_D], strides=[stride_ks, stride_kd], block_shape=[BLOCK_N, BLOCK_D], padding_option="zero")
    v_desc = tl.make_tensor_descriptor(v_ptr, shape=[S, BLOCK_D], strides=[stride_vs, stride_vd], block_shape=[BLOCK_N, BLOCK_D], padding_option="zero")
    dk_desc = tl.make_tensor_descriptor(dk_ptr, shape=[S, BLOCK_D], strides=[stride_dks, stride_dkd], block_shape=[BLOCK_N, BLOCK_D], padding_option="zero")
    dv_desc = tl.make_tensor_descriptor(dv_ptr, shape=[S, BLOCK_D], strides=[stride_dvs, stride_dvd], block_shape=[BLOCK_N, BLOCK_D], padding_option="zero")

    k = k_desc.load([start_n, 0])
    v = v_desc.load([start_n, 0])

    dk_acc = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    dv_acc = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)

    start_m_initial = (start_n // BLOCK_M) * BLOCK_M
    m_diag_end = (start_n + BLOCK_N + BLOCK_M - 1) // BLOCK_M
    m_steps_total = tl.cdiv(S, BLOCK_M)
    m_diag_end = tl.minimum(m_diag_end, m_steps_total)
    
    offs_n = start_n + tl.arange(0, BLOCK_N)
    mask_n = offs_n < S
    n_fully_valid = (start_n + BLOCK_N <= S)

    sm_scale_log2 = sm_scale * 1.4426950408889634

    # 1. Diagonal part 
    for m_idx in range(start_m_initial // BLOCK_M, m_diag_end):
        start_m = m_idx * BLOCK_M
        q = q_desc.load([start_m, 0])
        o = o_desc.load([start_m, 0])
        do = do_desc.load([start_m, 0])
        
        offs_m = start_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S
        m_fully_valid = (start_m + BLOCK_M <= S)
        
        l = tl.load(l_ptr + offs_m * stride_ls, mask=mask_m, other=0.0)
        l_log2 = l * 1.4426950408889634
        
        delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
        qk_log2 = tl.dot(q, k.T) * sm_scale_log2
        
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        if m_fully_valid and n_fully_valid:
            valid_mask = causal_mask
        else:
            valid_mask = mask_m[:, None] & mask_n[None, :] & causal_mask
            
        qk_log2 = tl.where(valid_mask, qk_log2, float("-inf"))
        p = tl.math.exp2(qk_log2 - l_log2[:, None])
        p = tl.where(valid_mask, p, 0.0)
        
        dp = tl.dot(do, v.T)
        ds = p * (dp - delta[:, None])
        ds = tl.where(valid_mask, ds, 0.0)
        
        dk_acc = tl.dot(ds.to(q.dtype).T, q, acc=dk_acc)
        dv_acc = tl.dot(p.to(q.dtype).T, do, acc=dv_acc)

    # 2. Full causal part (No explicit causal checking inside inner loops)
    m_full_end = m_steps_total - 1 if (S % BLOCK_M != 0) else m_steps_total
    m_full_end = tl.maximum(m_diag_end, m_full_end)
    
    for m_idx in range(m_diag_end, m_full_end):
        start_m = m_idx * BLOCK_M
        q = q_desc.load([start_m, 0])
        o = o_desc.load([start_m, 0])
        do = do_desc.load([start_m, 0])
        
        offs_m = start_m + tl.arange(0, BLOCK_M)
        l = tl.load(l_ptr + offs_m * stride_ls)
        l_log2 = l * 1.4426950408889634
        
        delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
        qk_log2 = tl.dot(q, k.T) * sm_scale_log2
        
        if not n_fully_valid:
            qk_log2 = tl.where(mask_n[None, :], qk_log2, float("-inf"))
            
        p = tl.math.exp2(qk_log2 - l_log2[:, None])
        
        if not n_fully_valid:
            p = tl.where(mask_n[None, :], p, 0.0)
        
        dp = tl.dot(do, v.T)
        ds = p * (dp - delta[:, None])
        
        if not n_fully_valid:
            ds = tl.where(mask_n[None, :], ds, 0.0)
        
        dk_acc = tl.dot(ds.to(q.dtype).T, q, acc=dk_acc)
        dv_acc = tl.dot(p.to(q.dtype).T, do, acc=dv_acc)
        
    # 3. Partial causal boundary part
    if m_full_end < m_steps_total:
        m_idx = m_steps_total - 1
        start_m = m_idx * BLOCK_M
        q = q_desc.load([start_m, 0])
        o = o_desc.load([start_m, 0])
        do = do_desc.load([start_m, 0])
        
        offs_m = start_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S
        l = tl.load(l_ptr + offs_m * stride_ls, mask=mask_m, other=0.0)
        l_log2 = l * 1.4426950408889634
        
        delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
        qk_log2 = tl.dot(q, k.T) * sm_scale_log2
        
        if n_fully_valid:
            valid_mask = mask_m[:, None]
        else:
            valid_mask = mask_m[:, None] & mask_n[None, :]
            
        qk_log2 = tl.where(valid_mask, qk_log2, float("-inf"))
        p = tl.math.exp2(qk_log2 - l_log2[:, None])
        p = tl.where(valid_mask, p, 0.0)
        
        dp = tl.dot(do, v.T)
        ds = p * (dp - delta[:, None])
        ds = tl.where(valid_mask, ds, 0.0)
        
        dk_acc = tl.dot(ds.to(q.dtype).T, q, acc=dk_acc)
        dv_acc = tl.dot(p.to(q.dtype).T, do, acc=dv_acc)

    dk_acc = dk_acc * sm_scale
    dk_desc.store([start_n, 0], dk_acc.to(k.dtype))
    dv_desc.store([start_n, 0], dv_acc.to(v.dtype))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes the backward pass for causal multi-head attention.
    Uses split ownership processing via device-side TensorDescriptors efficiently mapped to native TMA pathways.
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, d = Q.shape
    sm_scale = 1.0 / (d ** 0.5)

    grid_dq = lambda META: (triton.cdiv(S, META['BLOCK_M']), B * H)
    _bwd_dq_kernel[grid_dq](
        Q, K, V, O, dO, L, dQ,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        S, sm_scale, H,
        BLOCK_D=128
    )

    grid_dkv = lambda META: (triton.cdiv(S, META['BLOCK_N']), B * H)
    _bwd_dkv_kernel[grid_dkv](
        Q, K, V, O, dO, L, dK, dV,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        S, sm_scale, H,
        BLOCK_D=128
    )