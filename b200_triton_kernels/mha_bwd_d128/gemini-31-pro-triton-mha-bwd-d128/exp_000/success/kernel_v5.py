import torch
import triton
import triton.language as tl

def _tma_alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(_tma_alloc_fn)

def get_autotune_configs_dk_dv():
    return [
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=4),
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
    
    # Outer loop descriptors (loaded once per block)
    k_desc = tl.make_tensor_descriptor(
        K + batch_idx * stride_kb + head_idx * stride_kh,
        shape=[seq_len, BLOCK_D], strides=[stride_ks, stride_kd],
        block_shape=[BLOCK_N, BLOCK_D], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        V + batch_idx * stride_vb + head_idx * stride_vh,
        shape=[seq_len, BLOCK_D], strides=[stride_vs, stride_vd],
        block_shape=[BLOCK_N, BLOCK_D], padding_option="zero"
    )
    
    # Inner loop descriptors (pipelined)
    q_desc = tl.make_tensor_descriptor(
        Q + batch_idx * stride_qb + head_idx * stride_qh,
        shape=[seq_len, BLOCK_D], strides=[stride_qs, stride_qd],
        block_shape=[BLOCK_M, BLOCK_D], padding_option="zero"
    )
    do_desc = tl.make_tensor_descriptor(
        dO + batch_idx * stride_dob + head_idx * stride_doh,
        shape=[seq_len, BLOCK_D], strides=[stride_dos, stride_dod],
        block_shape=[BLOCK_M, BLOCK_D], padding_option="zero"
    )

    k = k_desc.load([start_n, 0])
    v = v_desc.load([start_n, 0])
    
    dk = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    
    offs_d = tl.arange(0, BLOCK_D)
    
    # Pre-compute pointer bases to avoid SMEM staging of `O` tensor
    # Direct pointer load reduces SMEM usage by 33%, keeping us safely within the Hopper 228KB limit
    o_ptrs_base = O + batch_idx * stride_ob + head_idx * stride_oh + offs_d[None, :] * stride_od
    l_base = L + batch_idx * stride_lb + head_idx * stride_lh
    
    mask_n = (start_n + tl.arange(0, BLOCK_N)) < seq_len
    
    log2_e = 1.4426950408889634
    sm_scale_log2 = sm_scale * log2_e
    
    for start_m in range(0, seq_len, BLOCK_M):
        q = q_desc.load([start_m, 0])
        do = do_desc.load([start_m, 0])
        
        offs_m = start_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < seq_len
        
        o_ptrs = o_ptrs_base + offs_m[:, None] * stride_os
        o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
        
        l = tl.load(l_base + offs_m * stride_ls, mask=mask_m, other=float('inf'))
        l_log2 = l * log2_e
        
        di = tl.sum(do.to(tl.float32) * o.to(tl.float32), axis=1)
        
        # Perfect b.T match for Hopper WGMMA execution
        kq = tl.dot(k, tl.trans(q))
        
        p_t = tl.exp2(kq * sm_scale_log2 - l_log2[None, :])
        p_t = tl.where(mask_n[:, None], p_t, 0.0)
        
        dv += tl.dot(p_t.to(tl.bfloat16), do)
        
        v_do_t = tl.dot(v, tl.trans(do))
        ds_t = p_t * (v_do_t - di[None, :]) * sm_scale
        
        dk += tl.dot(ds_t.to(tl.bfloat16), q)
        
    dk_desc = tl.make_tensor_descriptor(
        dK + batch_idx * stride_dkb + head_idx * stride_dkh,
        shape=[seq_len, BLOCK_D], strides=[stride_dks, stride_dkd],
        block_shape=[BLOCK_N, BLOCK_D]
    )
    dv_desc = tl.make_tensor_descriptor(
        dV + batch_idx * stride_dvb + head_idx * stride_dvh,
        shape=[seq_len, BLOCK_D], strides=[stride_dvs, stride_dvd],
        block_shape=[BLOCK_N, BLOCK_D]
    )
    dk_desc.store([start_n, 0], dk.to(tl.bfloat16))
    dv_desc.store([start_n, 0], dv.to(tl.bfloat16))


def get_autotune_configs_dq():
    return [
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
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
    
    # Outer loop descriptors
    q_desc = tl.make_tensor_descriptor(
        Q + batch_idx * stride_qb + head_idx * stride_qh,
        shape=[seq_len, BLOCK_D], strides=[stride_qs, stride_qd],
        block_shape=[BLOCK_M, BLOCK_D], padding_option="zero"
    )
    do_desc = tl.make_tensor_descriptor(
        dO + batch_idx * stride_dob + head_idx * stride_doh,
        shape=[seq_len, BLOCK_D], strides=[stride_dos, stride_dod],
        block_shape=[BLOCK_M, BLOCK_D], padding_option="zero"
    )
    
    # Inner loop descriptors (pipelined)
    k_desc = tl.make_tensor_descriptor(
        K + batch_idx * stride_kb + head_idx * stride_kh,
        shape=[seq_len, BLOCK_D], strides=[stride_ks, stride_kd],
        block_shape=[BLOCK_N, BLOCK_D], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        V + batch_idx * stride_vb + head_idx * stride_vh,
        shape=[seq_len, BLOCK_D], strides=[stride_vs, stride_vd],
        block_shape=[BLOCK_N, BLOCK_D], padding_option="zero"
    )

    q = q_desc.load([start_m, 0])
    do = do_desc.load([start_m, 0])
    
    offs_m = start_m + tl.arange(0, BLOCK_M)
    mask_m = offs_m < seq_len
    offs_d = tl.arange(0, BLOCK_D)
    
    o_ptrs_base = O + batch_idx * stride_ob + head_idx * stride_oh + offs_d[None, :] * stride_od
    o_ptrs = o_ptrs_base + offs_m[:, None] * stride_os
    o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
    
    l_base = L + batch_idx * stride_lb + head_idx * stride_lh
    l = tl.load(l_base + offs_m * stride_ls, mask=mask_m, other=float('inf'))
    
    # Computed once per program since O and dO are invariant to the N loop
    di = tl.sum(do.to(tl.float32) * o.to(tl.float32), axis=1)
    
    dq = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    log2_e = 1.4426950408889634
    sm_scale_log2 = sm_scale * log2_e
    l_log2 = l * log2_e
    
    for start_n in range(0, seq_len, BLOCK_N):
        k = k_desc.load([start_n, 0])
        v = v_desc.load([start_n, 0])
        
        qk = tl.dot(q, tl.trans(k))
        
        p = tl.exp2(qk * sm_scale_log2 - l_log2[:, None])
        mask_n = (start_n + tl.arange(0, BLOCK_N)) < seq_len
        p = tl.where(mask_n[None, :], p, 0.0)
        
        do_v_t = tl.dot(do, tl.trans(v))
        ds = p * (do_v_t - di[:, None]) * sm_scale
        
        dq += tl.dot(ds.to(tl.bfloat16), k)
        
    dq_desc = tl.make_tensor_descriptor(
        dQ + batch_idx * stride_dqb + head_idx * stride_dqh,
        shape=[seq_len, BLOCK_D], strides=[stride_dqs, stride_dqd],
        block_shape=[BLOCK_M, BLOCK_D]
    )
    dq_desc.store([start_m, 0], dq.to(tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Highly optimized multi-head attention backward relying on WGMMA Tensor Core instructions
    and TMA asynchronous pipelined loads. The kernels are structured exactly to respect SM90 rules,
    avoid memory and compute stalls, and avoid SMEM allocation limits causing pass manager crashes.
    """
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