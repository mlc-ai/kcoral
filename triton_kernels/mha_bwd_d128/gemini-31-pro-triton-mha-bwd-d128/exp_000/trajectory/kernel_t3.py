import torch
import triton
import triton.language as tl

def _tma_alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(_tma_alloc_fn)

@triton.jit
def zero_tensor_kernel(ptr, n_elements, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    tl.store(ptr + offs, 0.0, mask=mask)

def get_autotune_configs():
    return [
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
    ]

@triton.autotune(
    configs=get_autotune_configs(),
    key=['seq_len'],
)
@triton.jit
def bwd_kernel_single(
    Q, K, V, O, dO, L,
    dQ, dK, dV,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    seq_len, sm_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    tl.assume(seq_len % 64 == 0)
    batch_idx = tl.program_id(1)
    head_idx = tl.program_id(2)
    pid_m = tl.program_id(0)
    start_m = pid_m * BLOCK_M
    
    # TMA descriptors for single loads (Q, O, dO)
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

    q = q_desc.load([start_m, 0])
    o = o_desc.load([start_m, 0])
    do = do_desc.load([start_m, 0])
    
    offs_m = start_m + tl.arange(0, BLOCK_M)
    mask_m = offs_m < seq_len
    
    l_base = L + batch_idx * stride_lb + head_idx * stride_lh
    l = tl.load(l_base + offs_m * stride_ls, mask=mask_m, other=0.0)
    
    # Compute Di directly in FP32
    di = tl.sum(do.to(tl.float32) * o.to(tl.float32), axis=1)
    
    dq = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    # TMA descriptors for iteratively loaded chunks (K, V)
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

    offs_d = tl.arange(0, BLOCK_D)
    num_n_blocks = tl.cdiv(seq_len, BLOCK_N)

    # SW Pipelined loop over N
    # We stagger block starts to drastically reduce `tl.atomic_add` contention for dK & dV.
    for i in tl.range(0, num_n_blocks):
        n_block_idx = (pid_m + i) % num_n_blocks
        start_n = n_block_idx * BLOCK_N
        
        k = k_desc.load([start_n, 0])
        v = v_desc.load([start_n, 0])
        
        qk = tl.dot(q, tl.trans(k))
        qk = qk * sm_scale
        
        p = tl.exp(qk - l[:, None])
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < seq_len
        p = tl.where(mask_m[:, None] & mask_n[None, :], p, 0.0)
        
        dv_n = tl.dot(tl.trans(p.to(tl.bfloat16)), do)
        
        do_v_t = tl.dot(do, tl.trans(v))
        ds = p * (do_v_t - di[:, None])
        ds = ds * sm_scale
        
        dq += tl.dot(ds.to(tl.bfloat16), k)
        dk_n = tl.dot(tl.trans(ds.to(tl.bfloat16)), q)
        
        dv_ptrs = dV + batch_idx * stride_dvb + head_idx * stride_dvh + offs_n[:, None] * stride_dvs + offs_d[None, :] * stride_dvd
        tl.atomic_add(dv_ptrs, dv_n.to(tl.bfloat16), mask=mask_n[:, None], sem="relaxed")
        
        dk_ptrs = dK + batch_idx * stride_dkb + head_idx * stride_dkh + offs_n[:, None] * stride_dks + offs_d[None, :] * stride_dkd
        tl.atomic_add(dk_ptrs, dk_n.to(tl.bfloat16), mask=mask_n[:, None], sem="relaxed")
        
    dq_ptrs = dQ + batch_idx * stride_dqb + head_idx * stride_dqh + offs_m[:, None] * stride_dqs + offs_d[None, :] * stride_dqd
    tl.store(dq_ptrs, dq.to(tl.bfloat16), mask=mask_m[:, None])

def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes Multi-Head Attention backward pass with a highly optimized single-kernel pipeline.
    It avoids 50% of the memory loads compared to typical 2-kernel solutions on Hopper by staggering 
    atomic adds for optimal scaling and avoiding contention.
    """
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    sm_scale = 1.0 / (d ** 0.5)

    n_elements = dK.numel()
    block_size = 1024
    grid_zero = (triton.cdiv(n_elements, block_size),)
    
    zero_tensor_kernel[grid_zero](dK, n_elements, BLOCK=block_size)
    zero_tensor_kernel[grid_zero](dV, n_elements, BLOCK=block_size)

    grid = lambda META: (triton.cdiv(S, META['BLOCK_M']), B, H)
    
    bwd_kernel_single[grid](
        Q, K, V, O, dO, L,
        dQ, dK, dV,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        S, sm_scale, BLOCK_D=d
    )