import torch
import triton
import triton.language as tl
import math


def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'LOOP_STAGES': 3}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'LOOP_STAGES': 2}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'LOOP_STAGES': 4}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'LOOP_STAGES': 3}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'LOOP_STAGES': 3}, num_warps=4, num_stages=3),
    ],
    key=['S']
)
@triton.jit
def _mha_fwd_kernel(
    Q, K, V, O, LSE,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    B, H, S, sm_scale,
    D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, LOOP_STAGES: tl.constexpr
):
    pid = tl.program_id(0)
    
    num_pid_m = tl.cdiv(S, BLOCK_M)
    num_pid_bh = B * H
    
    # Swizzle Grid to maximize L2 Cache hit rates across shared K and V blocks natively
    GROUP_M = 8
    num_pid_in_group = GROUP_M * num_pid_bh
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    
    pid_m = first_pid_m + (pid % group_size_m)
    pid_bh = (pid % num_pid_in_group) // group_size_m
    
    batch_id = pid_bh // H
    head_id = pid_bh % H
    start_m = pid_m

    # Ignore completely out-of-bounds padded query blocks
    if start_m * BLOCK_M >= S:
        return

    q_base = Q + batch_id * stride_qb + head_id * stride_qh
    k_base = K + batch_id * stride_kb + head_id * stride_kh
    v_base = V + batch_id * stride_vb + head_id * stride_vh
    o_base = O + batch_id * stride_ob + head_id * stride_oh

    # Make physical TMA 2D descriptors 
    q_desc = tl.make_tensor_descriptor(
        q_base, shape=[S, D], strides=[stride_qs, 1], block_shape=[BLOCK_M, D], padding_option="zero"
    )
    k_desc = tl.make_tensor_descriptor(
        k_base, shape=[S, D], strides=[stride_ks, 1], block_shape=[BLOCK_N, D], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        v_base, shape=[S, D], strides=[stride_vs, 1], block_shape=[BLOCK_N, D], padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        o_base, shape=[S, D], strides=[stride_os, 1], block_shape=[BLOCK_M, D], padding_option="zero"
    )

    q = q_desc.load([start_m * BLOCK_M, 0])
    
    # Pre-scale Q completely outside the iterative loop to save repetitive FP32 math operations 
    RCP_LN2 = 1.4426950408889634
    q = (q * sm_scale * RCP_LN2).to(q.dtype)
    
    end_n = tl.minimum(S, (start_m + 1) * BLOCK_M)
    num_n_blocks = tl.cdiv(end_n, BLOCK_N)
    
    # Establish threshold bounds to segregate causal diagonals securely
    num_full_blocks = tl.minimum(S // BLOCK_N, (start_m * BLOCK_M) // BLOCK_N)
    
    m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    # 1. Fast Path: Full blocks entirely protected from causal overlaps
    for block_n_id in tl.range(0, num_full_blocks, num_stages=LOOP_STAGES):
        start_n = block_n_id * BLOCK_N
        k = k_desc.load([start_n, 0])
        v = v_desc.load([start_n, 0])
        
        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        
        alpha = tl.math.exp2(m_i - m_ij)
        p = tl.math.exp2(qk - m_ij[:, None])
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(v.dtype), v, acc)
        m_i = m_ij

    # 2. Careful Path: Bounding causal diagonals & final sequence tails natively
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    
    for block_n_id in range(num_full_blocks, num_n_blocks):
        start_n = block_n_id * BLOCK_N
        k = k_desc.load([start_n, 0])
        v = v_desc.load([start_n, 0])
        
        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        
        n_indices = start_n + offs_n[None, :]
        causal_mask = offs_m[:, None] >= n_indices
        qk = tl.where(causal_mask, qk, -float('inf'))
        
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        
        alpha = tl.math.exp2(m_i - m_ij)
        p = tl.math.exp2(qk - m_ij[:, None])
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(v.dtype), v, acc)
        m_i = m_ij

    # Fast row-normalization division
    rcp_l_i = 1.0 / l_i
    output = acc * rcp_l_i[:, None]
    
    # Convert active base-2 logsumexp back to the mandated natural log scale
    LN2 = 0.6931471805599453
    lse = (m_i + tl.math.log2(l_i)) * LN2
    
    # TMA dynamically bounds memory protection internally (ignores padding stores explicitly)
    o_desc.store([start_m * BLOCK_M, 0], output.to(O.dtype.element_ty))
    
    lse_base = LSE + batch_id * stride_lseb + head_id * stride_lseh
    lse_ptrs = lse_base + offs_m * stride_lses
    tl.store(lse_ptrs, lse, mask=(offs_m < S))


def run(Q, K, V, O, LSE):
    # Establish underlying host allocation necessary for TMA descriptions setup
    triton.set_allocator(alloc_fn)
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    sm_scale = 1.0 / math.sqrt(D)
    
    # Subscribed explicitly scaled grid for Z-order locality L2 caches
    grid = lambda META: (triton.cdiv(S, META['BLOCK_M']) * B * H,)
    
    _mha_fwd_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        B, H, S, sm_scale,
        D=128
    )