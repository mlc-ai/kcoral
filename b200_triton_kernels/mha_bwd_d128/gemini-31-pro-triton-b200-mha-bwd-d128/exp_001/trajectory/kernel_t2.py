import math
import torch
import triton
import triton.language as tl

# Register the standard host allocator required for creating device-side TMA descriptors safely
def _alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(_alloc_fn)

def get_single_configs():
    return [
        # Blackwell tuning parameters targeting memory staging constraints and warp specialization
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'NUM_STAGES': 2, 'WARP_SPECIALIZE': False}, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'NUM_STAGES': 3, 'WARP_SPECIALIZE': True},  num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'NUM_STAGES': 3, 'WARP_SPECIALIZE': False}, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'NUM_STAGES': 3, 'WARP_SPECIALIZE': False}, num_warps=4),
    ]

@triton.jit
def _zero_kernel(ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Safely zeroes an uninitialized tensor array entirely within Triton."""
    offs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    tl.store(ptr + offs, 0.0, mask=mask)

@triton.autotune(configs=get_single_configs(), key=['S'])
@triton.jit
def bwd_single_kernel_tma(
    Q, K, V, O, dO, dQ, dK, dV, L,
    stride_q_b, stride_q_h, stride_q_s, stride_q_d,
    stride_k_b, stride_k_h, stride_k_s, stride_k_d,
    stride_v_b, stride_v_h, stride_v_s, stride_v_d,
    stride_o_b, stride_o_h, stride_o_s, stride_o_d,
    stride_do_b, stride_do_h, stride_do_s, stride_do_d,
    stride_dq_b, stride_dq_h, stride_dq_s, stride_dq_d,
    stride_dk_b, stride_dk_h, stride_dk_s, stride_dk_d,
    stride_dv_b, stride_dv_h, stride_dv_s, stride_dv_d,
    stride_l_b, stride_l_h, stride_l_s,
    S, sm_scale, H,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    NUM_STAGES: tl.constexpr, WARP_SPECIALIZE: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    
    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    mask_m = offs_m < S
    
    q_base = Q + b * stride_q_b + h * stride_q_h
    o_base = O + b * stride_o_b + h * stride_o_h
    do_base = dO + b * stride_do_b + h * stride_do_h
    k_base = K + b * stride_k_b + h * stride_k_h
    v_base = V + b * stride_v_b + h * stride_v_h
    dq_base = dQ + b * stride_dq_b + h * stride_dq_h
    
    # Outer block dimension descriptors
    Q_desc = tl.make_tensor_descriptor(
        q_base, shape=[S, BLOCK_D], strides=[stride_q_s, stride_q_d],
        block_shape=[BLOCK_M, BLOCK_D], padding_option="zero"
    )
    O_desc = tl.make_tensor_descriptor(
        o_base, shape=[S, BLOCK_D], strides=[stride_o_s, stride_o_d],
        block_shape=[BLOCK_M, BLOCK_D], padding_option="zero"
    )
    dO_desc = tl.make_tensor_descriptor(
        do_base, shape=[S, BLOCK_D], strides=[stride_do_s, stride_do_d],
        block_shape=[BLOCK_M, BLOCK_D], padding_option="zero"
    )
    
    q = Q_desc.load([start_m, 0])
    o = O_desc.load([start_m, 0])
    do = dO_desc.load([start_m, 0])
    
    # Precompute Delta term strictly once per Q block outside the inner K/V iteration
    delta = tl.sum(do.to(tl.float32) * o.to(tl.float32), axis=1)
    dq = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    
    l_ptrs = L + b * stride_l_b + h * stride_l_h + offs_m * stride_l_s
    l = tl.load(l_ptrs, mask=mask_m, other=0.0)
    
    # K and V TMA descriptors for inner looping
    K_desc = tl.make_tensor_descriptor(
        k_base, shape=[S, BLOCK_D], strides=[stride_k_s, stride_k_d],
        block_shape=[BLOCK_N, BLOCK_D], padding_option="zero"
    )
    V_desc = tl.make_tensor_descriptor(
        v_base, shape=[S, BLOCK_D], strides=[stride_v_s, stride_v_d],
        block_shape=[BLOCK_N, BLOCK_D], padding_option="zero"
    )
    
    # Pointer bases targeted for accumulated atomic_adds per K/V loop
    dk_base_ptr = dK + b * stride_dk_b + h * stride_dk_h + offs_d[None, :] * stride_dk_d
    dv_base_ptr = dV + b * stride_dv_b + h * stride_dv_h + offs_d[None, :] * stride_dv_d
    
    total_n_blocks = tl.cdiv(S, BLOCK_N)
    
    # Sequence N reduction mapping
    for n_idx in tl.range(0, total_n_blocks, num_stages=NUM_STAGES, warp_specialize=WARP_SPECIALIZE):
        start_n = n_idx * BLOCK_N
        
        k = K_desc.load([start_n, 0])
        v = V_desc.load([start_n, 0])
        
        qk = tl.dot(q, k.T) * sm_scale
        p = tl.exp(qk - l[:, None])
        
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S
        p = tl.where(mask_m[:, None] & mask_n[None, :], p, 0.0)
        
        # Backward gradients propagation via optimized math grouping
        dv_update = tl.dot(p.to(q.dtype).T, do)
        
        dp = tl.dot(do, v.T) - delta[:, None]
        dp = p * dp
        dp_scaled = dp * sm_scale
        
        dq = tl.dot(dp_scaled.to(q.dtype), k, acc=dq)
        dk_update = tl.dot(dp_scaled.to(q.dtype).T, q)
        
        # Resolving Pointers mapped safely internally ensuring exact shape projection
        dk_ptrs = dk_base_ptr + offs_n[:, None] * stride_dk_s
        dv_ptrs = dv_base_ptr + offs_n[:, None] * stride_dv_s
        mask_nd = mask_n[:, None] & (offs_d[None, :] < BLOCK_D)
        
        # Native Hardware bfloat16 accumulated updates seamlessly handle block contention
        tl.atomic_add(dk_ptrs, dk_update.to(tl.bfloat16), mask=mask_nd, sem="relaxed")
        tl.atomic_add(dv_ptrs, dv_update.to(tl.bfloat16), mask=mask_nd, sem="relaxed")
        
    dQ_desc = tl.make_tensor_descriptor(
        dq_base, shape=[S, BLOCK_D], strides=[stride_dq_s, stride_dq_d],
        block_shape=[BLOCK_M, BLOCK_D], padding_option="zero"
    )
    dQ_desc.store([start_m, 0], dq.to(q.dtype))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Solves non-causal bfloat16 Multi-head Attention Backward.
    Efficiently reduces Math logic natively through a unified TMA execution.
    """
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    
    # 1. Re-initialize raw gradient buffers robustly
    n_elements_dk = dK.numel()
    n_elements_dv = dV.numel()
    block_size = 1024
    grid_zero_dk = (triton.cdiv(n_elements_dk, block_size),)
    grid_zero_dv = (triton.cdiv(n_elements_dv, block_size),)
    
    _zero_kernel[grid_zero_dk](dK, n_elements_dk, BLOCK_SIZE=block_size)
    _zero_kernel[grid_zero_dv](dV, n_elements_dv, BLOCK_SIZE=block_size)
    
    sm_scale = 1.0 / math.sqrt(d)
    
    # 2. Main Backward execution mapping
    grid = lambda META: (triton.cdiv(S, META['BLOCK_M']), B * H)
    bwd_single_kernel_tma[grid](
        Q, K, V, O, dO, dQ, dK, dV, L,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        S, sm_scale, H,
        BLOCK_D=128,
    )