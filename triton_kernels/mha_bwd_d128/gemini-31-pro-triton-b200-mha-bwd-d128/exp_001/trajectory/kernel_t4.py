import math
import torch
import triton
import triton.language as tl

# Define the Triton allocator hook explicitly for creating device-side TMA (tensor-descriptor) metadata
def _alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(_alloc_fn)

def get_safe_configs():
    # Safely clamped configurations strictly preserving Blackwell's 228 KiB shared-memory per SM bounds.
    return [
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64},  num_warps=4, num_stages=3),
    ]

@triton.jit
def _zero_kernel(ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Safely zeroes an uninitialized gradient tensor array directly before accumulation operations."""
    offs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    tl.store(ptr + offs, 0.0, mask=mask)


@triton.autotune(configs=get_safe_configs(), key=['S'])
@triton.jit
def bwd_single_kernel_outer_m(
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
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    b = pid_bh // H
    h = pid_bh % H
    
    start_m = pid_m * BLOCK_M
    
    # Outer block dimension descriptors (Stationary over the entire inner loop)
    q_base = Q + b * stride_q_b + h * stride_q_h
    o_base = O + b * stride_o_b + h * stride_o_h
    do_base = dO + b * stride_do_b + h * stride_do_h
    dq_base = dQ + b * stride_dq_b + h * stride_dq_h
    
    k_base = K + b * stride_k_b + h * stride_k_h
    v_base = V + b * stride_v_b + h * stride_v_h
    dk_base = dK + b * stride_dk_b + h * stride_dk_h
    dv_base = dV + b * stride_dv_b + h * stride_dv_h
    
    Q_desc = tl.make_tensor_descriptor(q_base, [S, BLOCK_D], [stride_q_s, stride_q_d], [BLOCK_M, BLOCK_D], "zero")
    O_desc = tl.make_tensor_descriptor(o_base, [S, BLOCK_D], [stride_o_s, stride_o_d], [BLOCK_M, BLOCK_D], "zero")
    dO_desc = tl.make_tensor_descriptor(do_base, [S, BLOCK_D], [stride_do_s, stride_do_d], [BLOCK_M, BLOCK_D], "zero")
    dQ_desc = tl.make_tensor_descriptor(dq_base, [S, BLOCK_D], [stride_dq_s, stride_dq_d], [BLOCK_M, BLOCK_D], "zero")
    
    K_desc = tl.make_tensor_descriptor(k_base, [S, BLOCK_D], [stride_k_s, stride_k_d], [BLOCK_N, BLOCK_D], "zero")
    V_desc = tl.make_tensor_descriptor(v_base, [S, BLOCK_D], [stride_v_s, stride_v_d], [BLOCK_N, BLOCK_D], "zero")
    
    q = Q_desc.load([start_m, 0])
    o = O_desc.load([start_m, 0])
    do = dO_desc.load([start_m, 0])
    
    # Precompute Delta strictly ONCE per Q block. Removes repetitive iteration cost completely.
    delta = tl.sum(do.to(tl.float32) * o.to(tl.float32), axis=1)
    dq = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    
    offs_m = start_m + tl.arange(0, BLOCK_M)
    mask_m = offs_m < S
    safe_offs_m = tl.where(mask_m, offs_m, 0)  # Suppresses out of bounds memory access on pointer logic
    
    l_ptrs = L + b * stride_l_b + h * stride_l_h + safe_offs_m * stride_l_s
    l = tl.load(l_ptrs, mask=mask_m, other=0.0)
    
    offs_d = tl.arange(0, BLOCK_D)
    dk_ptrs_base = dk_base + offs_d[None, :] * stride_dk_d
    dv_ptrs_base = dv_base + offs_d[None, :] * stride_dv_d
    
    for n_idx in tl.range(0, tl.cdiv(S, BLOCK_N)):
        start_n = n_idx * BLOCK_N
        
        k = K_desc.load([start_n, 0])
        v = V_desc.load([start_n, 0])
        
        # P calculation
        qk = tl.dot(q, k.T) * sm_scale
        p = tl.exp(qk - l[:, None])
        
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S
        p = tl.where(mask_m[:, None] & mask_n[None, :], p, 0.0)
        
        # Backward analytical gradients 
        dp = tl.dot(do, v.T) - delta[:, None]
        dp = p * dp
        
        dq = tl.dot(dp.to(tl.bfloat16), k, acc=dq)
        dv_update = tl.dot(p.to(tl.bfloat16).T, do)
        
        dp_scaled = dp * sm_scale
        dk_update = tl.dot(dp_scaled.to(tl.bfloat16).T, q)
        
        safe_offs_n = tl.where(mask_n, offs_n, 0)
        
        dk_ptrs = dk_ptrs_base + safe_offs_n[:, None] * stride_dk_s
        dv_ptrs = dv_ptrs_base + safe_offs_n[:, None] * stride_dv_s
        mask_nd = mask_n[:, None] & (offs_d[None, :] < BLOCK_D)
        
        # Native un-contended L2 bfloat16 hardware Atomics
        tl.atomic_add(dk_ptrs, dk_update.to(tl.bfloat16), mask=mask_nd, sem="relaxed")
        tl.atomic_add(dv_ptrs, dv_update.to(tl.bfloat16), mask=mask_nd, sem="relaxed")
        
    dq *= sm_scale
    dQ_desc.store([start_m, 0], dq.to(q.dtype))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes rigorous non-causal bfloat16 TMA multi-head attention backward pass.
    Writes properly tracked `dQ`, `dK`, `dV` inplace inherently coordinating one execution wave.
    """
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    sm_scale = 1.0 / math.sqrt(d)
    
    # Pre-Initialize destination gradient spaces natively accommodating additive atomic reduction
    block_size = 1024
    grid_zero = lambda meta: (triton.cdiv(dK.numel(), block_size),)
    _zero_kernel[grid_zero](dK, dK.numel(), BLOCK_SIZE=block_size)
    _zero_kernel[grid_zero](dV, dV.numel(), BLOCK_SIZE=block_size)
    
    grid = lambda META: (triton.cdiv(S, META['BLOCK_M']), B * H)
    bwd_single_kernel_outer_m[grid](
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