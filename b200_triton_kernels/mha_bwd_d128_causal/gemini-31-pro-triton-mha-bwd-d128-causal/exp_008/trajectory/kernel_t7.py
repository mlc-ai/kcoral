import math
import torch
import triton
import triton.language as tl

def alloc_fn(size: int, alignment: int, stream):
    """
    Standard allocator strictly for device-created TensorDescriptors used by Hopper TMA.
    """
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)


@triton.jit
def zero_kernel_nd(
    dK, dV, 
    stride_b, stride_h, stride_s, stride_d,
    B, H, S, d: tl.constexpr, 
    BLOCK_S: tl.constexpr
):
    """
    Zeroes out the pre-allocated dK and dV gradient buffers prior to the accumulation pass.
    Safely respects exact multi-dimensional tensor strides.
    """
    pid_s = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    b = pid_bh // H
    h = pid_bh % H
    
    start_s = pid_s * BLOCK_S
    if start_s >= S:
        return
        
    offs_s = start_s + tl.arange(0, BLOCK_S)
    mask_s = offs_s < S
    offs_d = tl.arange(0, d)
    
    ptrs_dk = dK + b * stride_b + h * stride_h + offs_s[:, None] * stride_s + offs_d[None, :] * stride_d
    ptrs_dv = dV + b * stride_b + h * stride_h + offs_s[:, None] * stride_s + offs_d[None, :] * stride_d
    
    zeros = tl.zeros([BLOCK_S, d], dtype=dK.dtype.element_ty)
    
    tl.store(ptrs_dk, zeros, mask=mask_s[:, None])
    tl.store(ptrs_dv, zeros, mask=mask_s[:, None])


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=4),
    ],
    key=["S"],
)
@triton.jit
def bwd_single_kernel_tma(
    Q, K, V, O, dO, L, dQ, dK, dV,
    stride_q_b, stride_q_h, stride_q_s, stride_q_d,
    stride_k_b, stride_k_h, stride_k_s, stride_k_d,
    stride_v_b, stride_v_h, stride_v_s, stride_v_d,
    stride_o_b, stride_o_h, stride_o_s, stride_o_d,
    stride_do_b, stride_do_h, stride_do_s, stride_do_d,
    stride_l_b, stride_l_h, stride_l_s,
    stride_dq_b, stride_dq_h, stride_dq_s, stride_dq_d,
    stride_dk_b, stride_dk_h, stride_dk_s, stride_dk_d,
    stride_dv_b, stride_dv_h, stride_dv_s, stride_dv_d,
    scale,
    B, H, S, d: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """
    Optimized single-kernel architecture avoiding redundant recomputations of D by processing
    Key blocks in the inner loop (WGMMA pipelining). Drastically reduces TMA inner-loop read bandwidth 
    to solely K and V while natively accommodating Hopper's hardware float16/bfloat16 atomics.
    """
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    b = pid_bh // H
    h = pid_bh % H
    
    start_m = pid_m * BLOCK_M
    if start_m >= S:
        return
    
    offs_m = start_m + tl.arange(0, BLOCK_M)
    mask_m = offs_m < S
    
    # 1. Setup Tensor Descriptors for Query Blocks (loaded once per CTA)
    q_desc = tl.make_tensor_descriptor(
        Q + b * stride_q_b + h * stride_q_h,
        shape=[S, d], strides=[stride_q_s, 1],
        block_shape=[BLOCK_M, d], padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        O + b * stride_o_b + h * stride_o_h,
        shape=[S, d], strides=[stride_o_s, 1],
        block_shape=[BLOCK_M, d], padding_option="zero"
    )
    do_desc = tl.make_tensor_descriptor(
        dO + b * stride_do_b + h * stride_do_h,
        shape=[S, d], strides=[stride_do_s, 1],
        block_shape=[BLOCK_M, d], padding_option="zero"
    )
    
    q = tl.load(q_desc, [start_m, 0])
    o = tl.load(o_desc, [start_m, 0])
    do = tl.load(do_desc, [start_m, 0])
    
    l_ptrs = L + b * stride_l_b + h * stride_l_h + offs_m * stride_l_s
    l = tl.load(l_ptrs, mask=mask_m, other=0.0)
    
    # Compute the D vector efficiently directly in SRAM bypassing global memory buffers.
    D = tl.sum(do.to(tl.float32) * o.to(tl.float32), axis=1)
    dq = tl.zeros([BLOCK_M, d], dtype=tl.float32)
    
    # Calculate boundaries separating Causal regions
    limit_n = (start_m // BLOCK_N) * BLOCK_N
    
    k_desc = tl.make_tensor_descriptor(
        K + b * stride_k_b + h * stride_k_h,
        shape=[S, d], strides=[stride_k_s, 1],
        block_shape=[BLOCK_N, d], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        V + b * stride_v_b + h * stride_v_h,
        shape=[S, d], strides=[stride_v_s, 1],
        block_shape=[BLOCK_N, d], padding_option="zero"
    )
    
    offs_d = tl.arange(0, d)
    
    # 2. Iterate dynamically over fully valid segments - strictly no causal masking penalties.
    for start_n in range(0, limit_n, BLOCK_N):
        k = tl.load(k_desc, [start_n, 0])
        v = tl.load(v_desc, [start_n, 0])
        
        s_ij = tl.dot(q, k.T, out_dtype=tl.float32) * scale
        s_ij = tl.where(mask_m[:, None], s_ij, float("-inf"))
        
        p_ij = tl.exp(s_ij - l[:, None])
        p_ij = tl.where(mask_m[:, None], p_ij, 0.0)
        
        dp_ij = tl.dot(do, v.T, out_dtype=tl.float32)
        ds_ij = p_ij * (dp_ij - D[:, None]) * scale
        
        ds_ij_bf16 = ds_ij.to(q.dtype)
        dq += tl.dot(ds_ij_bf16, k, out_dtype=tl.float32)
        
        # Calculate matching dk & dv
        dk = tl.dot(ds_ij_bf16.T, q, out_dtype=tl.float32)
        p_ij_bf16 = p_ij.to(do.dtype)
        dv = tl.dot(p_ij_bf16.T, do, out_dtype=tl.float32)
        
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S
        mask_nd = mask_n[:, None] & (offs_d[None, :] < d)
        
        dk_ptrs = dK + b * stride_dk_b + h * stride_dk_h + offs_n[:, None] * stride_dk_s + offs_d[None, :] * stride_dk_d
        dv_ptrs = dV + b * stride_dv_b + h * stride_dv_h + offs_n[:, None] * stride_dv_s + offs_d[None, :] * stride_dv_d
        
        tl.atomic_add(dk_ptrs, dk.to(tl.bfloat16), mask=mask_nd)
        tl.atomic_add(dv_ptrs, dv.to(tl.bfloat16), mask=mask_nd)

    # 3. Handle strictly confined boundary Block - precisely masking causal barriers.
    end_n = tl.minimum(S, start_m + BLOCK_M)
    for start_n in range(limit_n, end_n, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S
        
        k = tl.load(k_desc, [start_n, 0])
        v = tl.load(v_desc, [start_n, 0])
        
        s_ij = tl.dot(q, k.T, out_dtype=tl.float32) * scale
        
        valid_mask = (offs_m[:, None] >= offs_n[None, :]) & mask_m[:, None] & mask_n[None, :]
        s_ij = tl.where(valid_mask, s_ij, float("-inf"))
        
        p_ij = tl.exp(s_ij - l[:, None])
        p_ij = tl.where(valid_mask, p_ij, 0.0)
        
        dp_ij = tl.dot(do, v.T, out_dtype=tl.float32)
        ds_ij = p_ij * (dp_ij - D[:, None]) * scale
        
        ds_ij_bf16 = ds_ij.to(q.dtype)
        dq += tl.dot(ds_ij_bf16, k, out_dtype=tl.float32)
        
        dk = tl.dot(ds_ij_bf16.T, q, out_dtype=tl.float32)
        p_ij_bf16 = p_ij.to(do.dtype)
        dv = tl.dot(p_ij_bf16.T, do, out_dtype=tl.float32)
        
        mask_nd = mask_n[:, None] & (offs_d[None, :] < d)
        
        dk_ptrs = dK + b * stride_dk_b + h * stride_dk_h + offs_n[:, None] * stride_dk_s + offs_d[None, :] * stride_dk_d
        dv_ptrs = dV + b * stride_dv_b + h * stride_dv_h + offs_n[:, None] * stride_dv_s + offs_d[None, :] * stride_dv_d
        
        tl.atomic_add(dk_ptrs, dk.to(tl.bfloat16), mask=mask_nd)
        tl.atomic_add(dv_ptrs, dv.to(tl.bfloat16), mask=mask_nd)
        
    # 4. Stream isolated dQ results securely utilizing TMA stores.
    dq_desc = tl.make_tensor_descriptor(
        dQ + b * stride_dq_b + h * stride_dq_h,
        shape=[S, d], strides=[stride_dq_s, 1],
        block_shape=[BLOCK_M, d]
    )
    tl.store(dq_desc, [start_m, 0], dq.to(dQ.dtype.element_ty))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Highly Optimized Destination Passing Causal Forward Wrapper
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, d_val = Q.shape
    scale = 1.0 / math.sqrt(d_val)
    
    stride_l_b = L.stride(0)
    stride_l_h = L.stride(1)
    stride_l_s = L.stride(2) if L.dim() >= 3 else 1
    
    # 1. Neutralize memory footprint sequentially ensuring atomic correctness
    grid_zero = (triton.cdiv(S, 64), B * H)
    zero_kernel_nd[grid_zero](
        dK, dV,
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        B, H, S, d_val,
        BLOCK_S=64
    )
    
    # 2. Main Hardware pipelined instruction sequences executing concurrently resolving backprop graphs
    grid = lambda META: (triton.cdiv(S, META["BLOCK_M"]), B * H)
    bwd_single_kernel_tma[grid](
        Q, K, V, O, dO, L, dQ, dK, dV,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        stride_l_b, stride_l_h, stride_l_s,
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        scale,
        B, H, S, d=d_val,
    )
    
    return dQ, dK, dV