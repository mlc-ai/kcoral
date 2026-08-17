import math
import torch
import triton
import triton.language as tl

# Standard configuration required to construct device-side TMA descriptors
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=4),
    ],
    key=["S"],
)
@triton.jit
def bwd_dq_kernel_tma(
    Q, K, V, O, dO, L, dQ,
    stride_q_b, stride_q_h, stride_q_s, stride_q_d,
    stride_k_b, stride_k_h, stride_k_s, stride_k_d,
    stride_v_b, stride_v_h, stride_v_s, stride_v_d,
    stride_o_b, stride_o_h, stride_o_s, stride_o_d,
    stride_do_b, stride_do_h, stride_do_s, stride_do_d,
    stride_l_b, stride_l_h, stride_l_s,
    stride_dq_b, stride_dq_h, stride_dq_s, stride_dq_d,
    scale,
    B, H, S, d: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    b = pid_bh // H
    h = pid_bh % H
    
    start_m = pid_m * BLOCK_M
    if start_m >= S:
        return
    
    offs_m = start_m + tl.arange(0, BLOCK_M)
    mask_m = offs_m < S
    
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
    
    q = q_desc.load([start_m, 0])
    o = o_desc.load([start_m, 0])
    do = do_desc.load([start_m, 0])
    
    l_ptrs = L + b * stride_l_b + h * stride_l_h + offs_m * stride_l_s
    l = tl.load(l_ptrs, mask=mask_m, other=0.0)
    
    D = tl.sum(do.to(tl.float32) * o.to(tl.float32), axis=1)
    dq = tl.zeros([BLOCK_M, d], dtype=tl.float32)
    
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
    
    # 1. Fully Valid Blocks - Unrestricted causal processing 
    for start_n in range(0, limit_n, BLOCK_N):
        k = k_desc.load([start_n, 0])
        v = v_desc.load([start_n, 0])
        
        s_ij = tl.dot(q, k.T, out_dtype=tl.float32) * scale
        s_ij = tl.where(mask_m[:, None], s_ij, float("-inf"))
        
        p_ij = tl.exp(s_ij - l[:, None])
        p_ij = tl.where(mask_m[:, None], p_ij, 0.0)
        
        dp_ij = tl.dot(do, v.T, out_dtype=tl.float32)
        ds_ij = p_ij * (dp_ij - D[:, None]) * scale
        
        ds_ij_bf16 = ds_ij.to(q.dtype)
        dq += tl.dot(ds_ij_bf16, k, out_dtype=tl.float32)
        
    # 2. Boundary Block - Processed utilizing explicit mask enforcement constraints
    end_n = tl.minimum(S, start_m + BLOCK_M)
    for start_n in range(limit_n, end_n, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S
        
        k = k_desc.load([start_n, 0])
        v = v_desc.load([start_n, 0])
        
        s_ij = tl.dot(q, k.T, out_dtype=tl.float32) * scale
        
        valid_mask = (offs_m[:, None] >= offs_n[None, :]) & mask_m[:, None] & mask_n[None, :]
        s_ij = tl.where(valid_mask, s_ij, float("-inf"))
        
        p_ij = tl.exp(s_ij - l[:, None])
        p_ij = tl.where(valid_mask, p_ij, 0.0)
        
        dp_ij = tl.dot(do, v.T, out_dtype=tl.float32)
        ds_ij = p_ij * (dp_ij - D[:, None]) * scale
        
        ds_ij_bf16 = ds_ij.to(q.dtype)
        dq += tl.dot(ds_ij_bf16, k, out_dtype=tl.float32)
        
    offs_d = tl.arange(0, d)
    dq_ptrs = dQ + b * stride_dq_b + h * stride_dq_h + offs_m[:, None] * stride_dq_s + offs_d[None, :] * stride_dq_d
    tl.store(dq_ptrs, dq.to(dQ.dtype.element_ty), mask=mask_m[:, None])


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=4),
    ],
    key=["S"],
)
@triton.jit
def bwd_dkdv_kernel_tma(
    Q, K, V, O, dO, L, dK, dV,
    stride_q_b, stride_q_h, stride_q_s, stride_q_d,
    stride_k_b, stride_k_h, stride_k_s, stride_k_d,
    stride_v_b, stride_v_h, stride_v_s, stride_v_d,
    stride_o_b, stride_o_h, stride_o_s, stride_o_d,
    stride_do_b, stride_do_h, stride_do_s, stride_do_d,
    stride_l_b, stride_l_h, stride_l_s,
    stride_dk_b, stride_dk_h, stride_dk_s, stride_dk_d,
    stride_dv_b, stride_dv_h, stride_dv_s, stride_dv_d,
    scale,
    B, H, S, d: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    b = pid_bh // H
    h = pid_bh % H
    
    start_n = pid_n * BLOCK_N
    if start_n >= S:
        return
    
    offs_n = start_n + tl.arange(0, BLOCK_N)
    mask_n = offs_n < S
    offs_d = tl.arange(0, d)
    
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
    
    k = k_desc.load([start_n, 0])
    v = v_desc.load([start_n, 0])
    
    dk = tl.zeros([BLOCK_N, d], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, d], dtype=tl.float32)
    
    start_m_initial = (start_n // BLOCK_M) * BLOCK_M
    limit_m = ((start_n + BLOCK_N + BLOCK_M - 1) // BLOCK_M) * BLOCK_M
    
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
    
    # 1. Boundary Blocks
    for start_m in range(start_m_initial, tl.minimum(limit_m, S), BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S
        
        q = q_desc.load([start_m, 0])
        o = o_desc.load([start_m, 0])
        do = do_desc.load([start_m, 0])
        
        l_ptrs = L + b * stride_l_b + h * stride_l_h + offs_m * stride_l_s
        l = tl.load(l_ptrs, mask=mask_m, other=0.0)
        
        D = tl.sum(do.to(tl.float32) * o.to(tl.float32), axis=1)
        
        s_ji = tl.dot(k, q.T, out_dtype=tl.float32) * scale
        
        valid_mask_ji = (offs_n[:, None] <= offs_m[None, :]) & mask_n[:, None] & mask_m[None, :]
        s_ji = tl.where(valid_mask_ji, s_ji, float("-inf"))
        
        p_ji = tl.exp(s_ji - l[None, :])
        p_ji = tl.where(valid_mask_ji, p_ji, 0.0)
        
        dp_ji = tl.dot(v, do.T, out_dtype=tl.float32)
        ds_ji = p_ji * (dp_ji - D[None, :]) * scale
        
        dk += tl.dot(ds_ji.to(q.dtype), q, out_dtype=tl.float32)
        dv += tl.dot(p_ji.to(do.dtype), do, out_dtype=tl.float32)
        
    # 2. Fully Valid Block Loop
    for start_m in range(limit_m, S, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S
        
        q = q_desc.load([start_m, 0])
        o = o_desc.load([start_m, 0])
        do = do_desc.load([start_m, 0])
        
        l_ptrs = L + b * stride_l_b + h * stride_l_h + offs_m * stride_l_s
        l = tl.load(l_ptrs, mask=mask_m, other=0.0)
        
        D = tl.sum(do.to(tl.float32) * o.to(tl.float32), axis=1)
        
        s_ji = tl.dot(k, q.T, out_dtype=tl.float32) * scale
        
        valid_mask_ji = mask_n[:, None] & mask_m[None, :]
        s_ji = tl.where(valid_mask_ji, s_ji, float("-inf"))
        
        p_ji = tl.exp(s_ji - l[None, :])
        p_ji = tl.where(valid_mask_ji, p_ji, 0.0)
        
        dp_ji = tl.dot(v, do.T, out_dtype=tl.float32)
        ds_ji = p_ji * (dp_ji - D[None, :]) * scale
        
        dk += tl.dot(ds_ji.to(q.dtype), q, out_dtype=tl.float32)
        dv += tl.dot(p_ji.to(do.dtype), do, out_dtype=tl.float32)
        
    dk_ptrs = dK + b * stride_dk_b + h * stride_dk_h + offs_n[:, None] * stride_dk_s + offs_d[None, :] * stride_dk_d
    tl.store(dk_ptrs, dk.to(dK.dtype.element_ty), mask=mask_n[:, None])
    
    dv_ptrs = dV + b * stride_dv_b + h * stride_dv_h + offs_n[:, None] * stride_dv_s + offs_d[None, :] * stride_dv_d
    tl.store(dv_ptrs, dv.to(dV.dtype.element_ty), mask=mask_n[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    
    B, H, S, d_val = Q.shape
    scale = 1.0 / math.sqrt(d_val)
    
    stride_l_b = L.stride(0)
    stride_l_h = L.stride(1)
    stride_l_s = L.stride(2) if L.dim() >= 3 else 1
    
    # Process dQ elements natively on concurrent stream
    grid_dq = lambda META: (triton.cdiv(S, META["BLOCK_M"]), B * H)
    bwd_dq_kernel_tma[grid_dq](
        Q, K, V, O, dO, L, dQ,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        stride_l_b, stride_l_h, stride_l_s,
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        scale,
        B, H, S, d=d_val,
    )
    
    # Process dK and dV elements concurrently on the same stream
    grid_dkdv = lambda META: (triton.cdiv(S, META["BLOCK_N"]), B * H)
    bwd_dkdv_kernel_tma[grid_dkdv](
        Q, K, V, O, dO, L, dK, dV,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        stride_l_b, stride_l_h, stride_l_s,
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        scale,
        B, H, S, d=d_val,
    )
    
    return dQ, dK, dV