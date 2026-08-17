import math
import torch
import triton
import triton.language as tl

# Triton requires an allocator for device-created tensor descriptors
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device=torch.cuda.current_device(), dtype=torch.int8)

triton.set_allocator(alloc_fn)

@triton.jit
def bwd_preprocess_kernel(
    O_ptr, dO_ptr, dQ_ptr,
    stride_ob, stride_oh, stride_os,
    stride_dob, stride_doh, stride_dos,
    stride_dqb, stride_dqh, stride_dqs,
    S, H: tl.constexpr, d: tl.constexpr, BLOCK_M: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    b = pid_bh // H
    h = pid_bh % H
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < S
    offs_d = tl.arange(0, d)
    
    O_ptr_block = O_ptr + b * stride_ob + h * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :]
    dO_ptr_block = dO_ptr + b * stride_dob + h * stride_doh + offs_m[:, None] * stride_dos + offs_d[None, :]
    
    o = tl.load(O_ptr_block, mask=mask_m[:, None], other=0.0)
    do = tl.load(dO_ptr_block, mask=mask_m[:, None], other=0.0)
    
    # Compute row sum(O * dO) natively in float32
    d_val = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
    
    # Safely bit-pack float32 into two bfloat16 elements to store inside dQ
    # This acts as an out-of-band scratchpad to avoid slow allocations.
    d_u32 = d_val.to(tl.uint32, bitcast=True)
    d_hi = (d_u32 >> 16).to(tl.uint16).to(tl.bfloat16, bitcast=True)
    d_lo = (d_u32 & 0xFFFF).to(tl.uint16).to(tl.bfloat16, bitcast=True)
    
    dq_ptr_hi = dQ_ptr + b * stride_dqb + h * stride_dqh + offs_m * stride_dqs + 0
    dq_ptr_lo = dQ_ptr + b * stride_dqb + h * stride_dqh + offs_m * stride_dqs + 1
    
    tl.store(dq_ptr_hi, d_hi, mask=mask_m)
    tl.store(dq_ptr_lo, d_lo, mask=mask_m)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 128, 'BLOCK_M': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_M': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 64, 'BLOCK_M': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 64, 'BLOCK_M': 64}, num_warps=4, num_stages=3),
    ],
    key=['S']
)
@triton.jit
def bwd_dkv_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, dQ_ptr, dK_ptr, dV_ptr,
    stride_qb, stride_qh, stride_qs,
    stride_kb, stride_kh, stride_ks,
    stride_vb, stride_vh, stride_vs,
    stride_dob, stride_doh, stride_dos,
    stride_lb, stride_lh, stride_ls,
    stride_dqb, stride_dqh, stride_dqs,
    stride_dkb, stride_dkh, stride_dks,
    stride_dvb, stride_dvh, stride_dvs,
    S, alpha, H: tl.constexpr, d: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    b = pid_bh // H
    h = pid_bh % H
    
    # Device-created TMEM/TMA robust lowerings
    k_desc = tl.make_tensor_descriptor(
        K_ptr + b * stride_kb + h * stride_kh,
        shape=[S, d], strides=[stride_ks, 1],
        block_shape=[BLOCK_N, d], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        V_ptr + b * stride_vb + h * stride_vh,
        shape=[S, d], strides=[stride_vs, 1],
        block_shape=[BLOCK_N, d], padding_option="zero"
    )
    q_desc = tl.make_tensor_descriptor(
        Q_ptr + b * stride_qb + h * stride_qh,
        shape=[S, d], strides=[stride_qs, 1],
        block_shape=[BLOCK_M, d], padding_option="zero"
    )
    do_desc = tl.make_tensor_descriptor(
        dO_ptr + b * stride_dob + h * stride_doh,
        shape=[S, d], strides=[stride_dos, 1],
        block_shape=[BLOCK_M, d], padding_option="zero"
    )
    
    k = k_desc.load([pid_n * BLOCK_N, 0])
    v = v_desc.load([pid_n * BLOCK_N, 0])
    
    # Scale K outside the loop explicitly for optimal TF32/FP32 bounds
    k_scaled = (k.to(tl.float32) * alpha).to(tl.bfloat16)
    
    dk = tl.zeros([BLOCK_N, d], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, d], dtype=tl.float32)
    
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < S
    
    for start_m in range(0, S, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S
        
        q = q_desc.load([start_m, 0])
        do = do_desc.load([start_m, 0])
        
        l_ptr = L_ptr + b * stride_lb + h * stride_lh + offs_m * stride_ls
        l = tl.load(l_ptr, mask=mask_m, other=0.0)
        
        # Read the stored precomputed d_val sum
        dq_ptr_hi = dQ_ptr + b * stride_dqb + h * stride_dqh + offs_m * stride_dqs + 0
        dq_ptr_lo = dQ_ptr + b * stride_dqb + h * stride_dqh + offs_m * stride_dqs + 1
        
        d_hi = tl.load(dq_ptr_hi, mask=mask_m, other=0.0)
        d_lo = tl.load(dq_ptr_lo, mask=mask_m, other=0.0)
        
        d_hi_u32 = d_hi.to(tl.uint16, bitcast=True).to(tl.uint32)
        d_lo_u32 = d_lo.to(tl.uint16, bitcast=True).to(tl.uint32)
        d_val = ((d_hi_u32 << 16) | d_lo_u32).to(tl.float32, bitcast=True)
        
        s_attn = tl.dot(q, k_scaled.T, out_dtype=tl.float32)
        p = tl.exp(s_attn - l[:, None])
        p = tl.where(mask_m[:, None] & mask_n[None, :], p, 0.0)
        p_bf16 = p.to(tl.bfloat16)
        
        dv = tl.dot(p_bf16.T, do, acc=dv)
        
        dp = tl.dot(do, v.T, out_dtype=tl.float32)
        da = p * (dp - d_val[:, None])
        
        dk = tl.dot(da.to(tl.bfloat16).T, q, acc=dk)
        
    dk = dk * alpha
    
    offs_d = tl.arange(0, d)
    dk_ptr = dK_ptr + b * stride_dkb + h * stride_dkh + offs_n[:, None] * stride_dks + offs_d[None, :]
    dv_ptr = dV_ptr + b * stride_dvb + h * stride_dvh + offs_n[:, None] * stride_dvs + offs_d[None, :]
    
    tl.store(dk_ptr, dk.to(tl.bfloat16), mask=mask_n[:, None])
    tl.store(dv_ptr, dv.to(tl.bfloat16), mask=mask_n[:, None])

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
    ],
    key=['S']
)
@triton.jit
def bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, dQ_ptr,
    stride_qb, stride_qh, stride_qs,
    stride_kb, stride_kh, stride_ks,
    stride_vb, stride_vh, stride_vs,
    stride_dob, stride_doh, stride_dos,
    stride_lb, stride_lh, stride_ls,
    stride_dqb, stride_dqh, stride_dqs,
    S, alpha, H: tl.constexpr, d: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    b = pid_bh // H
    h = pid_bh % H
    
    q_desc = tl.make_tensor_descriptor(
        Q_ptr + b * stride_qb + h * stride_qh,
        shape=[S, d], strides=[stride_qs, 1],
        block_shape=[BLOCK_M, d], padding_option="zero"
    )
    do_desc = tl.make_tensor_descriptor(
        dO_ptr + b * stride_dob + h * stride_doh,
        shape=[S, d], strides=[stride_dos, 1],
        block_shape=[BLOCK_M, d], padding_option="zero"
    )
    k_desc = tl.make_tensor_descriptor(
        K_ptr + b * stride_kb + h * stride_kh,
        shape=[S, d], strides=[stride_ks, 1],
        block_shape=[BLOCK_N, d], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        V_ptr + b * stride_vb + h * stride_vh,
        shape=[S, d], strides=[stride_vs, 1],
        block_shape=[BLOCK_N, d], padding_option="zero"
    )
    
    q = q_desc.load([pid_m * BLOCK_M, 0])
    do = do_desc.load([pid_m * BLOCK_M, 0])
    
    q_scaled = (q.to(tl.float32) * alpha).to(tl.bfloat16)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < S
    
    l_ptr = L_ptr + b * stride_lb + h * stride_lh + offs_m * stride_ls
    l = tl.load(l_ptr, mask=mask_m, other=0.0)
    
    dq_ptr_hi = dQ_ptr + b * stride_dqb + h * stride_dqh + offs_m * stride_dqs + 0
    dq_ptr_lo = dQ_ptr + b * stride_dqb + h * stride_dqh + offs_m * stride_dqs + 1
    
    d_hi = tl.load(dq_ptr_hi, mask=mask_m, other=0.0)
    d_lo = tl.load(dq_ptr_lo, mask=mask_m, other=0.0)
    
    d_hi_u32 = d_hi.to(tl.uint16, bitcast=True).to(tl.uint32)
    d_lo_u32 = d_lo.to(tl.uint16, bitcast=True).to(tl.uint32)
    d_val = ((d_hi_u32 << 16) | d_lo_u32).to(tl.float32, bitcast=True)
    
    dq = tl.zeros([BLOCK_M, d], dtype=tl.float32)
    
    for start_n in range(0, S, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S
        
        k = k_desc.load([start_n, 0])
        v = v_desc.load([start_n, 0])
        
        s_attn = tl.dot(q_scaled, k.T, out_dtype=tl.float32)
        p = tl.exp(s_attn - l[:, None])
        p = tl.where(mask_m[:, None] & mask_n[None, :], p, 0.0)
        
        dp = tl.dot(do, v.T, out_dtype=tl.float32)
        da = p * (dp - d_val[:, None])
        
        dq = tl.dot(da.to(tl.bfloat16), k, acc=dq)
        
    dq = dq * alpha
    
    # Store directly replacing d_val that resided in the first indices
    offs_d = tl.arange(0, d)
    dq_ptr_store = dQ_ptr + b * stride_dqb + h * stride_dqh + offs_m[:, None] * stride_dqs + offs_d[None, :]
    tl.store(dq_ptr_store, dq.to(tl.bfloat16), mask=mask_m[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes FlashAttention backward pass for sequence blocks without causal mask.
    Optimally pipelines TMA/MMA with 128x128 max block sizes and precomputed D statistics.
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, d = Q.shape
    alpha = 1.0 / math.sqrt(d)
    
    # 1. Preprocess D_val and park safely in the dQ array padding indices 
    grid_pre = (triton.cdiv(S, 128), B * H)
    bwd_preprocess_kernel[grid_pre](
        O, dO, dQ,
        O.stride(0), O.stride(1), O.stride(2),
        dO.stride(0), dO.stride(1), dO.stride(2),
        dQ.stride(0), dQ.stride(1), dQ.stride(2),
        S, H, d, BLOCK_M=128
    )
    
    # 2. Iterate Q blocks against parked N K-V boundaries for dense dK / dV derivation
    grid_dkv = lambda META: (triton.cdiv(S, META['BLOCK_N']), B * H)
    bwd_dkv_kernel[grid_dkv](
        Q, K, V, dO, L, dQ, dK, dV,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        dO.stride(0), dO.stride(1), dO.stride(2),
        L.stride(0), L.stride(1), L.stride(2),
        dQ.stride(0), dQ.stride(1), dQ.stride(2),
        dK.stride(0), dK.stride(1), dK.stride(2),
        dV.stride(0), dV.stride(1), dV.stride(2),
        S, alpha, H=H, d=d
    )
    
    # 3. Finally accumulate matching dQ elements against static row states replacing the stored stats
    grid_dq = lambda META: (triton.cdiv(S, META['BLOCK_M']), B * H)
    bwd_dq_kernel[grid_dq](
        Q, K, V, dO, L, dQ,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        dO.stride(0), dO.stride(1), dO.stride(2),
        L.stride(0), L.stride(1), L.stride(2),
        dQ.stride(0), dQ.stride(1), dQ.stride(2),
        S, alpha, H=H, d=d
    )
    
    return dQ, dK, dV