import math
import torch
import triton
import triton.language as tl

# Triton requires an allocator for device-created tensor descriptors
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device=torch.cuda.current_device(), dtype=torch.int8)

triton.set_allocator(alloc_fn)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'WARP_SPEC': False}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'WARP_SPEC': True}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'WARP_SPEC': False}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'WARP_SPEC': False}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'WARP_SPEC': False}, num_warps=4, num_stages=4),
    ],
    key=['S']
)
@triton.jit
def bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dQ_ptr,
    stride_qb, stride_qh, stride_qs,
    stride_kb, stride_kh, stride_ks,
    stride_vb, stride_vh, stride_vs,
    stride_ob, stride_oh, stride_os,
    stride_dob, stride_doh, stride_dos,
    stride_lb, stride_lh, stride_ls,
    stride_dqb, stride_dqh, stride_dqs,
    S, alpha, H: tl.constexpr, d: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, WARP_SPEC: tl.constexpr
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
    o_desc = tl.make_tensor_descriptor(
        O_ptr + b * stride_ob + h * stride_oh,
        shape=[S, d], strides=[stride_os, 1],
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
    o = o_desc.load([pid_m * BLOCK_M, 0])
    do = do_desc.load([pid_m * BLOCK_M, 0])
    
    # Pre-reduce per-row norm bounds required for local normalization scale mapping locally
    d_val = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
    
    dq = tl.zeros([BLOCK_M, d], dtype=tl.float32)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < S
    l_ptr = L_ptr + b * stride_lb + h * stride_lh + offs_m * stride_ls
    l = tl.load(l_ptr, mask=mask_m, other=0.0)
    
    n_steps = tl.cdiv(S, BLOCK_N)
    for i in tl.range(0, n_steps, warp_specialize=WARP_SPEC):
        start_n = i * BLOCK_N
        curr_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = curr_n < S
        
        k = k_desc.load([start_n, 0])
        v = v_desc.load([start_n, 0])
        
        # Softmax inner backward components
        s_attn = tl.dot(q, k.T, out_dtype=tl.float32)
        s_attn = s_attn * alpha
        p = tl.exp(s_attn - l[:, None])
        p = tl.where(mask_m[:, None] & mask_n[None, :], p, 0.0)
        
        dp = tl.dot(do, v.T, out_dtype=tl.float32)
        da = p * (dp - d_val[:, None])
        
        dq = tl.dot(da.to(tl.bfloat16), k, acc=dq)
        
    dq = dq * alpha
    
    dq_desc = tl.make_tensor_descriptor(
        dQ_ptr + b * stride_dqb + h * stride_dqh,
        shape=[S, d], strides=[stride_dqs, 1],
        block_shape=[BLOCK_M, d], padding_option="zero"
    )
    dq_desc.store([pid_m * BLOCK_M, 0], dq.to(tl.bfloat16))


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 128, 'BLOCK_M': 64, 'WARP_SPEC': False}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_M': 64, 'WARP_SPEC': True}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 64, 'BLOCK_M': 128, 'WARP_SPEC': False}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 64, 'BLOCK_M': 64, 'WARP_SPEC': False}, num_warps=4, num_stages=4),
    ],
    key=['S']
)
@triton.jit
def bwd_dkv_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dK_ptr, dV_ptr,
    stride_qb, stride_qh, stride_qs,
    stride_kb, stride_kh, stride_ks,
    stride_vb, stride_vh, stride_vs,
    stride_ob, stride_oh, stride_os,
    stride_dob, stride_doh, stride_dos,
    stride_lb, stride_lh, stride_ls,
    stride_dkb, stride_dkh, stride_dks,
    stride_dvb, stride_dvh, stride_dvs,
    S, alpha, H: tl.constexpr, d: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr, WARP_SPEC: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    b = pid_bh // H
    h = pid_bh % H
    
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
    o_desc = tl.make_tensor_descriptor(
        O_ptr + b * stride_ob + h * stride_oh,
        shape=[S, d], strides=[stride_os, 1],
        block_shape=[BLOCK_M, d], padding_option="zero"
    )
    do_desc = tl.make_tensor_descriptor(
        dO_ptr + b * stride_dob + h * stride_doh,
        shape=[S, d], strides=[stride_dos, 1],
        block_shape=[BLOCK_M, d], padding_option="zero"
    )
    
    k = k_desc.load([pid_n * BLOCK_N, 0])
    v = v_desc.load([pid_n * BLOCK_N, 0])
    
    dk = tl.zeros([BLOCK_N, d], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, d], dtype=tl.float32)
    
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < S
    
    n_steps = tl.cdiv(S, BLOCK_M)
    for i in tl.range(0, n_steps, warp_specialize=WARP_SPEC):
        start_m = i * BLOCK_M
        curr_m = start_m + tl.arange(0, BLOCK_M)
        mask_m = curr_m < S
        
        q = q_desc.load([start_m, 0])
        o = o_desc.load([start_m, 0])
        do = do_desc.load([start_m, 0])
        
        l_ptr = L_ptr + b * stride_lb + h * stride_lh + curr_m * stride_ls
        l = tl.load(l_ptr, mask=mask_m, other=0.0)
        
        d_val = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
        
        s_attn = tl.dot(q, k.T, out_dtype=tl.float32)
        s_attn = s_attn * alpha
        p = tl.exp(s_attn - l[:, None])
        p = tl.where(mask_m[:, None] & mask_n[None, :], p, 0.0)
        
        p_bf16 = p.to(tl.bfloat16)
        dv = tl.dot(p_bf16.T, do, acc=dv)
        
        dp = tl.dot(do, v.T, out_dtype=tl.float32)
        da = p * (dp - d_val[:, None])
        
        dk = tl.dot(da.to(tl.bfloat16).T, q, acc=dk)
        
    dk = dk * alpha
    
    dk_desc = tl.make_tensor_descriptor(
        dK_ptr + b * stride_dkb + h * stride_dkh,
        shape=[S, d], strides=[stride_dks, 1],
        block_shape=[BLOCK_N, d], padding_option="zero"
    )
    dv_desc = tl.make_tensor_descriptor(
        dV_ptr + b * stride_dvb + h * stride_dvh,
        shape=[S, d], strides=[stride_dvs, 1],
        block_shape=[BLOCK_N, d], padding_option="zero"
    )
    
    dk_desc.store([pid_n * BLOCK_N, 0], dk.to(tl.bfloat16))
    dv_desc.store([pid_n * BLOCK_N, 0], dv.to(tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes FlashAttention backward pass for sequence blocks without causal mask.
    All inputs and outputs are Bfloat16. Uses fast device Tensor Descriptors for TMA.
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, d = Q.shape
    alpha = 1.0 / math.sqrt(d)
    
    # Process Q gradient sequentially matching local reduction footprints over N keys
    grid_dq = lambda META: (triton.cdiv(S, META['BLOCK_M']), B * H)
    bwd_dq_kernel[grid_dq](
        Q, K, V, O, dO, L, dQ,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        dO.stride(0), dO.stride(1), dO.stride(2),
        L.stride(0), L.stride(1), L.stride(2),
        dQ.stride(0), dQ.stride(1), dQ.stride(2),
        S, alpha, H=H, d=d
    )
    
    # Iteratively resolve K and V tracking accumulated matrix deltas per M rows
    grid_dkv = lambda META: (triton.cdiv(S, META['BLOCK_N']), B * H)
    bwd_dkv_kernel[grid_dkv](
        Q, K, V, O, dO, L, dK, dV,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        dO.stride(0), dO.stride(1), dO.stride(2),
        L.stride(0), L.stride(1), L.stride(2),
        dK.stride(0), dK.stride(1), dK.stride(2),
        dV.stride(0), dV.stride(1), dV.stride(2),
        S, alpha, H=H, d=d
    )
    
    return dQ, dK, dV