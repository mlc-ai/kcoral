import torch
import triton
import triton.language as tl

# Configure the infrastructure allocator for Hopper TMA descriptors.
# This does NOT allocate outputs; it only provides backing memory for Triton's device-side descriptors.
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

@triton.autotune(
    configs=[
        # Utilize num_warps=8 to aggressively avoid register spilling with large 128x128 tiles on Hopper WGMMA.
        # num_ctas > 1 enables Hopper's TMA Multicast feature, broadcasting K and V across CTAs natively.
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3, num_ctas=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3, num_ctas=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3, num_ctas=1),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=4, num_ctas=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=4, num_ctas=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=4, num_ctas=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=4, num_ctas=4),
    ],
    key=["S"],
)
@triton.jit
def _fwd_kernel(
    Q, K, V, O, LSE,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    S, H,
    qk_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    
    b = off_bh // H
    h = off_bh % H
    
    # Base pointers resolved statically per block
    q_ptr = Q + b * stride_qb + h * stride_qh
    k_ptr = K + b * stride_kb + h * stride_kh
    v_ptr = V + b * stride_vb + h * stride_vh
    o_ptr = O + b * stride_ob + h * stride_oh
    
    # Hopper Device TMA Descriptors 
    # Zero-padding handles bounds seamlessly so boundary masking inside hot paths can be stripped statically.
    q_desc = tl.make_tensor_descriptor(
        q_ptr, shape=[S, BLOCK_D], strides=[stride_qs, stride_qd],
        block_shape=[BLOCK_M, BLOCK_D], padding_option="zero"
    )
    k_desc = tl.make_tensor_descriptor(
        k_ptr, shape=[S, BLOCK_D], strides=[stride_ks, stride_kd],
        block_shape=[BLOCK_N, BLOCK_D], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        v_ptr, shape=[S, BLOCK_D], strides=[stride_vs, stride_vd],
        block_shape=[BLOCK_N, BLOCK_D], padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        o_ptr, shape=[S, BLOCK_D], strides=[stride_os, stride_od],
        block_shape=[BLOCK_M, BLOCK_D]
    )
    
    # Single asynchronous TMA load for Q and scalar multiplication for fused Log2 attention scaling. 
    # Scaled Q rests optimally in WGMMA Core registers.
    q = q_desc.load([start_m * BLOCK_M, 0])
    q = (q * qk_scale).to(tl.bfloat16)
    
    # Online FlashAttention Statistics tracked in base-2 mapping to fast hardware instructions
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    # Precompute boundary sequence dynamically to prevent conditionals running inside the WGMMA loop.
    limit = (S // BLOCK_N) * BLOCK_N
    
    # Hopper Multi-Stage Software Pipelined Matrix Iteration
    for start_n in range(0, S, BLOCK_N):
        # 1. TMA Load and WGMMA natively fused dot for K.T
        k = k_desc.load([start_n, 0])
        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        
        # 2. Only conditionally evaluate boundaries when the sequence length isn't perfectly divisible
        if start_n >= limit:
            offs_n = start_n + tl.arange(0, BLOCK_N)
            qk = tl.where(offs_n[None, :] < S, qk, float('-inf'))
            
        # 3. Base-2 scaled exponential (Hardware native MUFU.EX2 is slightly faster than tl.exp)
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        
        alpha = tl.exp2(m_i - m_ij)
        acc = acc * alpha[:, None]
        
        # 4. TMA Load Value and inline the core WGMMA addition using the native FP32 acc accumulator logic 
        v = v_desc.load([start_n, 0])
        acc = tl.dot(p.to(tl.bfloat16), v, acc, out_dtype=tl.float32)
        
        # 5. Advance
        m_i = m_ij
        l_i = l_i * alpha + l_ij
        
    # Final attention probability scaling 
    acc = acc / l_i[:, None]
    
    # Remap tracked Base-2 stats back to Standard Base-e Natural Logarithmic (LSE): ln(2) ≈ 0.693147 
    lse = m_i * 0.6931471805599453 + tl.log(l_i)
    
    # Zero-padded TMA transparently ignores output bounds
    o_desc.store([start_m * BLOCK_M, 0], acc.to(tl.bfloat16))
    
    # Store standard linear Flash LSE tensor with bounds checked boundary condition
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    lse_ptrs = LSE + b * stride_lseb + h * stride_lseh + offs_m * stride_lses
    if (start_m + 1) * BLOCK_M <= S:
        tl.store(lse_ptrs, lse)
    else:
        tl.store(lse_ptrs, lse, mask=offs_m < S)


def run(Q, K, V, O, LSE):
    """
    Executes a heavily optimized non-causal Flash Attention forward pass leveraging native Hopper
    Tensor Memory Accelerator (TMA), WGMMA pipeline optimizations, L2 Cache Multicasting, 
    and fast EX2 base-2 transcendental hardware ops.
    """
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    sm_scale = 1.0 / (D ** 0.5)
    log2_e = 1.4426950408889634
    qk_scale = sm_scale * log2_e
    
    # 2-Dimensional grid naturally allows for TMA Multicast of K and V dependencies implicitly
    grid = lambda META: (triton.cdiv(S, META["BLOCK_M"]), B * H)
    
    _fwd_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        S, H,
        qk_scale,
        BLOCK_D=128
    )