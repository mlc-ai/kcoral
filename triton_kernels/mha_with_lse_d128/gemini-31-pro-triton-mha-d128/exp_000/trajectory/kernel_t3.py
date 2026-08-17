import torch
import triton
import triton.language as tl

# Configure the standard allocator to provide infrastructure storage for Hopper TMA descriptors.
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

@triton.heuristics({
    # Determine at compile-time whether sequence length evenly divides the block sizes.
    # This enables statically removing all inner-loop boundary masking logic.
    "DIVISIBLE_N": lambda args: args["S"] % args["BLOCK_N"] == 0,
    "DIVISIBLE_M": lambda args: args["S"] % args["BLOCK_M"] == 0,
})
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=5),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=5),
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
    DIVISIBLE_M: tl.constexpr,
    DIVISIBLE_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    
    b = off_bh // H
    h = off_bh % H
    
    # Calculate starting pointers for the current Batch and Head
    q_ptr = Q + b * stride_qb + h * stride_qh
    k_ptr = K + b * stride_kb + h * stride_kh
    v_ptr = V + b * stride_vb + h * stride_vh
    o_ptr = O + b * stride_ob + h * stride_oh
    
    # Device-created Hopper TMA Descriptors. TMA automatically handles zero-padding boundaries!
    q_desc = tl.make_tensor_descriptor(
        q_ptr,
        shape=[S, BLOCK_D],
        strides=[stride_qs, stride_qd],
        block_shape=[BLOCK_M, BLOCK_D],
        padding_option="zero"
    )
    k_desc = tl.make_tensor_descriptor(
        k_ptr,
        shape=[S, BLOCK_D],
        strides=[stride_ks, stride_kd],
        block_shape=[BLOCK_N, BLOCK_D],
        padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        v_ptr,
        shape=[S, BLOCK_D],
        strides=[stride_vs, stride_vd],
        block_shape=[BLOCK_N, BLOCK_D],
        padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        o_ptr,
        shape=[S, BLOCK_D],
        strides=[stride_os, stride_od],
        block_shape=[BLOCK_M, BLOCK_D]
    )
    
    # Load Q block and aggressively pre-scale by both `sm_scale` and `log2(e)` 
    # This prepares it for fast base-2 exponential math inside the inner loop
    q = q_desc.load([start_m * BLOCK_M, 0])
    q = (q * qk_scale).to(tl.bfloat16)
    
    # Online Softmax (FlashAttention) statistics
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    offs_n_base = tl.arange(0, BLOCK_N)
    
    # Inner loop over the Sequence Length (N dimension)
    for start_n in range(0, S, BLOCK_N):
        # 1. TMA Load Key and execute natively fused WGMMA dot
        k = k_desc.load([start_n, 0])
        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        
        # 2. Masking is entirely eliminated statically if the sequence is a multiple of BLOCK_N
        if not DIVISIBLE_N:
            mask_n = (start_n + offs_n_base) < S
            qk = tl.where(mask_n[None, :], qk, float('-inf'))
            
        # 3. Softmax Step using high-performance base-2 transcendental hardware instructions
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        
        alpha = tl.exp2(m_i - m_ij)
        acc = acc * alpha[:, None]
        
        # 4. TMA Load Value and inline-accumulate via WGMMA
        v = v_desc.load([start_n, 0])
        acc = tl.dot(p.to(tl.bfloat16), v, acc, out_dtype=tl.float32)
        
        # 5. Commit state
        m_i = m_ij
        l_i = l_i * alpha + l_ij
        
    # Finalize attention output scaling
    acc = acc / l_i[:, None]
    
    # Compute standard natural-log LSE from the base-2 tracked statistics: ln(x) = log2(x) * ln(2)
    lse = (m_i + tl.log2(l_i)) * 0.6931471805599453
    
    # TMA properly ignores boundary rows automatically when executing the block store
    o_desc.store([start_m * BLOCK_M, 0], acc.to(tl.bfloat16))
    
    # Commit LSE array linearly with bounds check
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    lse_ptrs = LSE + b * stride_lseb + h * stride_lseh + offs_m * stride_lses
    if DIVISIBLE_M:
        tl.store(lse_ptrs, lse)
    else:
        tl.store(lse_ptrs, lse, mask=offs_m < S)


def run(Q, K, V, O, LSE):
    """
    Executes a heavily optimized, memory-efficient non-causal Flash Attention 
    forward pass natively utilizing Hopper Tensor Memory Accelerator (TMA) WGMMA features.
    
    Inputs:
        Q, K, V: [B, H, S, 128] bfloat16 tensors.
    Outputs:
        O: Preallocated [B, H, S, 128] bfloat16 tensor.
        LSE: Preallocated [B, H, S] float32 tensor.
    """
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    # Multi-factor scalar prepared sequentially to keep base-2 ops contained cleanly inside the loop
    sm_scale = 1.0 / (D ** 0.5)
    log2_e = 1.4426950408889634
    qk_scale = sm_scale * log2_e
    
    # 2D Grid separating parallelization spaces. 
    # Consecutive blocks across dimension 0 seamlessly maximize head-level L2 Cache retention 
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