import torch
import triton
import triton.language as tl

# Configure the infrastructure allocator for Hopper TMA descriptors.
# This does NOT allocate outputs; it only provides backing memory for Triton's device-side descriptors.
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

@triton.heuristics({
    # Compile-time evaluation to completely strip out boundary masking instructions when sequences divide cleanly
    "EVEN_N": lambda args: args["S"] % args["BLOCK_N"] == 0,
    "EVEN_M": lambda args: args["S"] % args["BLOCK_M"] == 0,
})
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=5),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=4),
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
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    
    # Isolate batch and head dimensions 
    b = off_bh // H
    h = off_bh % H
    
    # Establish local matrix base pointers for this block
    q_ptr = Q + b * stride_qb + h * stride_qh
    k_ptr = K + b * stride_kb + h * stride_kh
    v_ptr = V + b * stride_vb + h * stride_vh
    o_ptr = O + b * stride_ob + h * stride_oh
    
    # Build 2-Dimensional Hopper TMA Descriptors. 
    # Zero-padding implicitly bounds-checks all out-of-bounds loads and stores at the hardware level natively.
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
    
    # Pre-fetch Q from TMA. Scaled up-front efficiently so its registers can be fed directly to inner loop ops.
    q = q_desc.load([start_m * BLOCK_M, 0])
    q = (q * qk_scale).to(tl.bfloat16)
    
    # Online Softmax running states
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    offs_n = tl.arange(0, BLOCK_N)
    
    # Software Multi-Stage Pipelined Matrix Iterator
    for start_n in range(0, S, BLOCK_N):
        # 1. Pipeline K and V requests directly from L2 -> Shared Memory via TMA instructions
        k = k_desc.load([start_n, 0])
        v = v_desc.load([start_n, 0])
        
        # 2. Fully fused hardware dot yielding unnormalized block similarities
        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        
        # 3. Compile-time elision prevents this from evaluating when uniformly sized limits guarantee safety
        if not EVEN_N:
            mask = start_n + offs_n < S
            qk = tl.where(mask[None, :], qk, float('-inf'))
            
        # 4. Softmax probability evaluations utilizing ultra-low latency EX2 hardware approximation routines 
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        
        alpha = tl.exp2(m_i - m_ij)
        acc = acc * alpha[:, None]
        
        # 5. Native hardware wgmma vector-accumulation bypassing layout conversions
        acc = tl.dot(p.to(tl.bfloat16), v, acc, out_dtype=tl.float32)
        
        # 6. Commit chunked calculations to main block aggregators
        m_i = m_ij
        l_i = l_i * alpha + l_ij
        
    # Scale finalized attention predictions
    acc = acc / l_i[:, None]
    
    # Transform fast-tracked log2 LSE stats backward into the mathematically identical LogSumExp natural-logs equivalent 
    lse = m_i * 0.6931471805599453 + tl.log(l_i)
    
    # Export standardized blocks. Hopper TMA respects boundary dimensions internally with zero CPU masking overhead
    o_desc.store([start_m * BLOCK_M, 0], acc.to(tl.bfloat16))
    
    # Commit LSE array linearly. Avoid mask generation overhead conditionally if sequences naturally divide bounds 
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    lse_ptrs = LSE + b * stride_lseb + h * stride_lseh + offs_m * stride_lses
    if EVEN_M:
        tl.store(lse_ptrs, lse)
    else:
        tl.store(lse_ptrs, lse, mask=offs_m < S)


def run(Q, K, V, O, LSE):
    """
    Executes a heavily optimized non-causal Flash Attention forward pass seamlessly mapping algorithmics 
    to Tensor Memory Accelerator (TMA) components, pipeline optimizers, L2 Cache streaming layout formats, 
    and fast EX2 base-2 arithmetic accelerators internally on SM90.
    """
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    # Collapse operations beforehand by merging traditional sm_scale onto mathematically robust Base-2 Log(e) coefficients  
    sm_scale = 1.0 / (D ** 0.5)
    log2_e = 1.4426950408889634
    qk_scale = sm_scale * log2_e
    
    # Launch dimensions structure the blocks perfectly for streaming K/V tensors maximally spanning across shared L2 limits natively
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