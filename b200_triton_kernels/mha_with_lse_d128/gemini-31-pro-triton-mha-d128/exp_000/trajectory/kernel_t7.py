import torch
import triton
import triton.language as tl

# Infrastructure storage for Hopper TMA descriptors.
# This strictly provides backing memory for device-created TensorDescriptors and does not allocate output.
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

@triton.autotune(
    configs=[
        # Core TMA + WGMMA autotune configs spanning optimal tile setups for Hopper SM90 memory hierarchies
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=4),
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
    qk_scale_log2,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    
    b = off_bh // H
    h = off_bh % H
    
    # Establish local matrix base pointers for this block
    q_ptr = Q + b * stride_qb + h * stride_qh
    k_ptr = K + b * stride_kb + h * stride_kh
    v_ptr = V + b * stride_vb + h * stride_vh
    o_ptr = O + b * stride_ob + h * stride_oh
    
    # Device-created 2D Hopper TMA Descriptors. 
    # Zero-padding implicitly bounds-checks all out-of-bounds fetches at the hardware level seamlessly.
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
    
    # Initiate TMA fetch for Query sequence block
    q = q_desc.load([start_m * BLOCK_M, 0])
    
    # Online Softmax running states
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    # Software Multi-Stage Pipelined Matrix Iterator
    for start_n in range(0, S, BLOCK_N):
        # 1. Pipeline K directly from L2 -> Registers via TMA
        k = k_desc.load([start_n, 0])
        
        # 2. Fully fused hardware dot yielding unnormalized block similarities
        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        
        # Inline uniform scaling leveraging precomputed multi-coefficient
        qk = qk * qk_scale_log2
        
        # Unconditional bounds evaluation to guarantee a single basic block, allowing Triton's 
        # MLIR pipeliner to perfectly overlap WGMMA math and TMA loads implicitly.
        offs_n = start_n + tl.arange(0, BLOCK_N)
        qk = tl.where(offs_n[None, :] < S, qk, float('-inf'))
            
        # 3. Softmax evaluations utilizing ultra-low latency Hopper MUFU.EX2 hardware accelerators 
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        
        alpha = tl.exp2(m_i - m_ij)
        acc = acc * alpha[:, None]
        
        # 4. Fetch Value matrix and native hardware WGMMA accumulate bypassing overheads
        v = v_desc.load([start_n, 0])
        acc = tl.dot(p.to(tl.bfloat16), v, acc=acc, out_dtype=tl.float32)
        
        # 5. Commit chunked calculations to main block aggregators
        m_i = m_ij
        l_i = l_i * alpha + l_ij
        
    # Scale finalized attention predictions probabilities
    acc = acc / l_i[:, None]
    
    # Transform fast-tracked log2 LSE stats backward into the mathematically identical Natural-Log form
    lse = (m_i + tl.log2(l_i)) * 0.6931471805599453
    
    # Export standardized blocks. Hopper TMA natively respects outer dimension boundaries.
    o_desc.store([start_m * BLOCK_M, 0], acc.to(tl.bfloat16))
    
    # Commit linear LSE statistics utilizing conditional masking purely externally avoiding pipeline stalls
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    lse_ptrs = LSE + b * stride_lseb + h * stride_lseh + offs_m * stride_lses
    
    limit_m = (S // BLOCK_M) * BLOCK_M
    if start_m * BLOCK_M >= limit_m:
        tl.store(lse_ptrs, lse, mask=offs_m < S)
    else:
        tl.store(lse_ptrs, lse)


def run(Q, K, V, O, LSE):
    """
    Executes a heavily optimized non-causal Flash Attention forward pass seamlessly mapping 
    algorithmics to Hopper Tensor Memory Accelerator (TMA), pipeline optimizers, streaming 
    L2 Cache broadcast grids, and fast EX2 base-2 arithmetic hardware internally on SM90.
    """
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    # Collapse operations beforehand by merging traditional sm_scale onto robust Base-2 Log(e) coefficients  
    sm_scale = 1.0 / (D ** 0.5)
    log2_e = 1.4426950408889634
    qk_scale_log2 = sm_scale * log2_e
    
    # Launch dimensions structure blocks optimally. Scheduling contiguous M dimensions across the X axis
    # implicitly causes them to run concurrently for the same head, allowing multicast hardware to source
    # K and V directly from L2 caching without HBM reads.
    grid = lambda META: (triton.cdiv(S, META["BLOCK_M"]), B * H)
    
    _fwd_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        S, H,
        qk_scale_log2,
        BLOCK_D=128
    )