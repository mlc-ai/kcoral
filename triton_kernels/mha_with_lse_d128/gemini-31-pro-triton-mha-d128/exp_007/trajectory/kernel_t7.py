import torch
import triton
import triton.language as tl

# Standard Triton descriptor allocator required for tl.make_tensor_descriptor on Hopper
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "UNROLL": 2}, num_warps=8, num_stages=3, num_ctas=1),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "UNROLL": 2}, num_warps=8, num_stages=4, num_ctas=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "UNROLL": 4}, num_warps=8, num_stages=4, num_ctas=1),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "UNROLL": 1}, num_warps=8, num_stages=3, num_ctas=1),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "UNROLL": 2}, num_warps=8, num_stages=5, num_ctas=1),
        
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "UNROLL": 1}, num_warps=8, num_stages=3, num_ctas=1),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "UNROLL": 2}, num_warps=8, num_stages=3, num_ctas=2),
        
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "UNROLL": 1}, num_warps=8, num_stages=3, num_ctas=1),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "UNROLL": 2}, num_warps=8, num_stages=3, num_ctas=2),
        
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "UNROLL": 4}, num_warps=8, num_stages=5, num_ctas=1),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "UNROLL": 2}, num_warps=4, num_stages=4, num_ctas=1),
    ],
    key=["S"],
)
@triton.jit
def _fwd_kernel(
    Q, K, V, O, LSE,
    stride_qb, stride_qh, stride_qs,
    stride_kb, stride_kh, stride_ks,
    stride_vb, stride_vh, stride_vs,
    stride_ob, stride_oh, stride_os,
    stride_lseb, stride_lseh, stride_lses,
    S, sm_scale_log2,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    UNROLL: tl.constexpr,
):
    start_m = tl.program_id(0)
    head_id = tl.program_id(1)
    batch_id = tl.program_id(2)

    # Establish pointer offsets to the base of the current batch/head
    q_offset = batch_id * stride_qb + head_id * stride_qh
    k_offset = batch_id * stride_kb + head_id * stride_kh
    v_offset = batch_id * stride_vb + head_id * stride_vh
    o_offset = batch_id * stride_ob + head_id * stride_oh
    lse_offset = batch_id * stride_lseb + head_id * stride_lseh

    # Device-created 2D tensor descriptors leveraging Hopper TMA directly.
    # The innermost stride is statically 1 for the contiguous memory representation.
    q_desc = tl.make_tensor_descriptor(
        Q + q_offset, shape=[S, BLOCK_D], strides=[stride_qs, 1],
        block_shape=[BLOCK_M, BLOCK_D], padding_option="zero"
    )
    k_desc = tl.make_tensor_descriptor(
        K + k_offset, shape=[S, BLOCK_D], strides=[stride_ks, 1],
        block_shape=[BLOCK_N, BLOCK_D], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        V + v_offset, shape=[S, BLOCK_D], strides=[stride_vs, 1],
        block_shape=[BLOCK_N, BLOCK_D], padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        O + o_offset, shape=[S, BLOCK_D], strides=[stride_os, 1],
        block_shape=[BLOCK_M, BLOCK_D]
    )

    # TMA load of Q into shared memory
    q = q_desc.load([start_m * BLOCK_M, 0])
    
    # Mathematical optimization: Pre-scaling Q allows lifting the multiplication fully out of the 
    # inner loop. Moreover, doing this computes onto registers, which coerces the downstream Q @ K^T 
    # into a Hopper RS-GEMM WGMMA execution path that heavily reduces SMEM load bandwidth contention.
    q = (q * sm_scale_log2).to(tl.bfloat16)

    # Standard WGMMA accumulators mapped into FP32 registers natively
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # Separate fully divisible iteration sequences from dynamic bounds tracking to maximize pipeline
    num_steps = S // BLOCK_N
    rem = S % BLOCK_N

    # Hard-unrolling instructions feeds the pipeline deeper ensuring Hopper Multi-Function Units (MUFU) 
    # running exp2 operations can fully hide behind independent WGMMA instructions.
    for i in tl.range(0, num_steps, loop_unroll_factor=UNROLL):
        # Asynchronous loads software-pipelined via num_stages
        k = k_desc.load([i * BLOCK_N, 0])
        v = v_desc.load([i * BLOCK_N, 0])

        # Hardware lowers this directly to RS-GEMM (Register-Shared WGMMA FP32 accumulator)
        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        
        m_ij = tl.max(qk, 1)
        m_i_new = tl.maximum(m_i, m_ij)

        # Map hardware native base-2 exponential evaluation instructions
        alpha = tl.exp2(m_i - m_i_new)
        beta = tl.exp2(qk - m_i_new[:, None])

        l_i_new = alpha * l_i + tl.sum(beta, 1)
        acc = acc * alpha[:, None]
        
        # Hardware lowers this directly to RS-GEMM (Register-Shared WGMMA FP32 accumulator)
        acc = tl.dot(beta.to(tl.bfloat16), v, acc, out_dtype=tl.float32)

        m_i = m_i_new
        l_i = l_i_new

    # Safely evaluate remaining irregular bounds completely decoupled from the mainloop layout
    if rem > 0:
        i = num_steps
        k = k_desc.load([i * BLOCK_N, 0])
        v = v_desc.load([i * BLOCK_N, 0])

        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        
        offs_n = tl.arange(0, BLOCK_N)
        valid_mask = offs_n < rem
        qk = tl.where(valid_mask[None, :], qk, float("-inf"))

        m_ij = tl.max(qk, 1)
        m_i_new = tl.maximum(m_i, m_ij)

        alpha = tl.exp2(m_i - m_i_new)
        beta = tl.exp2(qk - m_i_new[:, None])
        
        l_i_new = alpha * l_i + tl.sum(beta, 1)
        acc = acc * alpha[:, None]
        
        acc = tl.dot(beta.to(tl.bfloat16), v, acc, out_dtype=tl.float32)

        m_i = m_i_new
        l_i = l_i_new

    # Outputs normalization relying on highly pipelined reciprocal hardware instructions
    inv_l_i = 1.0 / l_i
    acc = acc * inv_l_i[:, None]
    
    # Restoring native log scaling mathematically offsets the hardware base 2 mapping
    lse = m_i * 0.6931471805599453 + tl.log(l_i)

    # TMA store clips implicitly out-of-bounds coordinates cleanly due to descriptor logic
    o_desc.store([start_m * BLOCK_M, 0], acc.to(tl.bfloat16))
    
    # Rank-1 generic pointer stores handle scalar outputs correctly
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    lse_ptrs = LSE + lse_offset + offs_m * stride_lses
    tl.store(lse_ptrs, lse, mask=offs_m < S)


def run(Q, K, V, O, LSE):
    """
    Compute non-causal multi-head attention natively exploiting Hopper SM90 TMA descriptors.
    Outputs mathematically equivalent O and LSE formats matching SDPA requirements.
    """
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    if S == 0:
        return

    # Direct log2 scalar pre-processing structure mapping allows utilization of raw `tl.exp2` op natively
    sm_scale = 1.0 / (D ** 0.5)
    LOG2_E = 1.4426950408889634
    sm_scale_log2 = sm_scale * LOG2_E

    # Default grid mapping perfectly binds sequence batching and head alignments intuitively grouping L2 caching
    grid = lambda META: (triton.cdiv(S, META["BLOCK_M"]), H, B)

    _fwd_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        S, sm_scale_log2,
        BLOCK_D=128,
    )