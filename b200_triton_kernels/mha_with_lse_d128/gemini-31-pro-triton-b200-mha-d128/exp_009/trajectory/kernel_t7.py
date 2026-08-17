import torch
import triton
import triton.language as tl
import math

# Provides device-side backing infrastructure needed for TMA descriptor initializations
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

@triton.autotune(
    configs=[
        # Max-sized tiles optimizing WGMMA throughput directly over sequence lengths
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        
        # Pipelined asymmetric configurations strictly tailored within SM bounds (228 KiB limit)
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=4),
        
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=4),
    ],
    key=["S"],
)
@triton.heuristics({
    "EXACT_N": lambda args: args["S"] % args["BLOCK_N"] == 0,
    "EXACT_M": lambda args: args["S"] % args["BLOCK_M"] == 0,
})
@triton.jit
def _fwd_kernel(
    Q, K, V, O, LSE,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    B, H, S,
    softmax_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
    EXACT_N: tl.constexpr,
    EXACT_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    pid_b_64 = pid_b.to(tl.int64)
    pid_h_64 = pid_h.to(tl.int64)

    # Resolve base pointers logically bound to the current Batch & Head 
    q_base = Q + pid_b_64 * stride_qb + pid_h_64 * stride_qh
    k_base = K + pid_b_64 * stride_kb + pid_h_64 * stride_kh
    v_base = V + pid_b_64 * stride_vb + pid_h_64 * stride_vh
    
    # Device-created hardware TMA Descriptors providing automatic bounds-safe 2D block reads 
    q_desc = tl.make_tensor_descriptor(
        q_base, shape=[S, D], strides=[stride_qs, stride_qd], block_shape=[BLOCK_M, D], padding_option="zero"
    )
    k_desc = tl.make_tensor_descriptor(
        k_base, shape=[S, D], strides=[stride_ks, stride_kd], block_shape=[BLOCK_N, D], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        v_base, shape=[S, D], strides=[stride_vs, stride_vd], block_shape=[BLOCK_N, D], padding_option="zero"
    )

    offset_m = (pid_m * BLOCK_M).to(tl.int32)
    q = q_desc.load([offset_m, 0])
    
    # Pre-scale Q statically & convert to native base-2 dimension mapping
    RCP_LN2: tl.constexpr = 1.4426950408889634
    SCALE = softmax_scale * RCP_LN2
    dtype = q.dtype
    q = (q * SCALE).to(dtype)

    neg_inf = float("-inf")
    m_i = tl.full((BLOCK_M,), neg_inf, tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, D), tl.float32)

    n_tiles = tl.cdiv(S, BLOCK_N)
    
    # Dead-code eliminated internally if strictly bounds aligned
    if not EXACT_N:
        offs_n = tl.arange(0, BLOCK_N)
    
    for kv_tile in range(0, n_tiles):
        offset_n = (kv_tile * BLOCK_N).to(tl.int32)
        
        # Async descriptors
        k = k_desc.load([offset_n, 0])
        v = v_desc.load([offset_n, 0])
        
        # Q @ K^T executing natively supported transpose math mapping in FP32
        scores = tl.dot(q, k.T, out_dtype=tl.float32)
        
        # Boundary masking exclusively compiled out dynamically
        if not EXACT_N:
            curr_offs_n = kv_tile * BLOCK_N + offs_n
            valid_n = curr_offs_n < S
            scores = tl.where(valid_n[None, :], scores, neg_inf)
            
        # Running Normalization components computed cleanly
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        
        alpha = tl.math.exp2(m_i - m_ij)
        p = tl.math.exp2(scores - m_ij[:, None])
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        
        # Fast P @ V 
        acc = tl.dot(p.to(dtype), v, acc, out_dtype=tl.float32)
        m_i = m_ij

    # Reconstruct accurately mathematically valid Natural-Log representations
    LN2: tl.constexpr = 0.6931471805599453
    output = acc / l_i[:, None]
    lse_ln = (m_i + tl.math.log2(l_i)) * LN2

    # Deterministic output configurations bypassing hardware limits accurately
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_m_64 = offs_m.to(tl.int64)
    offs_d_64 = tl.arange(0, D).to(tl.int64)

    o_base = O + pid_b_64 * stride_ob + pid_h_64 * stride_oh
    o_ptrs = o_base + offs_m_64[:, None] * stride_os + offs_d_64[None, :] * stride_od
    
    lse_base = LSE + pid_b_64 * stride_lseb + pid_h_64 * stride_lseh
    lse_ptrs = lse_base + offs_m_64 * stride_lses

    if EXACT_M:
        tl.store(o_ptrs, output.to(dtype))
        tl.store(lse_ptrs, lse_ln)
    else:
        mask_m = offs_m < S
        tl.store(o_ptrs, output.to(dtype), mask=mask_m[:, None])
        tl.store(lse_ptrs, lse_ln, mask=mask_m)


def run(Q, K, V, O, LSE):
    # Enable internal global states
    triton.set_allocator(alloc_fn)
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    softmax_scale = 1.0 / math.sqrt(D)
    
    # Grid organically guarantees structural Swizzling priorities; executing the same Q heads concurrently to multicasters efficiently
    grid = lambda META: (
        triton.cdiv(S, META["BLOCK_M"]),
        B,
        H,
    )
    
    _fwd_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        B, H, S,
        softmax_scale,
        D=D
    )