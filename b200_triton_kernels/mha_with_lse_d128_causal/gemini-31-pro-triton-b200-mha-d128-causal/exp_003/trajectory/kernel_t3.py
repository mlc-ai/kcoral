import torch
import triton
import triton.language as tl
import math

# Required configuration for Blackwell device-side TMA descriptor allocation
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "PIPELINE_STAGES": 3}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "PIPELINE_STAGES": 3}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "PIPELINE_STAGES": 4}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "PIPELINE_STAGES": 3}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "PIPELINE_STAGES": 4}, num_stages=4, num_warps=4),
    ],
    key=["S"],
)
@triton.jit
def _attn_fwd_kernel(
    Q, K, V,
    O, LSE,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    sm_scale,
    B, H, S,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
):
    # Enforce safe causality unrolling bounds and bounds-checking optimizations
    tl.static_assert(BLOCK_M % BLOCK_N == 0, "BLOCK_M must be a multiple of BLOCK_N")
    
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    # Base pointers for the current batch and head
    Q_ptr = Q + pid_b * stride_qb + pid_h * stride_qh
    K_ptr = K + pid_b * stride_kb + pid_h * stride_kh
    V_ptr = V + pid_b * stride_vb + pid_h * stride_vh
    O_ptr = O + pid_b * stride_ob + pid_h * stride_oh

    # Create hardware TMA tensor descriptors globally handling out-of-bounds padding automatically
    q_desc = tl.make_tensor_descriptor(
        Q_ptr, shape=[S, D], strides=[stride_qs, stride_qd],
        block_shape=[BLOCK_M, D], padding_option="zero"
    )
    k_desc = tl.make_tensor_descriptor(
        K_ptr, shape=[S, D], strides=[stride_ks, stride_kd],
        block_shape=[BLOCK_N, D], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        V_ptr, shape=[S, D], strides=[stride_vs, stride_vd],
        block_shape=[BLOCK_N, D], padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        O_ptr, shape=[S, D], strides=[stride_os, stride_od],
        block_shape=[BLOCK_M, D], padding_option="zero"
    )

    m_start = pid_m * BLOCK_M
    q = q_desc.load([m_start, 0])
    
    # Pre-calculate base-2 scale mapping directly mapped to SM100 hardware math
    LOG2E = 1.4426950408889634
    scale = sm_scale * LOG2E
    
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    # 1. Fully Unmasked Blocks (Strictly Below Causal Diagonal)
    # Sidesteps bounds and masking overhead perfectly while utilizing asynchronous pipelining
    for start_n in tl.range(0, m_start, BLOCK_N, num_stages=PIPELINE_STAGES):
        k = k_desc.load([start_n, 0])
        v = v_desc.load([start_n, 0])
        
        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        qk = qk * scale
        
        m_ij = tl.max(qk, 1)
        m_new = tl.maximum(m_i, m_ij)
        
        alpha = tl.exp2(m_i - m_new)
        beta = tl.exp2(qk - m_new[:, None])
        
        acc = acc * alpha[:, None]
        
        beta_bf16 = beta.to(tl.bfloat16)
        acc += tl.dot(beta_bf16, v, out_dtype=tl.float32)
        
        l_i = l_i * alpha + tl.sum(beta, 1)
        m_i = m_new

    # 2. Partially Masked Blocks (Strictly the Causal Diagonal)
    # The max bound limits key sequences and bounds execution neatly over causal thresholds
    diag_end = tl.minimum(S, m_start + BLOCK_M)
    for start_n in range(m_start, diag_end, BLOCK_N):
        k = k_desc.load([start_n, 0])
        v = v_desc.load([start_n, 0])
        
        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        qk = qk * scale
        
        # Apply strict causal masking bounds
        offs_m = m_start + tl.arange(0, BLOCK_M)
        offs_n = start_n + tl.arange(0, BLOCK_N)
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        
        qk = tl.where(causal_mask, qk, float("-inf"))
        
        m_ij = tl.max(qk, 1)
        m_new = tl.maximum(m_i, m_ij)
        
        alpha = tl.exp2(m_i - m_new)
        beta = tl.exp2(qk - m_new[:, None])
        
        acc = acc * alpha[:, None]
        
        beta_bf16 = beta.to(tl.bfloat16)
        acc += tl.dot(beta_bf16, v, out_dtype=tl.float32)
        
        l_i = l_i * alpha + tl.sum(beta, 1)
        m_i = m_new

    # Final Softmax Reduction & Projection
    out = acc / l_i[:, None]
    
    # TMA efficiently stores outputs cleanly bounded by descriptor limits, naturally discarding padded out-of-bound `m`
    o_desc.store([m_start, 0], out.to(tl.bfloat16))
    
    # Safely convert internal base-2 scaling back into proper base-e natural log
    LSE_ptr = LSE + pid_b * stride_lseb + pid_h * stride_lseh
    offs_m_store = m_start + tl.arange(0, BLOCK_M)
    LSE_ptrs = LSE_ptr + offs_m_store * stride_lses
    
    LN2 = 0.6931471805599453
    lse_val = (m_i + tl.log2(l_i)) * LN2
    
    tl.store(LSE_ptrs, lse_val, mask=offs_m_store < S)


def run(Q, K, V, O, LSE):
    """
    Computes causal scaled dot product attention and its log-sum-exp (LSE).
    
    Args:
        Q: Query tensor of shape (B, H, S, D) and dtype bfloat16.
        K: Key tensor of shape (B, H, S, D) and dtype bfloat16.
        V: Value tensor of shape (B, H, S, D) and dtype bfloat16.
        O: Preallocated output tensor (B, H, S, D) for the attention output, dtype bfloat16.
        LSE: Preallocated output tensor (B, H, S) for the log-sum-exp, dtype float32.
    """
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    sm_scale = 1.0 / math.sqrt(D)

    # Placing sequence length across Grid-X guarantees Threadblocks accessing
    # matching sequence batches are requested sequentially by HxW layout grid.
    # Because dispatch groups CTAs onto available SMs together, CTAs of the same head 
    # run consecutively providing huge L2 Hit-rate caches for K and V fetching.
    grid = lambda META: (triton.cdiv(S, META["BLOCK_M"]), B * H)
    
    _attn_fwd_kernel[grid](
        Q, K, V,
        O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        sm_scale,
        B, H, S,
        D,
    )