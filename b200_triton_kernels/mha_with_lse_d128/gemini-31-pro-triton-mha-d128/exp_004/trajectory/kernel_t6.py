import torch
import triton
import triton.language as tl

# Define and register the device descriptor allocator required by Hopper standard Triton.
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3, num_ctas=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=4, num_ctas=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=4),
    ],
    key=["S"],
)
@triton.jit
def mha_fwd_kernel(
    Q, K, V, O, LSE,
    sm_scale_log2,
    S,
    H,
    stride_qb, stride_qh, stride_qs,
    stride_kb, stride_kh, stride_ks,
    stride_vb, stride_vh, stride_vs,
    stride_ob, stride_oh, stride_os,
    stride_lseb, stride_lseh, stride_lses,
    IS_EVEN_S: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh % H

    # 16-byte aligned base pointers for each batch/head
    q_ptr = Q + b * stride_qb + h * stride_qh
    k_ptr = K + b * stride_kb + h * stride_kh
    v_ptr = V + b * stride_vb + h * stride_vh
    o_ptr = O + b * stride_ob + h * stride_oh
    lse_ptr = LSE + b * stride_lseb + h * stride_lseh

    # Hopper device-created descriptors for TMA
    q_desc = tl.make_tensor_descriptor(
        q_ptr, shape=[S, BLOCK_D], strides=[stride_qs, 1],
        block_shape=[BLOCK_M, BLOCK_D], padding_option="zero"
    )
    k_desc = tl.make_tensor_descriptor(
        k_ptr, shape=[S, BLOCK_D], strides=[stride_ks, 1],
        block_shape=[BLOCK_N, BLOCK_D], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        v_ptr, shape=[S, BLOCK_D], strides=[stride_vs, 1],
        block_shape=[BLOCK_N, BLOCK_D], padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        o_ptr, shape=[S, BLOCK_D], strides=[stride_os, 1],
        block_shape=[BLOCK_M, BLOCK_D]
    )

    offset_m = pid_m * BLOCK_M
    
    # TMA load directly into shared memory
    q = q_desc.load([offset_m, 0])

    # Accumulator and reduction states
    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.full((BLOCK_M,), 0.0, dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    offs_n = tl.arange(0, BLOCK_N)
    num_n_blocks = tl.cdiv(S, BLOCK_N)
    
    for n_block_idx in tl.range(0, num_n_blocks):
        offset_n = n_block_idx * BLOCK_N
        
        # TMA load using device descriptor, pipelines automatically over tl.range
        k = k_desc.load([offset_n, 0])
        v = v_desc.load([offset_n, 0])
        
        # SS-GEMM: Shared-Shared WGMMA
        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        qk = qk * sm_scale_log2
        
        # Mask out-of-bounds keys only if sequence length is not a multiple of block sizes
        if not IS_EVEN_S:
            if offset_n + BLOCK_N > S:
                mask_n = (offset_n + offs_n) < S
                qk = tl.where(mask_n[None, :], qk, float("-inf"))
        
        # Softmax inner loop optimized to minimize FP32 ALU operations and limit register pressure
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        
        p = tl.exp2(qk - m_new[:, None])
        l_ij = tl.sum(p, axis=1)
        
        alpha = tl.exp2(m_i - m_new)
        l_i = l_i * alpha + l_ij
        
        acc = acc * alpha[:, None]
        p_bf16 = tl.cast(p, tl.bfloat16)
        
        # RS-GEMM: Register-Shared WGMMA
        acc = tl.dot(p_bf16, v, acc, out_dtype=tl.float32)
        
        m_i = m_new

    # Epilogue: reciprocal multiplication is faster than division
    inv_l_i = 1.0 / l_i
    acc = acc * inv_l_i[:, None]
    
    # Recover base-e LSE from scaled base-2 max
    # LSE = max(P) + ln(sum(exp(P - max(P))))
    # Since m_i is tracked as max(P * log2(e)), m_i / log2(e) = max(P)
    # LSE = m_i * ln(2) + ln(l_i) = (m_i + log2(l_i)) * ln(2)
    lse = (m_i + tl.log2(l_i)) * 0.6931471805599453

    # TMA store using device descriptor (automatically ignores out-of-bounds M)
    o_desc.store([offset_m, 0], tl.cast(acc, tl.bfloat16))
    
    # Standard pointer store for LSE
    offs_m = offset_m + tl.arange(0, BLOCK_M)
    lse_ptrs = lse_ptr + offs_m * stride_lses
    if IS_EVEN_S:
        tl.store(lse_ptrs, lse)
    else:
        tl.store(lse_ptrs, lse, mask=(offs_m < S))


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B = Q.shape[0]
    H = Q.shape[1]
    S = Q.shape[2]
    D = Q.shape[3]
    
    # Pre-scale for base-2 exponentiation: sm_scale * log2(e)
    sm_scale = 1.0 / (D ** 0.5)
    sm_scale_log2 = sm_scale * 1.4426950408889634
    
    is_even_s = (S % 128 == 0)
    
    # Grid: (M_blocks, B * H). 
    # Triton iterates over dim 0 (M_blocks) fastest, meaning multiple SMs fetch K/V blocks
    # for the SAME sequence head almost synchronously, achieving exceptional L2 cache reuse.
    grid = lambda META: (triton.cdiv(S, META["BLOCK_M"]), B * H, 1)
    
    mha_fwd_kernel[grid](
        Q, K, V, O, LSE,
        sm_scale_log2,
        S,
        H,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        IS_EVEN_S=is_even_s,
        BLOCK_D=D,
    )