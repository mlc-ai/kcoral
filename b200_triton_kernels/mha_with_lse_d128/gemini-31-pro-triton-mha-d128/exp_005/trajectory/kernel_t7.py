import torch
import triton
import triton.language as tl

# Configure Triton's descriptor allocator to allow device-side tl.make_tensor_descriptor
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)


@triton.autotune(
    configs=[
        # Peak SMEM efficiency for WGMMA and TMA
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3, num_ctas=1),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3, num_ctas=2),
        
        # Deeply pipelined smaller-N configs for memory latency hiding
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=6, num_ctas=1),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=5, num_ctas=1),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=6, num_ctas=1),
        
        # Smaller-M configs
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=3, num_ctas=1),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=3, num_ctas=1),
    ],
    key=["S"],
)
@triton.jit
def _fwd_kernel_tma_optimized(
    Q, K, V, O, LSE,
    stride_qb, stride_qh, stride_qs,
    stride_kb, stride_kh, stride_ks,
    stride_vb, stride_vh, stride_vs,
    stride_ob, stride_oh, stride_os,
    stride_lseb, stride_lseh, stride_lses,
    S,
    H: tl.constexpr, D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    b = off_hz // H
    h = off_hz % H

    # Base pointers for the current head
    q_base = Q + b * stride_qb + h * stride_qh
    k_base = K + b * stride_kb + h * stride_kh
    v_base = V + b * stride_vb + h * stride_vh
    o_base = O + b * stride_ob + h * stride_oh

    # Construct Hopper native TMA Descriptors guaranteeing fully coalesced asynchronous fetches
    q_desc = tl.make_tensor_descriptor(
        q_base, shape=[S, D], strides=[stride_qs, 1], block_shape=[BLOCK_M, D], padding_option="zero"
    )
    k_desc = tl.make_tensor_descriptor(
        k_base, shape=[S, D], strides=[stride_ks, 1], block_shape=[BLOCK_N, D], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        v_base, shape=[S, D], strides=[stride_vs, 1], block_shape=[BLOCK_N, D], padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        o_base, shape=[S, D], strides=[stride_os, 1], block_shape=[BLOCK_M, D]
    )

    offset_m = pid_m * BLOCK_M
    
    # Pre-fetch Q tile purely to Shared Memory for WGMMA SS-GEMM
    q = q_desc.load([offset_m, 0])

    # Convert scale logic to leverage Hardware Base-2 Exponent Math (MUFU.EX2)
    sm_scale_log2 = (1.0 / (float(D) ** 0.5)) * 1.4426950408889634
    
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    
    # -------------------------------------------------------------
    # FAST-PATH: Loop-Peeled zero ALU masking for strictly standard sequences
    # -------------------------------------------------------------
    if S % BLOCK_M == 0 and S % BLOCK_N == 0:
        num_steps = S // BLOCK_N
        
        # loop_unroll_factor drastically improves register allocation for interleaved MUFU operations
        for step in tl.range(0, num_steps, loop_unroll_factor=2):
            offset_n = step * BLOCK_N
            
            k = k_desc.load([offset_n, 0])
            
            # WGMMA SS-GEMM directly consumes TMA buffered k-tiles sequentially
            qk = tl.dot(q, k.T, out_dtype=tl.float32)
            qk = qk * sm_scale_log2

            m_ij = tl.max(qk, 1)
            m_i_new = tl.maximum(m_i, m_ij)
            
            alpha = tl.exp2(m_i - m_i_new)
            p = tl.exp2(qk - m_i_new[:, None])
            
            l_i_new = alpha * l_i + tl.sum(p, 1)
            p_bf16 = p.to(tl.bfloat16)
            
            # Defer TMA Load V exactly here to distribute SMEM bandwidth and TMA issuance load
            v = v_desc.load([offset_n, 0])
            
            # WGMMA RS-GEMM concurrently evaluates as TMA prefetches sequential pipelines
            acc = acc * alpha[:, None]
            acc = tl.dot(p_bf16, v, acc)
            
            m_i = m_i_new
            l_i = l_i_new

        # Reciprocal scale avoids an expensive vector Division constraint in the epilogue
        l_i_inv = 1.0 / l_i
        acc = acc * l_i_inv[:, None]
        out = acc.to(tl.bfloat16)
        
        o_desc.store([offset_m, 0], out)
        
        # Log2 base scaled symmetrically back to Mathematical Natural Logarithm limits
        lse = m_i * 0.6931471805599453 + tl.log(l_i)
        lse_offset = b * stride_lseb + h * stride_lseh
        
        offs_m = offset_m + tl.arange(0, BLOCK_M)
        tl.store(LSE + lse_offset + offs_m * stride_lses, lse)

    # -------------------------------------------------------------
    # SAFE-PATH: Safely executed for un-even edge sequence dimensions
    # -------------------------------------------------------------
    else:
        num_steps = tl.cdiv(S, BLOCK_N)
        offs_m = offset_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S
        
        m_i = tl.where(mask_m, m_i, 0.0)
        
        for step in tl.range(0, num_steps):
            offset_n = step * BLOCK_N
            k = k_desc.load([offset_n, 0])
            
            qk = tl.dot(q, k.T, out_dtype=tl.float32)
            qk = qk * sm_scale_log2

            offs_n = offset_n + tl.arange(0, BLOCK_N)
            qk_mask = mask_m[:, None] & (offs_n[None, :] < S)
            qk = tl.where(qk_mask, qk, float("-inf"))

            m_ij = tl.max(qk, 1)
            m_i_new = tl.maximum(m_i, m_ij)
            m_i_new = tl.where(mask_m, m_i_new, 0.0)
            
            alpha = tl.exp2(m_i - m_i_new)
            p = tl.exp2(qk - m_i_new[:, None])
            
            l_i_new = alpha * l_i + tl.sum(p, 1)
            p_bf16 = p.to(tl.bfloat16)
            
            v = v_desc.load([offset_n, 0])
            
            acc = acc * alpha[:, None]
            acc = tl.dot(p_bf16, v, acc)
            
            m_i = m_i_new
            l_i = l_i_new
            
        l_i_safe = tl.where(mask_m, l_i, 1.0)
        l_i_inv = 1.0 / l_i_safe
        acc = acc * l_i_inv[:, None]
        out = acc.to(tl.bfloat16)
        
        o_desc.store([offset_m, 0], out)
        
        lse = m_i * 0.6931471805599453 + tl.log(l_i_safe)
        lse_offset = b * stride_lseb + h * stride_lseh
        tl.store(LSE + lse_offset + offs_m * stride_lses, lse, mask=mask_m)


def run(Q, K, V, O, LSE):
    """
    Computes Scaled Dot-Product Attention natively utilizing Hopper TMA pathways, 
    highly hoisted WGMMA loops, MUFU scaling intrinsics, and large batch matrices.
    Outputs are safely committed incrementally to pre-allocated contiguous destination pointers.
    """
    torch.cuda.set_device(Q.device)
    
    B_sz, H_sz, S_sz, D_sz = Q.shape
    
    grid = lambda META: (
        triton.cdiv(S_sz, META["BLOCK_M"]),
        B_sz * H_sz
    )
    
    _fwd_kernel_tma_optimized[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        S=S_sz, H=H_sz, D=D_sz,
    )