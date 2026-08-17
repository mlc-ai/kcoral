import torch
import triton
import triton.language as tl

@triton.heuristics({
    "EVEN_M": lambda args: args["S"] % args["BLOCK_M"] == 0,
    "EVEN_N": lambda args: args["S"] % args["BLOCK_N"] == 0,
})
@triton.autotune(
    configs=[
        # Large M, high K/V reuse per iteration (Highly optimal for long sequences)
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64}, num_warps=8, num_stages=5),
        
        # 128x128 standard sweet spot
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
        
        # High depth staging pipeline
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=5),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=5),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=6),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=6),
        
        # Alternative 64x128 profile
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=3),
    ],
    key=["S"],
)
@triton.jit
def _attn_fwd_kernel(
    Q, K, V, O, LSE,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lb, stride_lh, stride_ls,
    S,
    sm_scale_log2,
    num_heads,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    start_m = tl.program_id(0)
    bh = tl.program_id(1)
    
    b = bh // num_heads
    h = bh % num_heads

    # Map TMA descriptor pointers strictly corresponding to the head domain
    q_ptr = Q + (b * stride_qb + h * stride_qh)
    k_ptr = K + (b * stride_kb + h * stride_kh)
    v_ptr = V + (b * stride_vb + h * stride_vh)
    o_ptr = O + (b * stride_ob + h * stride_oh)

    # Initialize Hardware Tensor Memory Accelerator (TMA) Async Descriptors
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

    offset_m = start_m * BLOCK_M
    
    # Pre-load and globally scale Q just once per block natively replacing over 500k scalar ops inside loop
    q = q_desc.load([offset_m, 0])
    q = (q * sm_scale_log2).to(tl.bfloat16)

    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    offs_n_base = tl.arange(0, BLOCK_N)
    num_n_blocks = tl.cdiv(S, BLOCK_N)
    
    for n_idx in range(0, num_n_blocks):
        offset_n = n_idx * BLOCK_N
        
        # Async TMA hardware loads caching pipeline stages securely into SMEM
        k = k_desc.load([offset_n, 0])
        v = v_desc.load([offset_n, 0])
        
        # SM90 WGMMA fast-path streaming (k.T handles column-major structurally transparently)
        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        
        # Eliminate control flow masks identically leveraging zero-padding on valid evenly scaled matrices
        if not EVEN_N:
            if offset_n + BLOCK_N > S:
                offs_n = offset_n + offs_n_base
                qk = tl.where(offs_n[None, :] < S, qk, float('-inf'))
        
        # Hardware-accelerated base-2 execution
        m_ij = tl.max(qk, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)
        
        alpha = tl.exp2(m_i - m_i_new)
        beta = tl.exp2(qk - m_i_new[:, None])
        
        l_ij = tl.sum(beta, axis=1)
        l_i_new = l_i * alpha + l_ij
        
        # Re-scale sums
        acc = acc * alpha[:, None]
        
        # Re-encode fractional iteration weights targeting fast native BF16 execution
        p = beta.to(tl.bfloat16)
        
        # Follow-up PV WGMMA calculation leveraging row-major V structurally seamlessly natively 
        acc = tl.dot(p, v, acc, out_dtype=tl.float32)
        
        m_i = m_i_new
        l_i = l_i_new

    # Output projection normalization
    out = acc / l_i[:, None]
    
    # Store evaluated sequence utilizing asynchronous TMA bounds checks naturally
    o_desc.store([offset_m, 0], out.to(tl.bfloat16))
    
    # Target PTX lg2 executing direct hardware base mappings avoiding complex e^x emulation expansions
    lse = (m_i + tl.log2(l_i)) * 0.6931471805599453
    
    # Push normalized LSE scaling
    offs_m = offset_m + tl.arange(0, BLOCK_M)
    lse_ptrs = LSE + (b * stride_lb + h * stride_lh + offs_m * stride_ls)
    
    if not EVEN_M:
        mask_m = offs_m < S
        tl.store(lse_ptrs, lse, mask=mask_m)
    else:
        tl.store(lse_ptrs, lse)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    if S == 0:
        return
        
    def alloc_fn(size: int, alignment: int, stream):
        return torch.empty(size, device=Q.device, dtype=torch.int8)
        
    triton.set_allocator(alloc_fn)
        
    # Extracted logarithmic equivalence scaling multiplier: (1 / sqrt(D)) * log2(e) 
    sm_scale_log2 = (1.0 / (D ** 0.5)) * 1.4426950408889634
    
    # Structuring (M, B*H, 1) to launch sequential sequence chunks assigned uniformly along grid-0 sequentially.
    # Yields implicit grouping natively causing different SMs to broadcast K/V reads overlapping in the multi-megabyte L2 caching effectively globally.
    grid = lambda META: (
        triton.cdiv(S, META["BLOCK_M"]),
        B * H,
        1
    )
    
    _attn_fwd_kernel[grid](
        Q=Q, K=K, V=V, O=O, LSE=LSE,
        stride_qb=Q.stride(0), stride_qh=Q.stride(1), stride_qs=Q.stride(2), stride_qd=Q.stride(3),
        stride_kb=K.stride(0), stride_kh=K.stride(1), stride_ks=K.stride(2), stride_kd=K.stride(3),
        stride_vb=V.stride(0), stride_vh=V.stride(1), stride_vs=V.stride(2), stride_vd=V.stride(3),
        stride_ob=O.stride(0), stride_oh=O.stride(1), stride_os=O.stride(2), stride_od=O.stride(3),
        stride_lb=LSE.stride(0), stride_lh=LSE.stride(1), stride_ls=LSE.stride(2),
        S=S,
        sm_scale_log2=sm_scale_log2,
        num_heads=H,
        BLOCK_D=128,
    )