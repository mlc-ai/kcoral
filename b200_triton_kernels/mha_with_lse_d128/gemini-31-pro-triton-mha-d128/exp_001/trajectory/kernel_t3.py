import torch
import triton
import triton.language as tl

@triton.heuristics({
    "REQUIRES_MASK_N": lambda args: args["S"] % args["BLOCK_N"] != 0,
    "REQUIRES_MASK_M": lambda args: args["S"] % args["BLOCK_M"] != 0,
})
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=4),
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
    REQUIRES_MASK_N: tl.constexpr,
    REQUIRES_MASK_M: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    start_m = tl.program_id(0)
    bh = tl.program_id(1)
    
    b = bh // num_heads
    h = bh % num_heads

    # Navigate base pointers for device TMA descriptors per head 
    q_ptr = Q + (b * stride_qb + h * stride_qh)
    k_ptr = K + (b * stride_kb + h * stride_kh)
    v_ptr = V + (b * stride_vb + h * stride_vh)
    o_ptr = O + (b * stride_ob + h * stride_oh)

    # Instantiate hardware Tensor Memory Accelerator (TMA) descriptors natively
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
    q = q_desc.load([offset_m, 0])
    
    # Scale Q globally for Base-2 prior to looping: Q' = Q * (1 / sqrt(D)) * log2(e)
    q = (q * sm_scale_log2).to(tl.bfloat16)

    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    offs_n_base = tl.arange(0, BLOCK_N)
    num_n_blocks = tl.cdiv(S, BLOCK_N)
    
    for n_idx in range(num_n_blocks):
        offset_n = n_idx * BLOCK_N
        
        # Async TMA loads
        k = k_desc.load([offset_n, 0])
        v = v_desc.load([offset_n, 0])
        
        # Hardware dot accumulates exactly down cleanly into WGMMA Tensor Cores
        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        
        # Completely disappears out of the compilation if structurally clean (S % BLOCK_N == 0)
        # Prevents any loop control-flow breaks and enables strict, robust software pipelining.
        if REQUIRES_MASK_N:
            offs_n = offset_n + offs_n_base
            qk = tl.where(offs_n[None, :] < S, qk, float('-inf'))
        
        # Evaluate standard FA operations in base-2 natively
        m_ij = tl.max(qk, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)
        
        alpha = tl.exp2(m_i - m_i_new)
        beta = tl.exp2(qk - m_i_new[:, None])
        
        l_ij = tl.sum(beta, axis=1)
        l_i_new = l_i * alpha + l_ij
        
        acc = acc * alpha[:, None]
        p = beta.to(tl.bfloat16)
        acc = tl.dot(p, v, acc, out_dtype=tl.float32)
        
        m_i = m_i_new
        l_i = l_i_new

    # Output Normalization
    out = acc / l_i[:, None]
    o_desc.store([offset_m, 0], out.to(tl.bfloat16))
    
    # Recover scale back to Natural-log precision safely mapping LSE_e = m_2 * ln(2) + ln(l)
    lse = m_i * 0.6931471805599453 + tl.log(l_i)
    
    offs_m = offset_m + tl.arange(0, BLOCK_M)
    lse_ptrs = LSE + (b * stride_lb + h * stride_lh + offs_m * stride_ls)
    
    if REQUIRES_MASK_M:
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
        
    sm_scale_log2 = (1.0 / (D ** 0.5)) * 1.4426950408889634
    
    # Sequential CTA progression favors holding Batch/Head continuous to maximize K and V cache retention globally in L2
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