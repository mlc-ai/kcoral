import torch
import triton
import triton.language as tl
import math

def alloc_fn(size: int, alignment: int, stream):
    """
    Allocator required by standard Triton Hopper guidelines to support device-created TMA descriptors.
    """
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)


@triton.autotune(
    configs=[
        # Optimized configurations tailored for H100 (SM90) accommodating 228KB Shared Memory limit
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64},  num_warps=8, num_stages=5),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64},  num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64},  num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128}, num_warps=4, num_stages=3),
    ],
    key=["S"]
)
@triton.jit
def _fwd_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr,
    sm_scale,
    S,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    H: tl.constexpr, D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    b_idx = pid_bh // H
    h_idx = pid_bh % H
    
    # Calculate base pointers isolating exactly one specific [Batch, Head] subspace
    q_base = q_ptr + b_idx * stride_qb + h_idx * stride_qh
    k_base = k_ptr + b_idx * stride_kb + h_idx * stride_kh
    v_base = v_ptr + b_idx * stride_vb + h_idx * stride_vh
    o_base = o_ptr + b_idx * stride_ob + h_idx * stride_oh
    
    # Generate 2D Tensor Descriptors natively bounds-checked and padded by TMA hardware
    q_desc = tl.make_tensor_descriptor(
        q_base, shape=[S, D], strides=[stride_qs, stride_qd],
        block_shape=[BLOCK_M, D], padding_option="zero"
    )
    k_desc = tl.make_tensor_descriptor(
        k_base, shape=[S, D], strides=[stride_ks, stride_kd],
        block_shape=[BLOCK_N, D], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        v_base, shape=[S, D], strides=[stride_vs, stride_vd],
        block_shape=[BLOCK_N, D], padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        o_base, shape=[S, D], strides=[stride_os, stride_od],
        block_shape=[BLOCK_M, D]
    )
    
    offset_m = pid_m * BLOCK_M
    
    # Single load for Q, scaled identically per element (combining natural-log Softmax with base-2 throughput math)
    q = q_desc.load([offset_m, 0])
    LOG2_E = 1.4426950408889634
    q = (q * (sm_scale * LOG2_E)).to(tl.bfloat16)
    
    # FP32 states initialization for softmax stability
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    
    num_full_blocks = S // BLOCK_N
    
    # Optimal uniform loop exclusively running sequence blocks that completely guarantee safe limits.
    # No if-else structures; provides perfectly seamless compiler software pipelining.
    for start_n in range(num_full_blocks):
        offset_n = start_n * BLOCK_N
        k = k_desc.load([offset_n, 0])
        v = v_desc.load([offset_n, 0])
        
        # In-place write to WGMMA results avoiding extraneous zeros init assignments
        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        
        alpha = tl.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        
        m_i = m_ij

    # Mask-guarded evaluation for arbitrary unaligned dimension remainder (tail block)
    if S % BLOCK_N != 0:
        offset_n = num_full_blocks * BLOCK_N
        k = k_desc.load([offset_n, 0])
        v = v_desc.load([offset_n, 0])
        
        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        
        # Masking zero-padded regions preventing them from impacting Softmax normalizers
        offs_n = offset_n + tl.arange(0, BLOCK_N)
        qk = tl.where(offs_n[None, :] < S, qk, float('-inf'))
        
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        
        alpha = tl.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        
        m_i = m_ij

    # Extremely critical: Replace large-scale block elementwise divisions with 1D vector reciprocals + scaled multiplications!
    inv_l_i = 1.0 / l_i
    acc = acc * inv_l_i[:, None]
    
    LN_2 = 0.6931471805599453
    lse = (m_i + tl.log2(l_i)) * LN_2
    
    # Store finalized evaluations discarding internally-padded TMA boundary violations dynamically
    o_desc.store([offset_m, 0], acc.to(tl.bfloat16))
    
    # Fallback to precise manual masking boundary drops for the LSE extraction
    offs_m = offset_m + tl.arange(0, BLOCK_M)
    mask_m = offs_m < S
    lse_ptrs = lse_ptr + b_idx * stride_lseb + h_idx * stride_lseh + offs_m * stride_lses
    tl.store(lse_ptrs, lse, mask=mask_m)


def run(Q, K, V, O, LSE):
    """
    Computes a batched non-causal multi-head attention forward pass, yielding output states and Softmax LSE metric.
    
    Inputs: Q, K, V are implicitly assumed contiguous BFloat16 [B, H, S, D]
    Outputs: O BFloat16 [B, H, S, D], LSE Float32 [B, H, S]
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    sm_scale = 1.0 / math.sqrt(D)
    
    # Z-curve row-major mapping leverages dense sequential evaluation on SM90 retaining heavy L2-level overlap.
    grid = lambda META: (
        triton.cdiv(S, META['BLOCK_M']),
        B * H
    )
    
    _fwd_kernel[grid](
        Q, K, V, O, LSE,
        sm_scale,
        S,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        H=H,
        D=D,
    )