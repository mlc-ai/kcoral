import torch
import triton
import triton.language as tl
import math

# Configure Triton allocator for device-created tensor descriptors (Hopper TMA)
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

@triton.autotune(
    configs=[
        # 128x128
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=2, num_warps=8),
        # 128x64
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=6, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=5, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
        # 64x128
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=3, num_warps=8),
        # 64x64
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=6, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=5, num_warps=4),
        # 256x64
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64}, num_stages=5, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64}, num_stages=3, num_warps=8),
        # 256x128
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128}, num_stages=2, num_warps=8),
    ],
    key=['S']
)
@triton.jit
def _attn_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, sm_scale_log2,
    O_ptr, LSE_ptr,
    stride_qb, stride_qh, stride_qs,
    stride_kb, stride_kh, stride_ks,
    stride_vb, stride_vh, stride_vs,
    stride_ob, stride_oh, stride_os,
    stride_lseb, stride_lseh, stride_lses,
    B, H, S, D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    EVEN_S: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)
    
    # Compute base pointers for this batch and head
    q_offset = pid_b * stride_qb + pid_h * stride_qh
    k_offset = pid_b * stride_kb + pid_h * stride_kh
    v_offset = pid_b * stride_vb + pid_h * stride_vh
    o_offset = pid_b * stride_ob + pid_h * stride_oh
    lse_offset = pid_b * stride_lseb + pid_h * stride_lseh
    
    Q_base = Q_ptr + q_offset
    K_base = K_ptr + k_offset
    V_base = V_ptr + v_offset
    O_base = O_ptr + o_offset
    LSE_base = LSE_ptr + lse_offset
    
    # Create Hopper TMA tensor descriptors (inner stride is statically 1 for contiguous features)
    q_desc = tl.make_tensor_descriptor(
        Q_base, shape=[S, D], strides=[stride_qs, 1],
        block_shape=[BLOCK_M, D], padding_option="zero"
    )
    k_desc = tl.make_tensor_descriptor(
        K_base, shape=[S, D], strides=[stride_ks, 1],
        block_shape=[BLOCK_N, D], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        V_base, shape=[S, D], strides=[stride_vs, 1],
        block_shape=[BLOCK_N, D], padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        O_base, shape=[S, D], strides=[stride_os, 1],
        block_shape=[BLOCK_M, D]
    )
    
    start_m = pid_m * BLOCK_M
    
    # Load Q tile and pre-scale to avoid loop multiplication overhead
    # We apply log2(e) scaling here so we can use hardware exp2 in the loop
    q = q_desc.load([start_m, 0])
    q = (q * sm_scale_log2).to(tl.bfloat16)
    
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    
    if not EVEN_S:
        offs_m = start_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S
    
    # K and V loop
    for start_n in range(0, S, BLOCK_N):
        # TMA descriptor loads
        k = k_desc.load([start_n, 0])
        v = v_desc.load([start_n, 0])
        
        # Q @ K.T (WGMMA)
        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        
        if not EVEN_S:
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_qk = mask_m[:, None] & (offs_n[None, :] < S)
            qk = tl.where(mask_qk, qk, float("-inf"))
        
        # Max scaling step
        m_ij = tl.max(qk, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        
        if not EVEN_S:
            # Prevents `-inf - (-inf) = NaN` for fully out-of-bounds rows
            m_i_new = tl.where(mask_m, m_i_new, 0.0)
        
        # Using exp2 instead of exp directly maps to fast hardware instructions
        alpha = tl.exp2(m_i - m_i_new)
        beta = tl.exp2(qk - m_i_new[:, None])
        
        # Update normalizer
        l_i_new = alpha * l_i + tl.sum(beta, 1)
        
        # (Softmax * V) accumulation step (WGMMA)
        p = beta.to(tl.bfloat16)
        acc = acc * alpha[:, None]
        acc = tl.dot(p, v, acc, out_dtype=tl.float32)
        
        m_i = m_i_new
        l_i = l_i_new
        
    # Finalize O
    acc = acc * (1.0 / l_i[:, None])
    O_val = acc.to(tl.bfloat16)
    
    # Store O using TMA (automatically handles out-of-bounds rows)
    o_desc.store([start_m, 0], O_val)
    
    # Store LSE using normal pointers since TMA descriptors are 2D+
    offs_m_store = start_m + tl.arange(0, BLOCK_M)
    lse_ptrs = LSE_base + (offs_m_store * stride_lses)
    
    # Convert base-2 max back to natural log domain: m_i_e = m_i_2 * ln(2)
    # LSE = (m_i + log2(l_i)) * ln(2)
    lse_val = (m_i + tl.log2(l_i)) * 0.6931471805599453
    
    if EVEN_S:
        tl.store(lse_ptrs, lse_val)
    else:
        tl.store(lse_ptrs, lse_val, mask=offs_m_store < S)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    
    if S == 0:
        return
        
    # Scale Q by 1/sqrt(D) AND log2(e) ahead of time for hardware exp2 
    sm_scale_log2 = (1.0 / math.sqrt(D)) * 1.4426950408889634
    
    # Statically determine if the sequence length is cleanly divisible by all possible tile configurations
    even_s = (S % 256 == 0)
    
    # Using a 3D grid natively maps pid_h and pid_b to avoid extra runtime integer arithmetic
    grid = lambda META: (triton.cdiv(S, META['BLOCK_M']), H, B)
    
    _attn_fwd_kernel[grid](
        Q, K, V, sm_scale_log2,
        O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        B, H, S, D=D, EVEN_S=even_s,
    )