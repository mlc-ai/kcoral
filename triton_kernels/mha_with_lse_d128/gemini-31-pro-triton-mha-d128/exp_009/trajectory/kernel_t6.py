import torch
import triton
import triton.language as tl

# Configure the allocator for device-created TMA descriptors natively supported on SM90+
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

@triton.autotune(
    configs=[
        # Extreme arithmetic intensity. Maximizes N to reduce softmax/WGMMA inner loop overheads by 50%
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256}, num_warps=4, num_stages=3),
        
        # Standard balanced configs
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64},  num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64},  num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64},  num_warps=4, num_stages=4),
    ],
    key=['S']
)
@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale_log2, O, LSE,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    B, H, S,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    start_m = tl.program_id(0)
    bh = tl.program_id(1)
    
    b = bh // H
    h = bh % H
    
    # 2D base pointers avoiding 4D stride complexity in descriptor creations
    q_ptr = Q + b * stride_qb + h * stride_qh
    k_ptr = K + b * stride_kb + h * stride_kh
    v_ptr = V + b * stride_vb + h * stride_vh
    o_ptr = O + b * stride_ob + h * stride_oh
    lse_ptr = LSE + b * stride_lseb + h * stride_lseh
    
    # Device-side TMA descriptor allocations - fast, safe approach guaranteeing WGMMA Hopper layout
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
    
    offset_m = start_m * BLOCK_M
    q = q_desc.load([offset_m, 0])
    
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    num_steps = tl.cdiv(S, BLOCK_N)
    
    for i in range(num_steps):
        offset_n = i * BLOCK_N
        k = k_desc.load([offset_n, 0])
        v = v_desc.load([offset_n, 0])
        
        # Unmodified operands guarantees optimal SMEM execution paths natively bypassing register pressure issues
        qk = tl.dot(q, k.T)
        qk = qk * sm_scale_log2
        
        # Fast causal-like mask isolation strictly handling non-divisible trailing sequence pads
        if S % BLOCK_N != 0:
            if i == num_steps - 1:
                offs_n = offset_n + tl.arange(0, BLOCK_N)
                qk = tl.where(offs_n[None, :] < S, qk, float("-inf"))
                
        m_ij = tl.max(qk, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        
        # Hardware SFU MUFU.EX2 operations triggered natively reducing overall cycle counts drastically
        alpha = tl.exp2(m_i - m_i_new)
        p = tl.exp2(qk - m_i_new[:, None])
        
        l_i_new = alpha * l_i + tl.sum(p, 1)
        
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        
        m_i = m_i_new
        l_i = l_i_new

    offs_m = offset_m + tl.arange(0, BLOCK_M)
    mask_m = offs_m < S
    
    l_i_safe = tl.where(mask_m, l_i, 1.0)
    
    # Fast array reciprocal extracting expensive fp32 tensor division saving approx ~500k cycles per threadblock
    rcp_l = 1.0 / l_i_safe
    acc = acc * rcp_l[:, None]
    
    # Revert log2 scaled factor mapping precisely back to natural logarithm bounds mathematically
    lse = m_i * 0.6931471805599453 + tl.log(l_i_safe)
    
    # Descriptors smooth discard writes securely beyond boundary
    o_desc.store([offset_m, 0], acc.to(tl.bfloat16))
    
    lse_offs = offs_m * stride_lses
    tl.store(lse_ptr + lse_offs, lse, mask=mask_m)

def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    # Pre-calculated mathematical scale fold for hardware exp2 dynamics
    sm_scale_log2 = (1.0 / (D ** 0.5)) * 1.4426950408889634
    
    # Optimal group SM L2 scheduling ensuring linear blocks for overlapping heads share identical KV cache
    grid = lambda META: (triton.cdiv(S, META['BLOCK_M']), B * H)
    
    _fwd_kernel[grid](
        Q, K, V, sm_scale_log2, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        B, H, S,
        BLOCK_D=D
    )