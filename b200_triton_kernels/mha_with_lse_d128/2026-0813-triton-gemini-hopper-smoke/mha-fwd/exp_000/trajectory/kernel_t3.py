import torch
import triton
import triton.language as tl

# Infrastructure storage allocator for device-side tensor descriptors (TMA)
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
    ],
    key=['S'],
)
@triton.jit
def mha_fwd_kernel_tma(
    Q, K, V, O, LSE,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    B, H, S, D, scale_log2,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    EVEN_S: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    b_id = pid_bh // H
    h_id = pid_bh % H
    
    # Establish local base pointers bounding to batch and head identifiers
    q_base = Q + b_id * stride_qb + h_id * stride_qh
    k_base = K + b_id * stride_kb + h_id * stride_kh
    v_base = V + b_id * stride_vb + h_id * stride_vh
    o_base = O + b_id * stride_ob + h_id * stride_oh
    
    # Hopper TMA descriptors seamlessly handle non-multiples bounds zero padding natively
    desc_q = tl.make_tensor_descriptor(
        q_base, shape=[S, D], strides=[stride_qs, stride_qd],
        block_shape=[BLOCK_M, BLOCK_D], padding_option="zero"
    )
    desc_k = tl.make_tensor_descriptor(
        k_base, shape=[S, D], strides=[stride_ks, stride_kd],
        block_shape=[BLOCK_N, BLOCK_D], padding_option="zero"
    )
    desc_v = tl.make_tensor_descriptor(
        v_base, shape=[S, D], strides=[stride_vs, stride_vd],
        block_shape=[BLOCK_N, BLOCK_D], padding_option="zero"
    )
    desc_o = tl.make_tensor_descriptor(
        o_base, shape=[S, D], strides=[stride_os, stride_od],
        block_shape=[BLOCK_M, BLOCK_D]
    )
    
    start_m = pid_m * BLOCK_M
    
    # Load completely unmasked via TMA descriptor abstraction
    q = desc_q.load([start_m, 0])
    
    # Pre-scale Q completely outside of the intensive iteration loop saving vast computation bounds
    q = (q * scale_log2).to(tl.bfloat16)
    
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    num_n_blocks = tl.cdiv(S, BLOCK_N)
    
    for n_idx in tl.range(0, num_n_blocks):
        start_n = n_idx * BLOCK_N
        
        # Pipelined loads stream directly utilizing Hopper's asynchronous collective loads
        k = desc_k.load([start_n, 0])
        v = desc_v.load([start_n, 0])
        
        # Matrix multiplication is bilinearly accurate utilizing our globally pre-scaled Q
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, k.T, acc=qk, out_dtype=tl.float32)
        
        # Exclusively map boundary checks statically resolving execution bounds when applicable
        if not EVEN_S:
            if start_n + BLOCK_N > S:
                curr_offs_n = start_n + tl.arange(0, BLOCK_N)
                mask_n = curr_offs_n < S
                qk = tl.where(mask_n[None, :], qk, -float('inf'))
        
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        
        alpha = tl.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        
        acc = acc * alpha[:, None]
        p_bf16 = p.to(tl.bfloat16)
        acc = tl.dot(p_bf16, v, acc=acc)
        
        m_i = m_ij

    # Safe log2 boundaries properly restoring exact mathematical bases natively over ln constants
    acc = acc / l_i[:, None]
    lse = (m_i + tl.log2(l_i)) * 0.6931471805599453
    
    # Hardware bounded masked writes
    desc_o.store([start_m, 0], acc.to(tl.bfloat16))
    
    offs_m = start_m + tl.arange(0, BLOCK_M)
    mask_m = offs_m < S
    lse_ptrs = LSE + b_id * stride_lseb + h_id * stride_lseh + offs_m * stride_lses
    
    if not EVEN_S:
        tl.store(lse_ptrs, lse, mask=mask_m)
    else:
        tl.store(lse_ptrs, lse)


def run(Q, K, V, O, LSE):
    """
    Computes Scaled Dot-Product non-causal Attention mapped sequentially for SM parallelism bounds.
    Writes outputs efficiently leveraging exact mathematically proven transformations.
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    
    # Pre-compose coefficients explicitly mapped over ln2 to completely avoid expensive exp() routines
    scale = 1.0 / (D ** 0.5)
    scale_log2 = scale * 1.4426950408889634
    
    # Static boundary resolution targeting sequences matched over 128 elements seamlessly bounded
    EVEN_S = (S % 128 == 0)
    
    grid = lambda META: (triton.cdiv(S, META['BLOCK_M']), B * H, 1)
    
    mha_fwd_kernel_tma[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        B, H, S, D, scale_log2,
        BLOCK_D=128,
        EVEN_S=EVEN_S
    )