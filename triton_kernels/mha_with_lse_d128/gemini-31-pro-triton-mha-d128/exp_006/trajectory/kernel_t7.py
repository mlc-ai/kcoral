import torch
import triton
import triton.language as tl

# Set Triton's allocator for device-created tensor descriptors.
# This strictly provides the necessary infrastructure storage without allocating model outputs.
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=5, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=4, num_warps=8),
    ],
    key=['S']
)
@triton.jit
def mha_fwd_kernel_tma_device(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    stride_qb, stride_qh, stride_qs,
    stride_kb, stride_kh, stride_ks,
    stride_vb, stride_vh, stride_vs,
    stride_ob, stride_oh, stride_os,
    stride_lseb, stride_lseh, stride_lses,
    B, H, S, sm_scale_log2,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN_S: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)
    
    q_offset = pid_b * stride_qb + pid_h * stride_qh
    k_offset = pid_b * stride_kb + pid_h * stride_kh
    v_offset = pid_b * stride_vb + pid_h * stride_vh
    o_offset = pid_b * stride_ob + pid_h * stride_oh
    lse_offset = pid_b * stride_lseb + pid_h * stride_lseh
    
    # Instantiate TMA 2D Device Descriptors enabling async copies
    q_desc = tl.make_tensor_descriptor(
        Q_ptr + q_offset,
        shape=[S, D],
        strides=[stride_qs, 1],
        block_shape=[BLOCK_M, D],
        padding_option="zero"
    )
    k_desc = tl.make_tensor_descriptor(
        K_ptr + k_offset,
        shape=[S, D],
        strides=[stride_ks, 1],
        block_shape=[BLOCK_N, D],
        padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        V_ptr + v_offset,
        shape=[S, D],
        strides=[stride_vs, 1],
        block_shape=[BLOCK_N, D],
        padding_option="zero"
    )
    # No padding specified for store descriptor to natively ignore out-of-bounds writes
    o_desc = tl.make_tensor_descriptor(
        O_ptr + o_offset,
        shape=[S, D],
        strides=[stride_os, 1],
        block_shape=[BLOCK_M, D]
    )
    
    start_m = pid_m * BLOCK_M
    # Constant 0 provided as an integer tensor to avoid broadcasting failures in tl.load descriptor coordinate mapping
    zero = tl.zeros([], dtype=tl.int32)
    
    q = tl.load(q_desc, [start_m, zero])
    
    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    
    num_n_blocks = tl.cdiv(S, BLOCK_N)
    
    for start_n_idx in range(num_n_blocks):
        start_n = start_n_idx * BLOCK_N
        
        # Issue pipelined asynchronous TMA fetches from Global directly into Shared memory 
        k = tl.load(k_desc, [start_n, zero])
        v = tl.load(v_desc, [start_n, zero])
        
        # Hopper natively schedules these operations via fast WGMMA instructions 
        qk = tl.dot(q, k.T)
        
        # Completely elide conditional masks when Sequence bounds are structurally clean (S is a multiple of BLOCK)
        if not EVEN_S:
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask = offs_n[None, :] < S
            qk = tl.where(mask, qk, float("-inf"))
            
        # Merge Softmax scaling into logarithm base-2 space scaling the intermediate FP32 accumulators
        qk = qk * sm_scale_log2
        
        # Base-2 Online Softmax extraction
        m_i_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp2(m_i - m_i_new)
        p = tl.exp2(qk - m_i_new[:, None])
        l_i_new = alpha * l_i + tl.sum(p, axis=1)
        
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        
        m_i = m_i_new
        l_i = l_i_new
        
    l_i_safe = tl.where(l_i == 0.0, 1.0, l_i)
    out = acc * (1.0 / l_i_safe[:, None])
    
    tl.store(o_desc, [start_m, zero], out.to(tl.bfloat16))
    
    offs_m = start_m + tl.arange(0, BLOCK_M)
    # Rescale base-2 logarithm tracking variables directly into Natural Log values to match Reference Output norms
    lse = (m_i + tl.log2(l_i_safe)) * 0.6931471805599453
    lse_ptrs = LSE_ptr + lse_offset + offs_m * stride_lses
    
    if EVEN_S:
        tl.store(lse_ptrs, lse)
    else:
        tl.store(lse_ptrs, lse, mask=offs_m < S)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=3, num_warps=4),
    ],
    key=['S']
)
@triton.jit
def mha_fwd_kernel_ptr(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    B, H, S, sm_scale_log2,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN_S: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    q_offset = pid_b * stride_qb + pid_h * stride_qh
    k_offset = pid_b * stride_kb + pid_h * stride_kh
    v_offset = pid_b * stride_vb + pid_h * stride_vh
    o_offset = pid_b * stride_ob + pid_h * stride_oh
    lse_offset = pid_b * stride_lseb + pid_h * stride_lseh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    q_ptrs = Q_ptr + q_offset + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    k_ptrs = K_ptr + k_offset + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = V_ptr + v_offset + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd

    if EVEN_S:
        q = tl.load(q_ptrs)
    else:
        q_mask = (offs_m[:, None] < S)
        q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    num_n_blocks = tl.cdiv(S, BLOCK_N)
    k_step = BLOCK_N * stride_ks
    v_step = BLOCK_N * stride_vs

    for start_n_idx in range(num_n_blocks):
        start_n = start_n_idx * BLOCK_N
        
        if EVEN_S:
            k = tl.load(k_ptrs)
            v = tl.load(v_ptrs)
        else:
            k_mask = (start_n + offs_n[:, None] < S)
            k = tl.load(k_ptrs, mask=k_mask, other=0.0)
            v = tl.load(v_ptrs, mask=k_mask, other=0.0)
        
        qk = tl.dot(q, tl.trans(k))
        
        if not EVEN_S:
            mask_n = start_n + offs_n < S
            qk = tl.where(mask_n[None, :], qk, float("-inf"))
            
        qk = qk * sm_scale_log2
        
        m_i_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp2(m_i - m_i_new)
        p = tl.exp2(qk - m_i_new[:, None])
        l_i_new = alpha * l_i + tl.sum(p, axis=1)
        
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        
        m_i = m_i_new
        l_i = l_i_new
        k_ptrs += k_step
        v_ptrs += v_step

    l_i_safe = tl.where(l_i == 0.0, 1.0, l_i)
    out = acc * (1.0 / l_i_safe[:, None])
    
    lse = (m_i + tl.log2(l_i_safe)) * 0.6931471805599453
    
    o_ptrs = O_ptr + o_offset + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    lse_ptrs = LSE_ptr + lse_offset + offs_m * stride_lses
    
    if EVEN_S:
        tl.store(o_ptrs, out.to(tl.bfloat16))
        tl.store(lse_ptrs, lse)
    else:
        out_mask = offs_m < S
        tl.store(o_ptrs, out.to(tl.bfloat16), mask=out_mask[:, None])
        tl.store(lse_ptrs, lse, mask=out_mask)


def run(Q, K, V, O, LSE):
    """
    Computes Non-Causal Multi-Head Attention safely executing WGMMA Tensor Core instructions 
    with natively optimal memory access via device-side TMA Descriptors on Hopper Architecture.
    
    Args:
        Q, K, V: bfloat16 input tensors of shape (B, H, S, D).
        O: preallocated bfloat16 output tensor of shape (B, H, S, D).
        LSE: preallocated float32 output tensor of shape (B, H, S).
    """
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    # Pre-merge standard dot scale normalization into the natural log scaling to avoid duplicate computation blocks
    sm_scale_log2 = (1.0 / (D ** 0.5)) * 1.4426950408889634
    
    # Evaluate sequence alignment boundaries dynamically (TMA Descriptor masking safely covers all padding scenarios,
    # however cleanly bounded block strides allow dropping inner loop pointer bounds tests entirely)
    EVEN_S = (S % 128 == 0)
    
    grid = lambda META: (triton.cdiv(S, META['BLOCK_M']), B, H)
    
    # Check robust physical memory prerequisites enabling TMA
    # Innermost dimensions must represent entirely continuous access elements
    # 2D Leading bounds must remain 16-byte aligned consistently across heads and batches (BF16 items process as 2 bytes)
    use_tma = (
        Q.stride(-1) == 1 and K.stride(-1) == 1 and V.stride(-1) == 1 and O.stride(-1) == 1 and
        (Q.stride(-2) * 2) % 16 == 0 and
        (K.stride(-2) * 2) % 16 == 0 and
        (V.stride(-2) * 2) % 16 == 0 and
        (O.stride(-2) * 2) % 16 == 0
    )
    
    if use_tma:
        mha_fwd_kernel_tma_device[grid](
            Q, K, V, O, LSE,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            LSE.stride(0), LSE.stride(1), LSE.stride(2),
            B, H, S, sm_scale_log2,
            D=D,
            EVEN_S=EVEN_S
        )
    else:
        mha_fwd_kernel_ptr[grid](
            Q, K, V, O, LSE,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            LSE.stride(0), LSE.stride(1), LSE.stride(2),
            B, H, S, sm_scale_log2,
            D=D,
            EVEN_S=EVEN_S
        )