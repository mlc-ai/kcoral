import torch
import triton
import triton.language as tl

# Comprehensive autotuning over block geometries and pipeline depths to saturate Blackwell's FP32 MAC rates.
# Memory pipeline is governed natively through num_stages overlapping TMA asynchronously with inner Tensor Core math.
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
    ],
    key=['S']
)
@triton.jit
def _fwd_kernel(
    Q, K, V, O, LSE,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    S, sm_scale, H,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    start_m = tl.program_id(0)
    bh = tl.program_id(1)

    # Early exit gracefully ignores completely out-of-bounds padded sequences 
    if start_m * BLOCK_M >= S:
        return

    batch = bh // H
    head = bh % H

    # Evaluate physical batch/head 2D offset bases 
    q_base = Q + batch * stride_qb + head * stride_qh
    k_base = K + batch * stride_kb + head * stride_kh
    v_base = V + batch * stride_vb + head * stride_vh
    o_base = O + batch * stride_ob + head * stride_oh

    # Native TMA descriptors resolve block bounds constraints invisibly via zero-padding,
    # omitting entirely the need for expensive pointer masking and conditionals in loops.
    q_desc = tl.make_tensor_descriptor(
        q_base, shape=[S, HEAD_DIM], strides=[stride_qs, 1], block_shape=[BLOCK_M, HEAD_DIM], padding_option="zero"
    )
    k_desc = tl.make_tensor_descriptor(
        k_base, shape=[S, HEAD_DIM], strides=[stride_ks, 1], block_shape=[BLOCK_N, HEAD_DIM], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        v_base, shape=[S, HEAD_DIM], strides=[stride_vs, 1], block_shape=[BLOCK_N, HEAD_DIM], padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        o_base, shape=[S, HEAD_DIM], strides=[stride_os, 1], block_shape=[BLOCK_M, HEAD_DIM], padding_option="zero"
    )

    m_idx = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    
    # Load bounded Q block strictly once
    q = q_desc.load([start_m * BLOCK_M, 0])

    # Multiply native fp32 softmax scale with log2 conversion ratio directly
    RCP_LN2: tl.constexpr = 1.4426950408889634
    scale = sm_scale * RCP_LN2

    # Retain maximal precision for intermediate running softmax registers
    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)

    # Resolve safe block tracking dynamically determining purely unmasked boundaries
    num_full_blocks = (start_m * BLOCK_M) // BLOCK_N

    # Phase 1: Unmasked Core Blocks 
    # Zero masking overhead required as we formally proven all `m_idx >= n_idx` for these keys, safely maximizing throughput.
    for k0 in range(0, num_full_blocks):
        offset_n = k0 * BLOCK_N
        k = k_desc.load([offset_n, 0])
        v = v_desc.load([offset_n, 0])
        
        qk = tl.dot(q, k.T, out_dtype=tl.float32) * scale
        
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.math.exp2(m_i - m_ij)
        p = tl.math.exp2(qk - m_ij[:, None])
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc, out_dtype=tl.float32)
        
        m_i = m_ij

    # Phase 2: Causally Masked Edge/Diagonal Blocks
    max_k_idx = (start_m + 1) * BLOCK_M
    num_total_blocks = tl.cdiv(max_k_idx, BLOCK_N)

    for k0 in range(num_full_blocks, num_total_blocks):
        offset_n = k0 * BLOCK_N
        if offset_n >= S:
            break
            
        k = k_desc.load([offset_n, 0])
        v = v_desc.load([offset_n, 0])
        
        qk = tl.dot(q, k.T, out_dtype=tl.float32) * scale
        
        # Purely elegant single constraint evaluation covers both lower-triangular causal tracking 
        # and implicitly zeroing elements outside Sequence padding boundaries safely.
        n_idx = offset_n + tl.arange(0, BLOCK_N)
        valid_score = m_idx[:, None] >= n_idx[None, :]
        qk = tl.where(valid_score, qk, -float("inf"))
        
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.math.exp2(m_i - m_ij)
        p = tl.math.exp2(qk - m_ij[:, None])
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc, out_dtype=tl.float32)
        
        m_i = m_ij

    # Safely evaluate normalization 
    out = acc / l_i[:, None]

    # Re-normalize hardware optimal base-2 metrics safely back to Standard Log LSE domain
    LN2: tl.constexpr = 0.6931471805599453
    lse = (m_i + tl.math.log2(l_i)) * LN2

    # Issue TMA block store implicitly dropping any computed pad sequences via bound constraints
    o_desc.store([start_m * BLOCK_M, 0], out.to(tl.bfloat16))

    # Finally enforce linear pointer boundary masks for simple single-dimension LSE metric
    lse_ptrs = LSE + batch * stride_lseb + head * stride_lseh + m_idx * stride_lses
    tl.store(lse_ptrs, lse, mask=m_idx < S)


def run(Q, K, V, O, LSE):
    """
    Standard Triton causal multi-head attention forward computing Output and Natural-Log-Sum-Exp.
    Writes entirely in-place to preallocated 'O' and 'LSE'.
    """
    torch.cuda.set_device(Q.device)
    
    # Authorize Triton device descriptors utilization infrastructure footprint without creating host-side block shape conflicts
    def alloc_fn(size: int, alignment: int, stream):
        return torch.empty(size, device=Q.device, dtype=torch.int8)
    triton.set_allocator(alloc_fn)

    B, H, S, D = Q.shape
    sm_scale = 1.0 / (D ** 0.5)

    # Enforce clustered grouping behavior implicitly via dimension packing on grid execution
    grid = lambda META: (triton.cdiv(S, META['BLOCK_M']), B * H)
    
    _fwd_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        S, sm_scale, H,
        HEAD_DIM=128
    )