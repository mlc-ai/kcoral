import torch
import triton
import triton.language as tl

def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

def get_autotune_config():
    return [
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=3, num_warps=4),
    ]

@triton.autotune(
    configs=get_autotune_config(),
    key=['S']
)
@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale_log2,
    O, LSE,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    S,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    start_m = tl.program_id(0) * BLOCK_M
    # Early exit for fully out-of-bounds blocks
    if start_m >= S:
        return

    off_b = tl.program_id(1)
    off_h = tl.program_id(2)

    # Base pointers for the current batch and head
    q_ptr = Q + off_b * stride_qb + off_h * stride_qh
    k_ptr = K + off_b * stride_kb + off_h * stride_kh
    v_ptr = V + off_b * stride_vb + off_h * stride_vh
    o_ptr = O + off_b * stride_ob + off_h * stride_oh

    # Device-side descriptors for TMA
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

    # Load Q block via TMA
    q = q_desc.load([start_m, 0])
    
    # Pre-scale Q with (sm_scale * log2(e)) to enable using tl.exp2 instead of tl.exp inside the loop
    q = (q * sm_scale_log2).to(tl.bfloat16)

    # Initialize softmax running statistics
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    end_m = start_m + BLOCK_M
    seq_limit = tl.minimum(S, end_m)
    num_steps = (seq_limit + BLOCK_N - 1) // BLOCK_N

    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    
    # Pre-compute combined sequence and causal limit bounds for a single optimized masking step
    limit = tl.minimum(S, offs_m + 1)

    # Single pipeline loop mapped optimally to WGMMA/TMA with tl.range
    for start_n_idx in tl.range(0, num_steps):
        start_n = start_n_idx * BLOCK_N
        
        # Load K and V blocks via TMA
        k = k_desc.load([start_n, 0])
        v = v_desc.load([start_n, 0])
        
        # First GEMM: Q @ K^T
        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        
        # Apply combined causal and sequence length mask dynamically
        col_indices = start_n + offs_n
        mask = col_indices[None, :] < limit[:, None]
        qk = tl.where(mask, qk, float("-inf"))
            
        # Compute online safe softmax (using fast base-2 arithmetic)
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp2(m_i - m_i_new)
        p = tl.exp2(qk - m_i_new[:, None])
        
        # Rescale accumulator and perform second GEMM: P @ V
        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(tl.bfloat16), v, out_dtype=tl.float32)
        
        # Update running statistics
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new

    # Epilogue: Normalize aggregated values
    inv_l_i = 1.0 / l_i
    acc = acc * inv_l_i[:, None]
    
    # Calculate the true LSE: convert the base-2 m_i back to natural log domain
    LN_2 = 0.6931471805599453
    lse = (m_i * LN_2) + tl.log(l_i)

    # Store normalized output via TMA (implicitly handles block bounding)
    o_desc.store([start_m, 0], acc.to(tl.bfloat16))

    # Manually store LSE stats safely handling sequence lengths
    lse_offset = off_b * stride_lseb + off_h * stride_lseh
    lse_ptrs = LSE + lse_offset + offs_m * stride_lses
    q_valid = offs_m < S
    tl.store(lse_ptrs, lse, mask=q_valid)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    
    # Configure the global allocator to furnish fast device-side descriptor storage
    triton.set_allocator(alloc_fn)
    
    B, H, S, D = Q.shape
    sm_scale = 1.0 / (D ** 0.5)
    
    # Compute the base-2 scale adjustment on the host to avoid doing it within the kernel
    LOG2_E = 1.4426950408889634
    sm_scale_log2 = sm_scale * LOG2_E

    # The chosen execution grid inherently establishes L2-optimized swizzling
    # layout for stream sharing among warps across M dimensions (K and V cache hits)
    grid = lambda META: (
        triton.cdiv(S, META['BLOCK_M']),
        B,
        H
    )

    _fwd_kernel[grid](
        Q, K, V, sm_scale_log2,
        O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        S,
        BLOCK_D=D,
    )