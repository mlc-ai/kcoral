import torch
import triton
import triton.language as tl

def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

def get_autotune_config():
    return [
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
    ]

@triton.autotune(
    configs=get_autotune_config(),
    key=['S']
)
@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale,
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
    off_b = tl.program_id(1)
    off_h = tl.program_id(2)

    # Base pointers for this batch and head
    q_ptr = Q + off_b * stride_qb + off_h * stride_qh
    k_ptr = K + off_b * stride_kb + off_h * stride_kh
    v_ptr = V + off_b * stride_vb + off_h * stride_vh
    o_ptr = O + off_b * stride_ob + off_h * stride_oh

    # Create device-side descriptors for TMA
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

    # Initialize running state
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    end_m = start_m + BLOCK_M
    seq_limit = tl.minimum(S, end_m)
    num_steps = (seq_limit + BLOCK_N - 1) // BLOCK_N

    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    
    # Pre-compute query mask for storing LSE
    q_valid = offs_m < S

    for start_n_idx in range(0, num_steps):
        start_n = start_n_idx * BLOCK_N
        
        # Load K and V blocks via TMA
        k = k_desc.load([start_n, 0])
        v = v_desc.load([start_n, 0])
        
        # First GEMM: Q @ K^T
        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        qk = qk * sm_scale
        
        # Apply causal mask for the blocks intersecting the diagonal
        if start_n + BLOCK_N > start_m:
            causal_mask = (start_n + offs_n)[None, :] <= offs_m[:, None]
            qk = tl.where(causal_mask, qk, float("-inf"))
            
        # Apply sequence length mask for the last block
        if start_n + BLOCK_N > S:
            valid_mask = (start_n + offs_n)[None, :] < S
            qk = tl.where(valid_mask, qk, float("-inf"))
            
        # Compute softmax running statistics
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])
        
        # Scale accumulator and perform second GEMM: P @ V
        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(q.dtype), v, out_dtype=tl.float32)
        
        # Update running statistics
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new

    # Epilogue: normalize with log-sum-exp
    acc = acc / l_i[:, None]
    lse = m_i + tl.log(l_i)

    # Store output via TMA
    o_desc.store([start_m, 0], acc.to(q.dtype))

    # Store LSE manually (since it's a 1D per-sequence output)
    lse_offset = off_b * stride_lseb + off_h * stride_lseh
    lse_ptrs = LSE + lse_offset + offs_m * stride_lses
    tl.store(lse_ptrs, lse, mask=q_valid)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    
    # Configure the global allocator for device-side descriptor creation
    triton.set_allocator(alloc_fn)
    
    B, H, S, D = Q.shape
    sm_scale = 1.0 / (D ** 0.5)

    grid = lambda META: (
        triton.cdiv(S, META['BLOCK_M']),
        B,
        H
    )

    _fwd_kernel[grid](
        Q, K, V, sm_scale,
        O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        S,
        BLOCK_D=D,
    )