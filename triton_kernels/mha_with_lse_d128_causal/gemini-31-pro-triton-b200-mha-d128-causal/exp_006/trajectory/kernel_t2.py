import torch
import triton
import triton.language as tl
import math

def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

@triton.jit
def _fwd_kernel(
    Q, K, V, O, LSE,
    stride_qb, stride_qh, stride_qs,
    stride_kb, stride_kh, stride_ks,
    stride_vb, stride_vh, stride_vs,
    stride_ob, stride_oh, stride_os,
    stride_lb, stride_lh, stride_ls,
    S,
    SCALE,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    start_m = tl.program_id(0) * BLOCK_M
    b = tl.program_id(1)
    h = tl.program_id(2)

    # Base pointers for this batch and head
    q_base = Q + b * stride_qb + h * stride_qh
    k_base = K + b * stride_kb + h * stride_kh
    v_base = V + b * stride_vb + h * stride_vh
    o_base = O + b * stride_ob + h * stride_oh

    # Create TMA descriptors
    q_desc = tl.make_tensor_descriptor(
        q_base, shape=[S, BLOCK_D], strides=[stride_qs, 1],
        block_shape=[BLOCK_M, BLOCK_D], padding_option="zero"
    )
    k_desc = tl.make_tensor_descriptor(
        k_base, shape=[S, BLOCK_D], strides=[stride_ks, 1],
        block_shape=[BLOCK_N, BLOCK_D], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        v_base, shape=[S, BLOCK_D], strides=[stride_vs, 1],
        block_shape=[BLOCK_N, BLOCK_D], padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        o_base, shape=[S, BLOCK_D], strides=[stride_os, 1],
        block_shape=[BLOCK_M, BLOCK_D], padding_option="zero"
    )

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # Load Q once via TMA and pre-scale in base-2
    q = q_desc.load([start_m, 0])
    q = (q.to(tl.float32) * SCALE).to(tl.bfloat16)

    # Calculate steps
    max_n = tl.minimum(S, start_m + BLOCK_M)
    num_steps = (max_n + BLOCK_N - 1) // BLOCK_N
    num_full_steps = tl.minimum(start_m // BLOCK_N, num_steps)

    # Fast path for blocks completely below the diagonal (no causal mask needed)
    for step in range(0, num_full_steps):
        start_n = step * BLOCK_N
        k = k_desc.load([start_n, 0])
        v = v_desc.load([start_n, 0])

        qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32)

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        
        p = tl.math.exp2(qk - m_ij[:, None])
        alpha = tl.math.exp2(m_i - m_ij)
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        m_i = m_ij

    # Coordinates for partial blocks
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    # Slow path for blocks overlapping the diagonal (requires causal/boundary mask)
    for step in range(num_full_steps, num_steps):
        start_n = step * BLOCK_N
        curr_offs_n = start_n + offs_n

        k = k_desc.load([start_n, 0])
        v = v_desc.load([start_n, 0])

        qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32)

        mask = (offs_m[:, None] >= curr_offs_n[None, :]) & (curr_offs_n[None, :] < S) & (offs_m[:, None] < S)
        qk = tl.where(mask, qk, -float("inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        
        # Safely handle completely masked rows to avoid inf - inf = NaN
        safe_m_ij = tl.where(m_ij == -float("inf"), 0.0, m_ij)
        
        p = tl.math.exp2(qk - safe_m_ij[:, None])
        alpha = tl.math.exp2(m_i - safe_m_ij)
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        m_i = m_ij

    safe_l_i = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / safe_l_i[:, None]
    
    # Convert LSE from base-2 back to natural logarithm
    LN2 = 0.6931471805599453
    lse_log2 = tl.where(l_i == 0.0, -float("inf"), m_i + tl.math.log2(safe_l_i))
    lse = lse_log2 * LN2

    # Store O via TMA (safely ignores out-of-bounds rows)
    o_desc.store([start_m, 0], acc.to(tl.bfloat16))

    # Store LSE using pointer math
    lse_ptrs = LSE + b * stride_lb + h * stride_lh + offs_m * stride_ls
    tl.store(lse_ptrs, lse, mask=offs_m < S)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    # Assert trailing dimension is contiguous for TMA requirement
    assert Q.stride(-1) == 1
    assert K.stride(-1) == 1
    assert V.stride(-1) == 1
    assert O.stride(-1) == 1
    
    # Scaling factor combined with base-2 logarithm scaling
    RCP_LN2 = 1.4426950408889634
    scale = (1.0 / math.sqrt(D)) * RCP_LN2
    
    BLOCK_M = 128
    BLOCK_N = 64
    grid = (triton.cdiv(S, BLOCK_M), B, H)
    
    # Set the allocator for device-created tensor descriptors
    triton.set_allocator(alloc_fn)
    
    _fwd_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        S,
        scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=D,
        num_warps=8,
        num_stages=3
    )