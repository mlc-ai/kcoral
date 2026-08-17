import torch
import triton
import triton.language as tl

def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

@triton.autotune(
    configs=[
        # High arithmetic intensity configs optimized for Blackwell's large shared memory
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        # Standard configs for robust fallback
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
    ],
    key=['S'],
)
@triton.jit
def _fwd_kernel(
    Q, K, V, O, LSE,
    stride_qb, stride_qh, stride_qs,
    stride_kb, stride_kh, stride_ks,
    stride_vb, stride_vh, stride_vs,
    stride_ob, stride_oh, stride_os,
    stride_lseb, stride_lseh, stride_lses,
    S, H, scale_log2,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_bh = tl.program_id(1).to(tl.int64)

    pid_b = pid_bh // H
    pid_h = pid_bh % H

    # Base pointers for the current batch and head
    q_base = Q + pid_b * stride_qb + pid_h * stride_qh
    k_base = K + pid_b * stride_kb + pid_h * stride_kh
    v_base = V + pid_b * stride_vb + pid_h * stride_vh
    o_base = O + pid_b * stride_ob + pid_h * stride_oh

    # Create Tensor descriptors mapping directly to TMA hardware
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

    offset_m = (pid_m * BLOCK_M).to(tl.int32)
    q = q_desc.load([offset_m, 0])

    # Initialize online softmax state. Unpeeled loop guarantees m_ij is strictly finite.
    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    k_tiles = tl.cdiv(S, BLOCK_N)
    
    # Unpeeled loop for fully valid KV tiles. Completely avoids vector ALU masking in the critical path.
    for kv_tile in tl.range(0, k_tiles - 1, num_stages=3):
        offset_n = (kv_tile * BLOCK_N).to(tl.int32)
        
        k = k_desc.load([offset_n, 0])
        v = v_desc.load([offset_n, 0])

        qk = tl.dot(q, k.T, out_dtype=tl.float32) * scale_log2

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        
        alpha = tl.math.exp2(m_i - m_ij)
        p = tl.math.exp2(qk - m_ij[:, None])

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        
        acc = tl.dot(p.to(tl.bfloat16), v, acc, out_dtype=tl.float32)
        m_i = m_ij

    # Peeled loop for the last KV tile (which may be partial).
    # Executed unconditionally since S > 0 guarantees k_tiles >= 1.
    offset_n = ((k_tiles - 1) * BLOCK_N).to(tl.int32)
    k = k_desc.load([offset_n, 0])
    v = v_desc.load([offset_n, 0])

    qk = tl.dot(q, k.T, out_dtype=tl.float32) * scale_log2
    
    # Masking sequence tails just once
    offs_n = tl.arange(0, BLOCK_N)
    curr_offs_n = offset_n + offs_n
    qk = tl.where(curr_offs_n[None, :] < S, qk, -float("inf"))

    m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
    
    alpha = tl.math.exp2(m_i - m_ij)
    p = tl.math.exp2(qk - m_ij[:, None])

    l_i = l_i * alpha + tl.sum(p, axis=1)
    acc = acc * alpha[:, None]
    
    acc = tl.dot(p.to(tl.bfloat16), v, acc, out_dtype=tl.float32)
    m_i = m_ij

    # Finalize softmax and compute exact natural-log LSE inline
    out = acc / l_i[:, None]
    lse_log2 = m_i + tl.math.log2(l_i)
    
    LN2 = 0.6931471805599453
    lse_ln = lse_log2 * LN2

    # Store output O seamlessly bounded via TMA out-of-bounds drop mapping
    o_desc.store([offset_m, 0], out.to(tl.bfloat16))

    # Store output LSE explicitly masked for its 1D dimension
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    lse_ptrs = LSE + pid_b * stride_lseb + pid_h * stride_lseh + offs_m * stride_lses
    tl.store(lse_ptrs, lse_ln, mask=(offs_m < S))


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    
    # Configure the Triton descriptor allocator needed for device-created TMA descriptors.
    triton.set_allocator(alloc_fn)
    
    B, H, S, D = Q.shape
    
    # Precompute standard scale, converting to base-2 mathematically 
    # to naturally hit Triton's heavily optimized `exp2` routine inline.
    sm_scale = 1.0 / (D ** 0.5)
    RCP_LN2 = 1.4426950408889634
    scale_log2 = sm_scale * RCP_LN2
    
    # Grid construction clusters adjacent M query tiles into concurrent SMs, inherently utilizing
    # shared L2 cache efficiently for their universally identical K and V reads. 
    grid = lambda META: (
        triton.cdiv(S, META['BLOCK_M']),
        B * H,
    )
    
    _fwd_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        S, H, scale_log2,
        BLOCK_D=D,
    )