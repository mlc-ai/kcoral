import torch
import triton
import triton.language as tl

def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
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
    S_DIVISIBLE: tl.constexpr,
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

    # Create Tensor descriptors perfectly aligning physical sequence layout to TMA
    q_desc = tl.make_tensor_descriptor(
        q_base,
        shape=[S, BLOCK_D],
        strides=[stride_qs, 1],
        block_shape=[BLOCK_M, BLOCK_D],
        padding_option="zero"
    )
    k_desc = tl.make_tensor_descriptor(
        k_base,
        shape=[S, BLOCK_D],
        strides=[stride_ks, 1],
        block_shape=[BLOCK_N, BLOCK_D],
        padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        v_base,
        shape=[S, BLOCK_D],
        strides=[stride_vs, 1],
        block_shape=[BLOCK_N, BLOCK_D],
        padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        o_base,
        shape=[S, BLOCK_D],
        strides=[stride_os, 1],
        block_shape=[BLOCK_M, BLOCK_D],
        padding_option="zero"
    )

    offset_m = (pid_m * BLOCK_M).to(tl.int32)
    q = q_desc.load([offset_m, 0])

    # Initialize online softmax state
    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    k_tiles = tl.cdiv(S, BLOCK_N)
    
    if not S_DIVISIBLE:
        offs_n = tl.arange(0, BLOCK_N)

    # Standard python range seamlessly mapped to launch parameter `num_stages` software pipelining
    for kv_tile in range(0, k_tiles):
        offset_n = (kv_tile * BLOCK_N).to(tl.int32)
        
        # Core TMA stream
        k = k_desc.load([offset_n, 0])
        v = v_desc.load([offset_n, 0])

        # Q @ K.T
        scores = tl.dot(q, k.T, out_dtype=tl.float32) * scale_log2
        
        if S_DIVISIBLE:
            # Fast path completely avoiding interior loop vector ALU masking
            m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
            alpha = tl.math.exp2(m_i - m_ij)
            p = tl.math.exp2(scores - m_ij[:, None])
            
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None]
            
            acc = tl.dot(p.to(tl.bfloat16), v, acc, out_dtype=tl.float32)
            m_i = m_ij
        else:
            # Safe branch explicitly bounding sequence tails avoiding padded dots propagating
            curr_offs_n = offset_n + offs_n
            scores = tl.where(curr_offs_n[None, :] < S, scores, -float("inf"))

            m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
            safe_m_ij = tl.where(m_ij == -float("inf"), 0.0, m_ij)
            
            alpha = tl.math.exp2(m_i - safe_m_ij)
            p = tl.math.exp2(scores - safe_m_ij[:, None])

            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None]
            
            acc = tl.dot(p.to(tl.bfloat16), v, acc, out_dtype=tl.float32)
            m_i = m_ij

    LN2 = 0.6931471805599453

    if S_DIVISIBLE:
        # Avoid explicit safe l_i checks since fully uniform blocks guarantee l_i >= 1.0 
        out = acc / l_i[:, None]
        lse_log2 = m_i + tl.math.log2(l_i)
        lse_ln = lse_log2 * LN2
        
        # Store O implicitly bounds dropped via TMA descriptor bounds logic 
        o_desc.store([offset_m, 0], out.to(tl.bfloat16))

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        lse_ptrs = LSE + pid_b * stride_lseb + pid_h * stride_lseh + offs_m * stride_lses
        tl.store(lse_ptrs, lse_ln)
    else:
        safe_l_i = tl.where(l_i == 0.0, 1.0, l_i)
        out = acc / safe_l_i[:, None]
        lse_log2 = tl.where(l_i == 0.0, -float("inf"), m_i + tl.math.log2(safe_l_i))
        lse_ln = lse_log2 * LN2

        o_desc.store([offset_m, 0], out.to(tl.bfloat16))

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        lse_ptrs = LSE + pid_b * stride_lseb + pid_h * stride_lseh + offs_m * stride_lses
        tl.store(lse_ptrs, lse_ln, mask=(offs_m < S))


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    
    # Configure the Triton descriptor allocator needed for device-created descriptors.
    triton.set_allocator(alloc_fn)
    
    B, H, S, D = Q.shape
    
    # Precompute standard scale, converting to base-2 mathematically 
    # to hit Triton's optimized `exp2` routine inline
    sm_scale = 1.0 / (D ** 0.5)
    RCP_LN2 = 1.4426950408889634
    scale_log2 = sm_scale * RCP_LN2
    
    # Maximum config parameter BLOCK_M or BLOCK_N size determines global divisibility safety
    S_DIVISIBLE = (S % 256 == 0)
    
    # Grouping pid_m on the inner dimension to maximize L2 hit rates for K and V CTAs across sequence
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
        S_DIVISIBLE,
        BLOCK_D=D,
    )