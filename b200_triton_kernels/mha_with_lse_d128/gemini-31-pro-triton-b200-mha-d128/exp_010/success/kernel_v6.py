import torch
import triton
import triton.language as tl

def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

@triton.autotune(
    configs=[
        # High arithmetic intensity configs optimized for Blackwell's large shared memory and TMEM
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
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

    # Decode batch and head
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    # Base pointers for the current batch and head
    q_base = Q + pid_b * stride_qb + pid_h * stride_qh
    k_base = K + pid_b * stride_kb + pid_h * stride_kh
    v_base = V + pid_b * stride_vb + pid_h * stride_vh
    o_base = O + pid_b * stride_ob + pid_h * stride_oh

    # Device-created Tensor descriptors mapping directly to TMA hardware
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

    # Initialize online softmax state variables
    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    k_tiles = tl.cdiv(S, BLOCK_N)
    
    # Completely specialize the loop behavior to eliminate dead ALU masking overhead
    if S_DIVISIBLE:
        # Fast path perfectly overlapping TMA and Matrix Math without interior bounds checks
        for kv_tile in range(0, k_tiles):
            offset_n = (kv_tile * BLOCK_N).to(tl.int32)
            
            k = k_desc.load([offset_n, 0])
            v = v_desc.load([offset_n, 0])

            scores = tl.dot(q, k.T, out_dtype=tl.float32) * scale_log2

            m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
            
            alpha = tl.math.exp2(m_i - m_ij)
            p = tl.math.exp2(scores - m_ij[:, None])

            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None]
            
            acc = tl.dot(p.to(tl.bfloat16), v, acc, out_dtype=tl.float32)
            
            m_i = m_ij
    else:
        # General safe path explicitly bounding trailing sequence elements
        offs_n = tl.arange(0, BLOCK_N)
        for kv_tile in range(0, k_tiles):
            offset_n = (kv_tile * BLOCK_N).to(tl.int32)
            
            k = k_desc.load([offset_n, 0])
            v = v_desc.load([offset_n, 0])

            scores = tl.dot(q, k.T, out_dtype=tl.float32) * scale_log2
            
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

    # Finalize softmax output scaling alongside exact natural-log LSE computation 
    if S_DIVISIBLE:
        rcp_l_i = 1.0 / l_i
        out = acc * rcp_l_i[:, None]
        lse_log2 = m_i + tl.math.log2(l_i)
        lse_ln = lse_log2 * LN2
        
        # O bounds dropping implicitly handled by TMA architecture
        o_desc.store([offset_m, 0], out.to(tl.bfloat16))

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        lse_ptrs = LSE + pid_b * stride_lseb + pid_h * stride_lseh + offs_m * stride_lses
        tl.store(lse_ptrs, lse_ln)
    else:
        safe_l_i = tl.where(l_i == 0.0, 1.0, l_i)
        rcp_l_i = 1.0 / safe_l_i
        out = acc * rcp_l_i[:, None]
        lse_log2 = tl.where(l_i == 0.0, -float("inf"), m_i + tl.math.log2(safe_l_i))
        lse_ln = lse_log2 * LN2

        o_desc.store([offset_m, 0], out.to(tl.bfloat16))

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        lse_ptrs = LSE + pid_b * stride_lseb + pid_h * stride_lseh + offs_m * stride_lses
        tl.store(lse_ptrs, lse_ln, mask=(offs_m < S))


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    
    # Configure the standard Triton infrastructure allocator required for dynamic descriptors
    triton.set_allocator(alloc_fn)
    
    B, H, S, D = Q.shape
    
    # Precompute logical scale into computationally aligned base-2 for optimized `tl.math.exp2`
    sm_scale = 1.0 / (D ** 0.5)
    RCP_LN2 = 1.4426950408889634
    scale_log2 = sm_scale * RCP_LN2
    
    # Compile-time shortcut eliminating masked operations internally when layout is uniform
    # We guarantee this safety mathematically since standard configs evaluate cleanly through multiples of 256
    S_DIVISIBLE = bool(S % 256 == 0)
    
    # Scheduling matrix coordinates: outer loop tracks spatial iteration sequence natively ordering 
    # consecutive query iterations identically along sequence offsets thus massively favoring L2 hit rates 
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