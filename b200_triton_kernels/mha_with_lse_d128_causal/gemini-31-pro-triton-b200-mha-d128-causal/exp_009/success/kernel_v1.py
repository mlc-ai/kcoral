import torch
import triton
import triton.language as tl
import math

def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

def get_autotune_config():
    return [
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=5),
    ]

@triton.autotune(
    configs=get_autotune_config(),
    key=['S'],
)
@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale_log2,
    O, LSE,
    stride_qb, stride_qh, stride_qs,
    stride_kb, stride_kh, stride_ks,
    stride_vb, stride_vh, stride_vs,
    stride_ob, stride_oh, stride_os,
    stride_lse_b, stride_lse_h, stride_lse_s,
    S,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    if pid_m * BLOCK_M >= S:
        return

    q_offset = pid_b * stride_qb + pid_h * stride_qh
    k_offset = pid_b * stride_kb + pid_h * stride_kh
    v_offset = pid_b * stride_vb + pid_h * stride_vh
    o_offset = pid_b * stride_ob + pid_h * stride_oh

    # Create TMA descriptors for efficient memory access
    q_desc = tl.make_tensor_descriptor(
        Q + q_offset,
        shape=[S, HEAD_DIM],
        strides=[stride_qs, 1],
        block_shape=[BLOCK_M, HEAD_DIM],
        padding_option="zero",
    )
    k_desc = tl.make_tensor_descriptor(
        K + k_offset,
        shape=[S, HEAD_DIM],
        strides=[stride_ks, 1],
        block_shape=[BLOCK_N, HEAD_DIM],
        padding_option="zero",
    )
    v_desc = tl.make_tensor_descriptor(
        V + v_offset,
        shape=[S, HEAD_DIM],
        strides=[stride_vs, 1],
        block_shape=[BLOCK_N, HEAD_DIM],
        padding_option="zero",
    )
    o_desc = tl.make_tensor_descriptor(
        O + o_offset,
        shape=[S, HEAD_DIM],
        strides=[stride_os, 1],
        block_shape=[BLOCK_M, HEAD_DIM],
        padding_option="zero",
    )

    # Load Q tile once
    q = q_desc.load([pid_m * BLOCK_M, 0])

    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

    # Calculate exact unmasked and masked loop boundaries
    max_unmasked_kv_tiles = (pid_m * BLOCK_M) // BLOCK_N
    max_possible_unmasked = S // BLOCK_N
    if max_unmasked_kv_tiles > max_possible_unmasked:
        max_unmasked_kv_tiles = max_possible_unmasked

    max_kv_tiles = ((pid_m + 1) * BLOCK_M + BLOCK_N - 1) // BLOCK_N
    max_possible_kv_tiles = (S + BLOCK_N - 1) // BLOCK_N
    if max_kv_tiles > max_possible_kv_tiles:
        max_kv_tiles = max_possible_kv_tiles

    # Fully unmasked tiles (causal constraint strictly true and entirely in-bounds)
    for kv_tile in range(0, max_unmasked_kv_tiles):
        offset_n = kv_tile * BLOCK_N
        k = k_desc.load([offset_n, 0])
        v = v_desc.load([offset_n, 0])

        # Using out_dtype=tl.float32 allows keeping the accumulator in FP32
        scores = tl.dot(q, k.T, out_dtype=tl.float32) * sm_scale_log2
        
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        safe_m_ij = tl.where(m_ij == float("-inf"), 0.0, m_ij)

        alpha = tl.math.exp2(m_i - safe_m_ij)
        p = tl.math.exp2(scores - safe_m_ij[:, None])

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        m_i = m_ij

    # Partially masked tiles
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    
    for kv_tile in range(max_unmasked_kv_tiles, max_kv_tiles):
        offset_n = kv_tile * BLOCK_N
        k = k_desc.load([offset_n, 0])
        v = v_desc.load([offset_n, 0])
        
        scores = tl.dot(q, k.T, out_dtype=tl.float32) * sm_scale_log2
        
        kv_indices = offset_n + offs_n
        causal_mask = (offs_m[:, None] >= kv_indices[None, :]) & (offs_m[:, None] < S) & (kv_indices[None, :] < S)
        
        scores = tl.where(causal_mask, scores, float("-inf"))
        
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        safe_m_ij = tl.where(m_ij == float("-inf"), 0.0, m_ij)

        alpha = tl.math.exp2(m_i - safe_m_ij)
        p = tl.math.exp2(scores - safe_m_ij[:, None])

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        m_i = m_ij

    # Normalize output
    safe_l_i = tl.where(l_i == 0.0, 1.0, l_i)
    output = acc / safe_l_i[:, None]
    
    # Store Attention Output via descriptor (handles bounds padding automatically)
    o_desc.store([pid_m * BLOCK_M, 0], output.to(tl.bfloat16))

    # Compute and Store LSE (natural logarithm expected)
    lse_log2 = tl.where(l_i == 0.0, float("-inf"), m_i + tl.math.log2(safe_l_i))
    lse = lse_log2 * 0.6931471805599453

    lse_offset = pid_b * stride_lse_b + pid_h * stride_lse_h
    lse_ptrs = LSE + lse_offset + offs_m * stride_lse_s
    lse_mask = offs_m < S
    tl.store(lse_ptrs, lse, mask=lse_mask)

def run(Q, K, V, O, LSE):
    """
    Computes Causal Multi-Head Attention forward pass using TMA descriptors
    and writing out O and LSE. 
    """
    torch.cuda.set_device(Q.device)
    
    # Needs to be set for the device-created descriptors logic in Triton standard implementation
    triton.set_allocator(alloc_fn)
    
    B, H, S, D = Q.shape

    # Fast exponential uses base 2, absorb the constant in the upfront scaling
    sm_scale_log2 = (1.0 / math.sqrt(D)) * 1.4426950408889634
    
    # Launch configuration
    grid = lambda META: (triton.cdiv(S, META['BLOCK_M']), B, H)
    
    _fwd_kernel[grid](
        Q, K, V, sm_scale_log2,
        O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        S,
        HEAD_DIM=D
    )