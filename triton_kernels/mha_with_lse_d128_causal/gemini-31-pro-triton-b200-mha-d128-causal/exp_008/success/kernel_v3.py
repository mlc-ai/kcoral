import torch
import triton
import triton.language as tl

def alloc_fn(size: int, alignment: int, stream):
    """Host device allocator for Triton TMA descriptor creation."""
    return torch.empty(size, device="cuda", dtype=torch.int8)

@triton.jit
def _attn_fwd_kernel(
    Q, K, V, O, LSE,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    B, H, S,
    SCALE,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    
    offset_m = (pid_m * BLOCK_M).to(tl.int32)
    # Early termination for out-of-bounds sequence tiles
    if offset_m >= S:
        return

    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    # Base pointers strictly scoped to the current batch and head 
    q_base = Q + pid_b * stride_qb + pid_h * stride_qh
    k_base = K + pid_b * stride_kb + pid_h * stride_kh
    v_base = V + pid_b * stride_vb + pid_h * stride_vh
    o_base = O + pid_b * stride_ob + pid_h * stride_oh

    # 2D TMA descriptors scoped over the sequence elements for batched hardware loading
    q_desc = tl.make_tensor_descriptor(
        q_base, shape=[S, D], strides=[stride_qs, 1],
        block_shape=[BLOCK_M, D], padding_option="zero"
    )
    k_desc = tl.make_tensor_descriptor(
        k_base, shape=[S, D], strides=[stride_ks, 1],
        block_shape=[BLOCK_N, D], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        v_base, shape=[S, D], strides=[stride_vs, 1],
        block_shape=[BLOCK_N, D], padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        o_base, shape=[S, D], strides=[stride_os, 1],
        block_shape=[BLOCK_M, D], padding_option="zero"
    )

    # Load queries via TMA natively
    q = q_desc.load([offset_m, 0])

    m_i = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, D), tl.float32)

    # Sequence tile limits and diagonal bound optimizations
    limit_full_causal = (pid_m * BLOCK_M) // BLOCK_N
    limit_full_s = S // BLOCK_N
    num_full_kv_tiles = tl.minimum(limit_full_causal, limit_full_s)

    # ===================================================================
    # 1. Main loop for NON-CAUSAL tiles (pure pipeline without masking)
    # ===================================================================
    for kv_tile in tl.range(0, num_full_kv_tiles, num_stages=2):
        offset_n = (kv_tile * BLOCK_N).to(tl.int32)
        
        k = k_desc.load([offset_n, 0])
        v = v_desc.load([offset_n, 0])

        scores_b2 = tl.dot(q, k.T) * SCALE
        
        m_ij = tl.maximum(m_i, tl.max(scores_b2, axis=1))
        
        alpha = tl.math.exp2(m_i - m_ij)
        p = tl.math.exp2(scores_b2 - m_ij[:, None])

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        
        m_i = m_ij

    # ===================================================================
    # 2. Causal / Sequence Boundary loop 
    # Handles overlap on the diagonal requiring sequence/causal boundary logic
    # ===================================================================
    limit_n = S if S < offset_m + BLOCK_M else offset_m + BLOCK_M
    num_total_kv_tiles = (limit_n + BLOCK_N - 1) // BLOCK_N
    
    offs_m = offset_m + tl.arange(0, BLOCK_M)

    for kv_tile in range(num_full_kv_tiles, num_total_kv_tiles):
        offset_n = (kv_tile * BLOCK_N).to(tl.int32)
        
        k = k_desc.load([offset_n, 0])
        v = v_desc.load([offset_n, 0])

        scores = tl.dot(q, k.T)
        
        # Strictly enforce causal masking and boundary protection
        offs_n_curr = offset_n + tl.arange(0, BLOCK_N)
        valid = (offs_m[:, None] >= offs_n_curr[None, :]) & (offs_n_curr[None, :] < S) & (offs_m[:, None] < S)
        scores_b2 = tl.where(valid, scores * SCALE, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(scores_b2, axis=1))
        safe_m_ij = tl.where(m_ij == float("-inf"), 0.0, m_ij)

        alpha = tl.math.exp2(m_i - safe_m_ij)
        p = tl.math.exp2(scores_b2 - safe_m_ij[:, None])

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        
        m_i = m_ij

    # Epilogue softmax normalizations
    safe_l_i = tl.where(l_i == 0.0, 1.0, l_i)
    out = acc / safe_l_i[:, None]

    # Convert metrics back to natural logarithmic scale for explicit contract
    LN2 = 0.6931471805599453
    lse_base2 = tl.where(l_i == 0.0, float("-inf"), m_i + tl.math.log2(safe_l_i))
    lse = lse_base2 * LN2

    # Return Output sequentially mapped out securely through TMA descriptor 
    o_desc.store([offset_m, 0], out.to(tl.bfloat16))

    # Masked standard writeback mapping for vector reduced metrics (LSE)
    lse_ptr = LSE + pid_b * stride_lseb + pid_h * stride_lseh + offs_m * stride_lses
    tl.store(lse_ptr, lse, mask=(offs_m < S))


def run(Q, K, V, O, LSE):
    """
    Dest-passing entry point for Causal Multi-Head Attention forward block.
    Receives preallocated inputs [Q, K, V] followed securely by outputs [O, LSE].
    
    Optimized strictly to exploit Blackwell architecture via:
    1. Maximum dimension block mappings combined with low stage counts for SMEM fit logic
    2. Causal strict overlapping slicing limits mapped for TMA instructions
    """
    torch.cuda.set_device(Q.device)
    # Required for device-level `make_tensor_descriptor` TMA routing logic
    triton.set_allocator(alloc_fn)
    
    B, H, S, D = Q.shape
    sm_scale = 1.0 / (D ** 0.5)
    
    # Mathematical scalar precalculated
    SCALE = sm_scale * 1.4426950408889634
    
    # 128x128 mapped to push full maximum parallel capability per iteration across SM100 warp
    BLOCK_M = 128
    BLOCK_N = 128
    
    grid = (triton.cdiv(S, BLOCK_M), B, H)
    
    _attn_fwd_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        B, H, S,
        SCALE,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        D=128,
        num_warps=8,
        num_stages=2
    )