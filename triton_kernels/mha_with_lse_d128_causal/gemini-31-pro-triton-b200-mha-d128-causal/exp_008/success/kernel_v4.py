import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

@triton.jit
def _attn_fwd_kernel(
    q_desc, k_desc, v_desc, o_desc, LSE,
    stride_lseb, stride_lseh, stride_lses,
    S, SCALE,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr
):
    # Map execution to M blocks
    pid_m = tl.program_id(0)
    offset_m = pid_m * BLOCK_M
    
    # Early exit for sequence tail boundary
    if offset_m >= S:
        return

    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    # 4D Host Descriptor Load (avoids any device-side descriptor creation overhead)
    # The loaded tile has shape [1, 1, BLOCK_M, D] and is logically reshaped to [BLOCK_M, D]
    q_4d = q_desc.load([pid_b, pid_h, offset_m, 0])
    q = tl.reshape(q_4d, (BLOCK_M, D))

    # Initialize online softmax state and accumulators natively in FP32
    m_i = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, D), tl.float32)

    offs_m = offset_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    # Calculate optimal boundary limits to skip padding chunks and avoid full-causal tailing computations
    limit = offset_m + BLOCK_M
    max_n = S if S < limit else limit
    num_kv_tiles = (max_n + BLOCK_N - 1) // BLOCK_N

    # Highly pipelined loop optimized for unrolled native Blackwell memory scheduling
    for kv_tile in tl.range(0, num_kv_tiles, num_stages=3):
        offset_n = kv_tile * BLOCK_N
        
        # Load corresponding K and V chunks from descriptor pipelines
        k_4d = k_desc.load([pid_b, pid_h, offset_n, 0])
        v_4d = v_desc.load([pid_b, pid_h, offset_n, 0])
        
        k = tl.reshape(k_4d, (BLOCK_N, D))
        v = tl.reshape(v_4d, (BLOCK_N, D))

        # Core Blackwell accelerating MACs
        scores = tl.dot(q, k.T)
        
        # Strict causal bounds evaluation masking directly mapped without branch complexity
        causal_mask = offs_m[:, None] >= (offset_n + offs_n)[None, :]
        scores_b2 = tl.where(causal_mask, scores * SCALE, float("-inf"))

        # Reductions tracking
        m_ij = tl.maximum(m_i, tl.max(scores_b2, axis=1))
        safe_m_ij = tl.where(m_ij == float("-inf"), 0.0, m_ij)

        # Base-2 mapped exponentials avoiding standard library call limitations
        alpha = tl.math.exp2(m_i - safe_m_ij)
        p = tl.math.exp2(scores_b2 - safe_m_ij[:, None])

        # Step variables updates
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        
        m_i = m_ij

    # Epilogue standard normalizations
    safe_l_i = tl.where(l_i == 0.0, 1.0, l_i)
    out = acc / safe_l_i[:, None]

    # Remap tracked LSE back to mathematically required Base-E Natural Log space
    LN2 = 0.6931471805599453
    lse = tl.where(l_i == 0.0, float("-inf"), (m_i + tl.math.log2(safe_l_i)) * LN2)

    # Native dimensional reshaping directly emitted through TMA Hardware mapping bounds constraints
    out_4d = tl.reshape(out.to(tl.bfloat16), (1, 1, BLOCK_M, D))
    o_desc.store([pid_b, pid_h, offset_m, 0], out_4d)

    # Write-back 1D LSE vectors masked accurately for any striding configurations bounds
    lse_ptr = LSE + pid_b * stride_lseb + pid_h * stride_lseh + offs_m * stride_lses
    tl.store(lse_ptr, lse, mask=(offs_m < S))


def run(Q, K, V, O, LSE):
    """
    Destination-passing entry for Causal Multi-Head Attention standard execution logic.
    Receives mapped input formats [Q, K, V] followed directly by allocated pre-defined limits [O, LSE].

    Implemented extensively using full TMA mapping capability via `TensorDescriptor` created natively from
    Host CPU execution mitigating heavy Device side JIT synchronization/allocation penalties across loops.
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape

    # Balanced mapping capability optimized purely for memory bandwidth across SMs without specialization
    BLOCK_M = 128
    BLOCK_N = 128

    # Host constructed descriptors enforcing true multidimensional access without local loop creation bounds scaling mapping
    q_desc = TensorDescriptor.from_tensor(Q, [1, 1, BLOCK_M, D])
    k_desc = TensorDescriptor.from_tensor(K, [1, 1, BLOCK_N, D])
    v_desc = TensorDescriptor.from_tensor(V, [1, 1, BLOCK_N, D])
    o_desc = TensorDescriptor.from_tensor(O, [1, 1, BLOCK_M, D])

    # Transform scaling metrics mapping mathematically precalculated constants targeting execution bandwidth
    sm_scale = 1.0 / (D ** 0.5)
    SCALE = sm_scale * 1.4426950408889634

    # Native multi-layered grid distribution limits scaling blocks targeting all SM partitions limits
    grid = (triton.cdiv(S, BLOCK_M), B, H)
    
    _attn_fwd_kernel[grid](
        q_desc, k_desc, v_desc, o_desc, LSE,
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        S, SCALE,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, D=D,
        num_warps=8, num_stages=3
    )