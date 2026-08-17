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
    pid_m = tl.program_id(0)
    offset_m = pid_m * BLOCK_M
    
    # Drop cleanly when out of bounds for the sequence M dimension
    if offset_m >= S:
        return

    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    # 4D TMA descriptor loading natively mapped to TTGIR logical shapes
    q_4d = q_desc.load([pid_b, pid_h, offset_m, 0])
    q = tl.reshape(q_4d, (BLOCK_M, D))

    # Initialize online softmax FP32 tracking
    m_i = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, D), tl.float32)

    offs_m = offset_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    
    # Optimization: Bound iteration natively eliminating loop tail bounds checking overheads
    limit = offset_m + BLOCK_M
    max_n = S if S < limit else limit
    num_kv_tiles = (max_n + BLOCK_N - 1) // BLOCK_N

    # Pipelined loop mapped aggressively to TMA scheduling with large stage counts
    for kv_tile in tl.range(0, num_kv_tiles, num_stages=4):
        offset_n = kv_tile * BLOCK_N
        
        k_4d = k_desc.load([pid_b, pid_h, offset_n, 0])
        v_4d = v_desc.load([pid_b, pid_h, offset_n, 0])
        
        k = tl.reshape(k_4d, (BLOCK_N, D))
        v = tl.reshape(v_4d, (BLOCK_N, D))

        # Core MM evaluation targeting FP32 accumulators
        scores = tl.dot(q, k.T, out_dtype=tl.float32) * SCALE
        
        # Reduced causal mask logic; drops sequence edge tracking since mathematically guaranteed bounded
        valid_score = offs_m[:, None] >= (offset_n + offs_n)[None, :]
        scores = tl.where(valid_score, scores, float("-inf"))

        # Reduction row steps
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        
        # Guard -inf when rows are heavily masked
        safe_m_ij = tl.where(m_ij == float("-inf"), 0.0, m_ij)
        
        alpha = tl.math.exp2(m_i - safe_m_ij)
        p = tl.math.exp2(scores - safe_m_ij[:, None])

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc, out_dtype=tl.float32)
        
        m_i = m_ij

    # Epilogue standard FP32 normalization
    safe_l_i = tl.where(l_i == 0.0, 1.0, l_i)
    out = acc / safe_l_i[:, None]

    # Recalculate metrics mapping into Log-E required space (Natural Log)
    LN2 = 0.6931471805599453
    lse = tl.where(l_i == 0.0, float("-inf"), (m_i + tl.math.log2(safe_l_i)) * LN2)

    # Dimensional format reshaping to utilize native TMA dimensional block storage padding limits
    out_4d = tl.reshape(out.to(tl.bfloat16), (1, 1, BLOCK_M, D))
    o_desc.store([pid_b, pid_h, offset_m, 0], out_4d)

    # Standard pointer mapped vector logic logic limits
    lse_ptr = LSE + pid_b * stride_lseb + pid_h * stride_lseh + offs_m * stride_lses
    tl.store(lse_ptr, lse, mask=(offs_m < S))


def run(Q, K, V, O, LSE):
    """
    Destination-passing evaluation for standard Causal Multi-Head Attention logic mapping targeting SM100.
    Implements FlashAttention strictly over `TensorDescriptor` avoiding pointer evaluation instructions for 
    main logic pipeline segments.
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    
    # 128x64 balanced scaling pushes extremely safe Shared Memory usage bounds across 4 heavily pipelined stages
    BLOCK_M = 128
    BLOCK_N = 64

    # Host construction removes heavy local TTGIR instruction emission scaling required for device instantiations
    q_desc = TensorDescriptor.from_tensor(Q, [1, 1, BLOCK_M, D])
    k_desc = TensorDescriptor.from_tensor(K, [1, 1, BLOCK_N, D])
    v_desc = TensorDescriptor.from_tensor(V, [1, 1, BLOCK_N, D])
    o_desc = TensorDescriptor.from_tensor(O, [1, 1, BLOCK_M, D])

    # Scales mathematically combined mapping directly with LN2 tracking constants
    sm_scale = 1.0 / (D ** 0.5)
    SCALE = sm_scale * 1.4426950408889634

    # Standard mapped limits evaluated over the fastest M bounds
    grid = (triton.cdiv(S, BLOCK_M), B, H)
    
    _attn_fwd_kernel[grid](
        q_desc, k_desc, v_desc, o_desc, LSE,
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        S, SCALE,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, D=D,
        num_warps=4, num_stages=4
    )