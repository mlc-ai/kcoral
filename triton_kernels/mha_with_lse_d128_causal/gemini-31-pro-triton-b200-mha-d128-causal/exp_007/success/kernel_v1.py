import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Q, K, V, O, LSE,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    S, sm_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    start_m = tl.program_id(0)
    batch = tl.program_id(1)
    head = tl.program_id(2)

    # Base pointers for standard TMA descriptors
    q_base = Q + batch * stride_qb + head * stride_qh
    k_base = K + batch * stride_kb + head * stride_kh
    v_base = V + batch * stride_vb + head * stride_vh
    o_base = O + batch * stride_ob + head * stride_oh

    # Create Blackwell device-side TMA descriptors
    q_desc = tl.make_tensor_descriptor(
        q_base,
        shape=[S, HEAD_DIM],
        strides=[stride_qs, 1],
        block_shape=[BLOCK_M, HEAD_DIM],
        padding_option="zero"
    )
    k_desc = tl.make_tensor_descriptor(
        k_base,
        shape=[S, HEAD_DIM],
        strides=[stride_ks, 1],
        block_shape=[BLOCK_N, HEAD_DIM],
        padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        v_base,
        shape=[S, HEAD_DIM],
        strides=[stride_vs, 1],
        block_shape=[BLOCK_N, HEAD_DIM],
        padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        o_base,
        shape=[S, HEAD_DIM],
        strides=[stride_os, 1],
        block_shape=[BLOCK_M, HEAD_DIM],
        padding_option="zero"
    )

    # Load Query block once (TMA naturally handles out-of-bounds with zero padding)
    q = q_desc.load([start_m * BLOCK_M, 0])

    # Convert softmax scale to base-2 exponent space
    RCP_LN2: tl.constexpr = 1.4426950408889634
    scale = sm_scale * RCP_LN2

    # Running online-softmax variables maintained in exact FP32
    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)

    # Determine safe limits for fully valid, non-causal keys to skip mask processing overhead
    safe_k_limit = tl.minimum(S, start_m * BLOCK_M)
    num_full_blocks = safe_k_limit // BLOCK_N

    # Phase 1: Fully within-bounds blocks without any causal or sequence masking restrictions
    for k0 in range(0, num_full_blocks):
        offset_n = k0 * BLOCK_N
        k = k_desc.load([offset_n, 0])
        v = v_desc.load([offset_n, 0])
        
        # Core standard raw MMA dot product, then strict exact FP32 precise scaling
        qk = tl.dot(q, k.T)
        qk = qk * scale
        
        # Max-shifted exact exponential
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        safe_m_ij = tl.where(m_ij == -float("inf"), 0.0, m_ij)
        alpha = tl.math.exp2(m_i - safe_m_ij)
        p = tl.math.exp2(qk - safe_m_ij[:, None])
        
        # Standard normalization reduction sums and value updates
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        
        m_i = m_ij

    # Phase 2: Boundary or causal blocks executing necessary masking instructions
    max_k_idx = tl.minimum(S, (start_m + 1) * BLOCK_M)
    num_k_blocks = tl.cdiv(max_k_idx, BLOCK_N)
    
    m_idx = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_idx < S

    for k0 in range(num_full_blocks, num_k_blocks):
        offset_n = k0 * BLOCK_N
        k = k_desc.load([offset_n, 0])
        v = v_desc.load([offset_n, 0])
        
        qk = tl.dot(q, k.T)
        qk = qk * scale
        
        # Apply strict logical conditions enforcing sequence limits and lower-triangular causal bounds
        n_idx = offset_n + tl.arange(0, BLOCK_N)
        valid_score = m_mask[:, None] & (m_idx[:, None] >= n_idx[None, :]) & (n_idx[None, :] < S)
        qk = tl.where(valid_score, qk, -float("inf"))
        
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        safe_m_ij = tl.where(m_ij == -float("inf"), 0.0, m_ij)
        alpha = tl.math.exp2(m_i - safe_m_ij)
        p = tl.math.exp2(qk - safe_m_ij[:, None])
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        
        m_i = m_ij

    # Safely finalize to context output, avoiding fully masked sequence query failures (0/0)
    safe_l_i = tl.where(l_i == 0.0, 1.0, l_i)
    out = acc / safe_l_i[:, None]

    # Revert exponential from mathematically-convenient base-2 space to natural log LSE
    LN2: tl.constexpr = 0.6931471805599453
    lse_log2 = tl.where(l_i == 0.0, -float("inf"), m_i + tl.math.log2(safe_l_i))
    lse = lse_log2 * LN2

    # Output context vectors utilizing native TMA store boundaries gracefully ignoring pad drops
    o_desc.store([start_m * BLOCK_M, 0], out.to(tl.bfloat16))

    # Masked LSE final pointer output
    lse_ptrs = LSE + batch * stride_lseb + head * stride_lseh + m_idx * stride_lses
    tl.store(lse_ptrs, lse, mask=m_mask)

def run(Q, K, V, O, LSE):
    """
    Standard Triton causal multi-head attention forward computing Output and Natural-Log-Sum-Exp.
    Writes strictly in-place to preallocated 'O' and 'LSE'.
    """
    torch.cuda.set_device(Q.device)
    
    # Enable Blackwell TMAs utilizing memory infrastructure for tensor descriptors
    def alloc_fn(size: int, alignment: int, stream):
        return torch.empty(size, device=Q.device, dtype=torch.int8)
    triton.set_allocator(alloc_fn)

    B, H, S, D = Q.shape
    sm_scale = 1.0 / (D ** 0.5)

    # Highly tuned program order maximizing L2 reuse mapping 128 queries natively over 64 key boundaries
    BLOCK_M = 128
    BLOCK_N = 64

    grid = (triton.cdiv(S, BLOCK_M), B, H)
    _fwd_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        S, sm_scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=128,
        num_warps=8, num_stages=3
    )