import torch
import triton
import triton.language as tl

# Configure Triton's allocator for device-side tensor descriptors
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

try:
    triton.set_allocator(alloc_fn)
except Exception:
    pass


@triton.jit
def _fwd_kernel(
    Q, K, V, O, LSE,
    stride_qb, stride_qh, stride_qs,
    stride_kb, stride_kh, stride_ks,
    stride_vb, stride_vh, stride_vs,
    stride_ob, stride_oh, stride_os,
    stride_lseb, stride_lseh, stride_lses,
    S,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
):
    b = tl.program_id(1)
    h = tl.program_id(2)
    m_block = tl.program_id(0)
    
    offset_m = m_block * BLOCK_M

    # Base pointers for this specific batch and head
    q_base = Q + b * stride_qb + h * stride_qh
    k_base = K + b * stride_kb + h * stride_kh
    v_base = V + b * stride_vb + h * stride_vh
    o_base = O + b * stride_ob + h * stride_oh
    
    # Create 2D device descriptors for the [S, D] slices
    # TMA descriptors are ideal for Blackwell, ensuring highly efficient boundary checks and cache policies.
    q_desc = tl.make_tensor_descriptor(
        q_base, shape=[S, D], strides=[stride_qs, 1], block_shape=[BLOCK_M, D], padding_option="zero"
    )
    k_desc = tl.make_tensor_descriptor(
        k_base, shape=[S, D], strides=[stride_ks, 1], block_shape=[BLOCK_N, D], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        v_base, shape=[S, D], strides=[stride_vs, 1], block_shape=[BLOCK_N, D], padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        o_base, shape=[S, D], strides=[stride_os, 1], block_shape=[BLOCK_M, D], padding_option="zero"
    )
    
    # Load Q tile
    q = q_desc.load([offset_m, 0])
    
    # Apply combined Softmax scale + Base-2 conversion scale statically outside the loop
    q = (q * scale).to(tl.bfloat16)
    
    # Initialize online softmax state
    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, D), tl.float32)
    
    num_n_blocks_full = S // BLOCK_N
    
    # FULL BLOCKS LOOP (avoids sequence tail masking overhead inside the core loop)
    for n_block in tl.range(0, num_n_blocks_full):
        offset_n = n_block * BLOCK_N
        
        k = k_desc.load([offset_n, 0])
        v = v_desc.load([offset_n, 0])
        
        scores = tl.dot(q, k.T)
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        
        alpha = tl.math.exp2(m_i - m_ij)
        p = tl.math.exp2(scores - m_ij[:, None])
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        m_i = m_ij

    # PARTIAL BLOCK LOOP (handles any remainder K/V tail sequences)
    if (S % BLOCK_N) != 0:
        offset_n = num_n_blocks_full * BLOCK_N
        
        k = k_desc.load([offset_n, 0])
        v = v_desc.load([offset_n, 0])
        
        scores = tl.dot(q, k.T)
        
        # Mask out out-of-bounds keys manually for partial blocks
        k_offsets = offset_n + tl.arange(0, BLOCK_N)
        mask = k_offsets[None, :] < S
        scores = tl.where(mask, scores, -float("inf"))
        
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        safe_m_ij = tl.where(m_ij == -float("inf"), 0.0, m_ij)
        
        alpha = tl.math.exp2(m_i - safe_m_ij)
        p = tl.math.exp2(scores - safe_m_ij[:, None])
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        m_i = m_ij

    safe_l_i = tl.where(l_i == 0.0, 1.0, l_i)
    output = acc / safe_l_i[:, None]
    
    # Store Output Tile
    # Using TMA descriptor handles bounds safely inside hardware without pointers
    o_desc.store([offset_m, 0], output.to(tl.bfloat16))
    
    # Convert Base-2 LSE to Natural Log LSE
    LN2: tl.constexpr = 0.6931471805599453
    lse = (m_i + tl.math.log2(safe_l_i)) * LN2
    lse = tl.where(l_i == 0.0, -float("inf"), lse)
    
    # Store LSE securely respecting bounds
    m_offsets = offset_m + tl.arange(0, BLOCK_M)
    lse_mask = m_offsets < S
    lse_base = LSE + b * stride_lseb + h * stride_lseh
    tl.store(lse_base + m_offsets * stride_lses, lse, mask=lse_mask)


def run(Q, K, V, O, LSE):
    """Compute Non-Causal Multi-Head Attention forward with LSE using standard Triton pattern."""
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    
    # Precompute Softmax scaling combined with natural-to-base-2 exponent translation multiplier
    RCP_LN2 = 1.4426950408889634
    scale = (1.0 / (D ** 0.5)) * RCP_LN2
    
    # High-Performance Golden Configuration for NVIDIA Blackwell Tensor Core Math
    BLOCK_M = 128
    BLOCK_N = 64
    
    grid = (triton.cdiv(S, BLOCK_M), B, H)
    
    _fwd_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        S, scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        D=D,
        num_warps=8,
        num_stages=4
    )