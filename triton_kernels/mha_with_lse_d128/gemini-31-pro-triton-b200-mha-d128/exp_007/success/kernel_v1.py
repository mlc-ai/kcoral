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


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256}, num_warps=8, num_stages=4),
    ],
    key=["S"],
)
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

    # Navigate to the specific batch and head for each pointer
    q_base = Q + b * stride_qb + h * stride_qh
    k_base = K + b * stride_kb + h * stride_kh
    v_base = V + b * stride_vb + h * stride_vh
    o_base = O + b * stride_ob + h * stride_oh
    
    # Create device-side tensor descriptors mapping the 2D slices [S, D]
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
    
    # Load Query tile and apply scaling + base-2 conversion outside the loop
    q = q_desc.load([offset_m, 0])
    q = (q * scale).to(tl.bfloat16)
    
    # Initialize online softmax state
    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, D), tl.float32)
    
    # Split execution into full blocks and a potential partial block 
    # to avoid sequence-tail mask evaluations inside the main hot loop.
    num_n_blocks_full = S // BLOCK_N
    
    # FULL BLOCKS LOOP
    for n_block in range(num_n_blocks_full):
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
        
    # PARTIAL BLOCK
    if (S % BLOCK_N) != 0:
        n_block = num_n_blocks_full
        offset_n = n_block * BLOCK_N
        k = k_desc.load([offset_n, 0])
        v = v_desc.load([offset_n, 0])
        
        scores = tl.dot(q, k.T)
        
        # Apply mask for sequence tail
        k_offsets = offset_n + tl.arange(0, BLOCK_N)
        mask = k_offsets[None, :] < S
        scores = tl.where(mask, scores, -float("inf"))
        
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        
        alpha = tl.math.exp2(m_i - m_ij)
        p = tl.math.exp2(scores - m_ij[:, None])
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        m_i = m_ij
        
    output = acc / l_i[:, None]
    
    # Convert base-2 LSE back to natural-log LSE
    LN2: tl.constexpr = 0.6931471805599453
    lse = (m_i + tl.math.log2(l_i)) * LN2
    
    # Store output tile using TMA descriptor (automatically ignores out-of-bounds rows)
    o_desc.store([offset_m, 0], output.to(tl.bfloat16))
    
    # Store LSE using masked pointer memory bounds
    m_offsets = offset_m + tl.arange(0, BLOCK_M)
    lse_mask = m_offsets < S
    lse_base = LSE + b * stride_lseb + h * stride_lseh
    tl.store(lse_base + m_offsets * stride_lses, lse, mask=lse_mask)


def run(Q, K, V, O, LSE):
    """Compute Non-Causal Multi-Head Attention forward with LSE."""
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    
    # Fold Softmax scale (1/sqrt(D)) and base-2 exponential scaling into Q
    RCP_LN2 = 1.4426950408889634
    scale = (1.0 / (D ** 0.5)) * RCP_LN2
    
    grid = lambda META: (triton.cdiv(S, META["BLOCK_M"]), B, H)
    
    _fwd_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        S,
        scale,
        D=D
    )