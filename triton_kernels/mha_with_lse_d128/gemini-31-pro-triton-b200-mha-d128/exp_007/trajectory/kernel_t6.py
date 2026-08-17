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
        # BLOCK_M=256 massively reduces K and V memory traffic, which can bottleneck large sequence kernels.
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "NUM_STAGES": 2}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "NUM_STAGES": 3}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "NUM_STAGES": 4}, num_warps=8, num_stages=4),
        
        # Standard configs
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "NUM_STAGES": 3}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "NUM_STAGES": 3}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "NUM_STAGES": 4}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "NUM_STAGES": 4}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "NUM_STAGES": 5}, num_warps=8, num_stages=5),
        
        # Fallback dense configs
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "NUM_STAGES": 3}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "NUM_STAGES": 4}, num_warps=4, num_stages=4),
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
    S, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, NUM_STAGES: tl.constexpr, D: tl.constexpr
):
    b = tl.program_id(1)
    h = tl.program_id(2)
    m_block = tl.program_id(0)
    
    offset_m = m_block * BLOCK_M

    # Advanced pointers to exactly the batch/head physical layout offset slice
    q_base = Q + b * stride_qb + h * stride_qh
    k_base = K + b * stride_kb + h * stride_kh
    v_base = V + b * stride_vb + h * stride_vh
    o_base = O + b * stride_ob + h * stride_oh
    
    # TMA natively bounds-checked and cached memory abstractions on Blackwell
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
    
    # Pre-load and globally pre-scale the Query matrix outside the hot KV loop 
    q = q_desc.load([offset_m, 0])
    q = (q * scale).to(tl.bfloat16)
    
    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, D), tl.float32)
    
    num_n_blocks_full = S // BLOCK_N
    
    # === FULL BLOCKS HOT LOOP ===
    # Utilizing specific tl.range with explicit software pipeline stages
    for n_block in tl.range(0, num_n_blocks_full, num_stages=NUM_STAGES):
        offset_n = n_block * BLOCK_N
        k = k_desc.load([offset_n, 0])
        v = v_desc.load([offset_n, 0])
        
        # Explicit out_dtype ensures robust accumulation in single-precision mapping to standard hardware FP32 accumulators
        scores = tl.dot(q, k.T, out_dtype=tl.float32)
        
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        
        alpha = tl.math.exp2(m_i - m_ij)
        p = tl.math.exp2(scores - m_ij[:, None])
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        m_i = m_ij

    # === PARTIAL TAIL BOUNDARY ===
    if (S % BLOCK_N) != 0:
        offset_n = num_n_blocks_full * BLOCK_N
        k = k_desc.load([offset_n, 0])
        v = v_desc.load([offset_n, 0])
        
        scores = tl.dot(q, k.T, out_dtype=tl.float32)
        
        # Apply standard padding-mask boundary logic to exclude invalid elements from contributing
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

    # Safely prevent NaN propagation completely empty rows
    safe_l_i = tl.where(l_i == 0.0, 1.0, l_i)
    output = acc / safe_l_i[:, None]
    
    # Bounds-checked storage automatically handled safely by the TMA descriptor hardware mapping
    o_desc.store([offset_m, 0], output.to(tl.bfloat16))
    
    # Translate base-2 Log-Sum-Exp mapping mathematically securely onto conventional Standard Natural Log Format 
    LN2: tl.constexpr = 0.6931471805599453
    lse = (m_i + tl.math.log2(safe_l_i)) * LN2
    
    m_offsets = offset_m + tl.arange(0, BLOCK_M)
    lse_mask = m_offsets < S
    lse_base = LSE + b * stride_lseb + h * stride_lseh
    tl.store(lse_base + m_offsets * stride_lses, lse, mask=lse_mask)


def run(Q, K, V, O, LSE):
    """Compute Non-Causal Multi-Head Attention forward returning O and LSE in standard Flash Attention style."""
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    
    # Fold softmax scale logically coupled with Base-2 exponential hardware transformation
    RCP_LN2 = 1.4426950408889634
    scale = (1.0 / (D ** 0.5)) * RCP_LN2
    
    # A standard 3D grid layout automatically takes advantage of L2 cache reuse across SMs
    # because program_id(0) sequences across M-blocks naturally maximizing K/V block sharing per head.
    grid = lambda META: (triton.cdiv(S, META["BLOCK_M"]), B, H)
    
    _fwd_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        S, scale,
        D=D
    )