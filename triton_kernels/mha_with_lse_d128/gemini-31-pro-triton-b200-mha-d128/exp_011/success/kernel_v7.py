import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "NUM_STAGES": 2}, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "NUM_STAGES": 3}, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "NUM_STAGES": 3}, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "NUM_STAGES": 4}, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "NUM_STAGES": 3}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "NUM_STAGES": 4}, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "NUM_STAGES": 4}, num_warps=4),
    ],
    key=["S"],
)
@triton.jit
def _attention_kernel(
    Q, K, V, O, LSE,
    S,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    sm_scale_log2,
    EVEN_S: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    batch_idx = tl.program_id(1)
    head_idx = tl.program_id(2)
    m_block_idx = tl.program_id(0)

    # Calculate baseline alignments mapping to independent sequence lengths
    q_base = Q + batch_idx * stride_qb + head_idx * stride_qh
    k_base = K + batch_idx * stride_kb + head_idx * stride_kh
    v_base = V + batch_idx * stride_vb + head_idx * stride_vh

    # Instantiate hardware-bound 2D Device Descriptors
    q_desc = tl.make_tensor_descriptor(
        q_base, shape=[S, 128], strides=[stride_qs, 1], block_shape=[BLOCK_M, 128], padding_option="zero"
    )
    k_desc = tl.make_tensor_descriptor(
        k_base, shape=[S, 128], strides=[stride_ks, 1], block_shape=[BLOCK_N, 128], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        v_base, shape=[S, 128], strides=[stride_vs, 1], block_shape=[BLOCK_N, 128], padding_option="zero"
    )

    offset_m = (m_block_idx * BLOCK_M).to(tl.int32)
    
    # Pre-scale Q explicitly enabling the matrix accumulation loop below to stream seamlessly
    q = q_desc.load([offset_m, 0])
    q = (q * sm_scale_log2).to(q.dtype)

    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, 128), tl.float32)

    offs_m = offset_m + tl.arange(0, BLOCK_M)
    q_valid = offs_m < S
    k_tiles = tl.cdiv(S, BLOCK_N)
    
    # Establish unified software pipelining (TMA -> TMEM/MMA overlapping)
    for kv_tile in tl.range(0, k_tiles, num_stages=NUM_STAGES):
        offset_n = (kv_tile * BLOCK_N).to(tl.int32)
        
        # Dispatch TMA 
        k = k_desc.load([offset_n, 0])
        v = v_desc.load([offset_n, 0])

        # Compute dot mapped to Tensor Cores internally
        scores = tl.dot(q, k.T, out_dtype=tl.float32)
        
        # Completely bypass masking bounds logic statically if sequence permits
        if not EVEN_S:
            offs_n = offset_n + tl.arange(0, BLOCK_N)
            valid_score = q_valid[:, None] & (offs_n[None, :] < S)
            scores = tl.where(valid_score, scores, -float("inf"))

        # Reductions resolved algorithmically optimized against standard Safe bounds
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.math.exp2(m_i - m_ij)
        p = tl.math.exp2(scores - m_ij[:, None])

        # Stream vector scaling outputs 
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(q.dtype), v.to(q.dtype), acc, out_dtype=tl.float32)
        m_i = m_ij

    # Guarantee final bounds since valid input sequences S > 0 prevent complete -inf collapses
    output = acc / l_i[:, None]
    
    # Convert natively tracked Base-2 logarithm metrics to Natural logarithm norms
    lse_log2 = m_i + tl.math.log2(l_i)
    LN2: tl.constexpr = 0.6931471805599453
    lse = lse_log2 * LN2

    # Map sequential memory writeback explicitly avoiding arbitrary 4D pointer complications
    o_base = O + batch_idx * stride_ob + head_idx * stride_oh
    offs_d = tl.arange(0, 128)
    o_ptrs = o_base + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(o_ptrs, output.to(q.dtype), mask=q_valid[:, None])

    lse_base = LSE + batch_idx * stride_lseb + head_idx * stride_lseh
    lse_ptrs = lse_base + offs_m * stride_lses
    tl.store(lse_ptrs, lse, mask=q_valid)


def run(Q, K, V, O, LSE):
    """
    Computes a High-Performance Multi-Head Attention forward pass returning Outputs and LogSumExp vectors.
    """
    torch.cuda.set_device(Q.device)
    
    # Prime Triton allocator ensuring native allocations exclusively target device-side mapping hooks
    def alloc_fn(size: int, alignment: int, stream):
        return torch.empty(size, device=Q.device, dtype=torch.int8)
    triton.set_allocator(alloc_fn)

    B, H, S, D = Q.shape
    
    sm_scale = 1.0 / (D ** 0.5)
    RCP_LN2 = 1.4426950408889634
    sm_scale_log2 = sm_scale * RCP_LN2
    
    EVEN_S = (S % 128 == 0)
    
    # Q tiles map fastest promoting subsequent block K, V hits directly within L2 cache buffers
    grid = lambda META: (triton.cdiv(S, META["BLOCK_M"]), B, H)

    _attention_kernel[grid](
        Q, K, V, O, LSE,
        S,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        sm_scale_log2,
        EVEN_S=EVEN_S,
    )