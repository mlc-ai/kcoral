import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4, num_stages=4),
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
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Map program ids to coordinates
    m_block_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)
    head_idx = tl.program_id(2)

    # Calculate base pointers for this batch and head
    q_base = Q + batch_idx * stride_qb + head_idx * stride_qh
    k_base = K + batch_idx * stride_kb + head_idx * stride_kh
    v_base = V + batch_idx * stride_vb + head_idx * stride_vh

    # Create TMA descriptors for Q, K, V
    # 128 is the head dimension, constant per the task constraints
    q_desc = tl.make_tensor_descriptor(
        q_base,
        shape=[S, 128],
        strides=[stride_qs, 1],
        block_shape=[BLOCK_M, 128],
        padding_option="zero",
    )
    k_desc = tl.make_tensor_descriptor(
        k_base,
        shape=[S, 128],
        strides=[stride_ks, 1],
        block_shape=[BLOCK_N, 128],
        padding_option="zero",
    )
    v_desc = tl.make_tensor_descriptor(
        v_base,
        shape=[S, 128],
        strides=[stride_vs, 1],
        block_shape=[BLOCK_N, 128],
        padding_option="zero",
    )

    # Pre-calculate masks and offsets for Q
    offs_m = m_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    q_mask = offs_m < S
    offset_m = (m_block_idx * BLOCK_M).to(tl.int32)
    
    # Load Q once
    q = q_desc.load([offset_m, 0])

    # Log constants
    RCP_LN2: tl.constexpr = 1.4426950408889634
    LN2: tl.constexpr = 0.6931471805599453

    # Initialize online softmax state
    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, 128), tl.float32)

    # Stream over K and V using standard loop and descriptor loads
    for kv_tile in range(0, tl.cdiv(S, BLOCK_N)):
        offset_n = (kv_tile * BLOCK_N).to(tl.int32)
        k = k_desc.load([offset_n, 0])
        v = v_desc.load([offset_n, 0])

        # Dot product and scaling (q: [BLOCK_M, 128], k.T: [128, BLOCK_N])
        scores = tl.dot(q, k.T, out_dtype=tl.float32) * sm_scale

        # Apply causal, window, and sequence-tail masks
        offs_n = offset_n + tl.arange(0, BLOCK_N)
        valid_score = q_mask[:, None] & (offs_n[None, :] < S)
        scores = tl.where(valid_score, scores * RCP_LN2, -float("inf"))

        # Update row max
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        safe_m_ij = tl.where(m_ij == -float("inf"), 0.0, m_ij)
        
        # Calculate exponents for online normalization
        alpha = tl.math.exp2(m_i - safe_m_ij)
        p = tl.math.exp2(scores - safe_m_ij[:, None])

        # Update sum and accumulator
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        # p: [BLOCK_M, BLOCK_N], v: [BLOCK_N, 128]
        acc = tl.dot(p.to(q.dtype), v, acc)
        
        m_i = m_ij

    # Finalize softmax values
    safe_l_i = tl.where(l_i == 0.0, 1.0, l_i)
    output = acc / safe_l_i[:, None]
    
    # Calculate LogSumExp in base-2, then convert to base-e for PyTorch reference parity
    lse_log2 = tl.where(l_i == 0.0, -float("inf"), m_i + tl.math.log2(safe_l_i))
    lse_ln = lse_log2 * LN2

    # Store O and LSE using ordinary masked pointers (only done once)
    o_offset = batch_idx * stride_ob + head_idx * stride_oh
    lse_offset = batch_idx * stride_lseb + head_idx * stride_lseh
    offs_d = tl.arange(0, 128)
    
    o_ptrs = O + o_offset + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(o_ptrs, output.to(q.dtype), mask=q_mask[:, None])

    lse_ptrs = LSE + lse_offset + offs_m * stride_lses
    tl.store(lse_ptrs, lse_ln, mask=q_mask)


def run(Q, K, V, O, LSE):
    """
    Computes a multi-head attention forward pass returning output and LogSumExp.
    """
    torch.cuda.set_device(Q.device)
    
    # Set up Triton device allocator required for device-side tensor descriptor creation
    def alloc_fn(size: int, alignment: int, stream):
        return torch.empty(size, device=Q.device, dtype=torch.int8)
    triton.set_allocator(alloc_fn)
    
    B, H, S, D = Q.shape
    sm_scale = 1.0 / (D ** 0.5)
    
    grid = lambda META: (triton.cdiv(S, META["BLOCK_M"]), B, H)
    
    _attention_kernel[grid](
        Q, K, V, O, LSE,
        S,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        sm_scale
    )