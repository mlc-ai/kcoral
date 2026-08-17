import torch
import triton
import triton.language as tl

# Set up Triton allocator for device-side descriptor infrastructure
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=4),
    ],
    key=["S"],
)
@triton.jit
def _attn_fwd_kernel(
    Q, K, V, O, LSE,
    sm_scale_log2,
    S,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    H,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    batch_head = tl.program_id(1)
    
    batch_idx = batch_head // H
    head_idx = batch_head % H
    
    # Early exit if the sequence tile is fully out of bounds
    if start_m * BLOCK_M >= S:
        return

    # Compute base pointers for the specific batch and head
    q_base = Q + batch_idx * stride_qb + head_idx * stride_qh
    k_base = K + batch_idx * stride_kb + head_idx * stride_kh
    v_base = V + batch_idx * stride_vb + head_idx * stride_vh
    o_base = O + batch_idx * stride_ob + head_idx * stride_oh
    lse_base = LSE + batch_idx * stride_lseb + head_idx * stride_lseh
    
    # TMA 2D Descriptors for optimal hardware memory tracking
    q_desc = tl.make_tensor_descriptor(
        q_base, shape=[S, D], strides=[stride_qs, stride_qd],
        block_shape=[BLOCK_M, D], padding_option="zero"
    )
    k_desc = tl.make_tensor_descriptor(
        k_base, shape=[S, D], strides=[stride_ks, stride_kd],
        block_shape=[BLOCK_N, D], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        v_base, shape=[S, D], strides=[stride_vs, stride_vd],
        block_shape=[BLOCK_N, D], padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        o_base, shape=[S, D], strides=[stride_os, stride_od],
        block_shape=[BLOCK_M, D]
    )
    
    # Query load is hoisted and automatically routed to SMEM for standard WGMMA efficiency
    q = q_desc.load([start_m * BLOCK_M, 0])
    
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    q_mask = offs_m < S
    
    max_n = tl.minimum((start_m + 1) * BLOCK_M, S)
    num_full_blocks = tl.minimum(S // BLOCK_N, start_m * BLOCK_M // BLOCK_N)
    end_n_blocks = (max_n + BLOCK_N - 1) // BLOCK_N
    
    # Phase 1: Fully unmasked blocks (no boundary branches)
    for start_n in range(0, num_full_blocks):
        offset_n = start_n * BLOCK_N
        k = k_desc.load([offset_n, 0])
        v = v_desc.load([offset_n, 0])
        
        qk = tl.dot(q, k.T)
        qk = qk * sm_scale_log2
        
        # Base-2 scaled calculations to replace heavy `tl.exp` with `tl.exp2`
        m_i_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp2(m_i - m_i_new)
        p = tl.exp2(qk - m_i_new[:, None])
        
        l_i_new = alpha * l_i + tl.sum(p, axis=1)
        
        acc = acc * alpha[:, None]
        p_bf16 = p.to(tl.bfloat16)
        acc = tl.dot(p_bf16, v, acc=acc)
        
        m_i = m_i_new
        l_i = l_i_new

    # Phase 2: Causal blocks 
    for start_n in range(num_full_blocks, end_n_blocks):
        offset_n = start_n * BLOCK_N
        k = k_desc.load([offset_n, 0])
        v = v_desc.load([offset_n, 0])
        
        qk = tl.dot(q, k.T)
        qk = qk * sm_scale_log2
        
        offs_n = offset_n + tl.arange(0, BLOCK_N)
        k_mask = offs_n < S
        valid_mask = (offs_m[:, None] >= offs_n[None, :]) & k_mask[None, :]
        
        # Fallback true mask for completely out-of-bounds queries to avoid NaN calculations via `-inf - (-inf)`
        final_mask = valid_mask | (~q_mask[:, None])
        qk = tl.where(final_mask, qk, float("-inf"))
        
        m_i_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp2(m_i - m_i_new)
        p = tl.exp2(qk - m_i_new[:, None])
        
        l_i_new = alpha * l_i + tl.sum(p, axis=1)
        
        acc = acc * alpha[:, None]
        p_bf16 = p.to(tl.bfloat16)
        acc = tl.dot(p_bf16, v, acc=acc)
        
        m_i = m_i_new
        l_i = l_i_new

    acc = acc / l_i[:, None]
    
    # Store output via TMA descriptor, inherently ignoring elements row >= S safely
    o_desc.store([start_m * BLOCK_M, 0], acc.to(tl.bfloat16))
    
    # Convert base-2 LSE back to standard natural-log LSE
    ln2 = 0.6931471805599453
    lse = (m_i + tl.log2(l_i)) * ln2
    
    lse_ptrs = lse_base + offs_m * stride_lses
    tl.store(lse_ptrs, lse, mask=q_mask)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    
    # Precompute scaling paired with exp2 transformation coefficient
    sm_scale = 1.0 / (D ** 0.5)
    log2_e = 1.4426950408889634
    sm_scale_log2 = sm_scale * log2_e
    
    # Standard grid implicitly handles L2 locality grouping by iterating M per batch/head sequence
    grid = lambda META: (triton.cdiv(S, META["BLOCK_M"]), B * H)
    
    _attn_fwd_kernel[grid](
        Q, K, V, O, LSE,
        sm_scale_log2,
        S,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        H,
        D=D,
    )