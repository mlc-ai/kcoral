import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale,
    O, LSE,
    stride_qb, stride_qh, stride_qs,
    stride_kb, stride_kh, stride_ks,
    stride_vb, stride_vh, stride_vs,
    stride_ob, stride_oh, stride_os,
    stride_lb, stride_lh, stride_ls,
    B, H, S,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    # Extract batch and head indices
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    # 1. Device-side Descriptor Creation (TMA-backed)
    # The physical layout matches perfectly the required 16-byte leading stride alignment.
    q_base = Q + pid_b * stride_qb + pid_h * stride_qh
    q_desc = tl.make_tensor_descriptor(
        q_base,
        shape=[S, D],
        strides=[stride_qs, 1],
        block_shape=[BLOCK_M, D],
        padding_option="zero"
    )
    
    k_base = K + pid_b * stride_kb + pid_h * stride_kh
    k_desc = tl.make_tensor_descriptor(
        k_base,
        shape=[S, D],
        strides=[stride_ks, 1],
        block_shape=[BLOCK_N, D],
        padding_option="zero"
    )
    
    v_base = V + pid_b * stride_vb + pid_h * stride_vh
    v_desc = tl.make_tensor_descriptor(
        v_base,
        shape=[S, D],
        strides=[stride_vs, 1],
        block_shape=[BLOCK_N, D],
        padding_option="zero"
    )
    
    o_base = O + pid_b * stride_ob + pid_h * stride_oh
    o_desc = tl.make_tensor_descriptor(
        o_base,
        shape=[S, D],
        strides=[stride_os, 1],
        block_shape=[BLOCK_M, D],
        padding_option="zero"
    )

    RCP_LN2: tl.constexpr = 1.4426950408889634
    LN2: tl.constexpr = 0.6931471805599453

    # Load Q block
    q = q_desc.load([pid_m * BLOCK_M, 0])

    # Initialize softmax state natively for base-2 calculations
    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, D), tl.float32)

    # Masking setup
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    q_valid = offs_m < S
    
    # Causal sequence length for this block
    max_kv_len = tl.minimum(S, (pid_m + 1) * BLOCK_M)
    num_kv_tiles = tl.cdiv(max_kv_len, BLOCK_N)

    # Precompute combined scale
    scale_log2 = sm_scale * RCP_LN2

    # Inner online-softmax loop
    for kv_tile in range(0, num_kv_tiles):
        offset_n = kv_tile * BLOCK_N
        k = k_desc.load([offset_n, 0])
        v = v_desc.load([offset_n, 0])

        acc_scores = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        scores = tl.dot(q, k.T, acc_scores)
        
        # Apply scaling in FP32 to avoid catastrophic precision loss
        scores = scores * scale_log2

        # Causal mask - offs_n boundary handling is naturally absorbed by this
        offs_n = offset_n + tl.arange(0, BLOCK_N)
        valid_score = (offs_m[:, None] >= offs_n[None, :]) & q_valid[:, None]
        scores = tl.where(valid_score, scores, -float("inf"))

        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        safe_m_ij = tl.where(m_ij == -float("inf"), 0.0, m_ij)
        
        alpha = tl.math.exp2(m_i - safe_m_ij)
        p = tl.math.exp2(scores - safe_m_ij[:, None])

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        m_i = m_ij

    # Epilogue & normalizations
    safe_l_i = tl.where(l_i == 0.0, 1.0, l_i)
    out = acc / safe_l_i[:, None]

    # Convert base-2 LSE back to standard natural-log LSE scale
    lse_log2 = tl.where(l_i == 0.0, -float("inf"), m_i + tl.math.log2(safe_l_i))
    lse = lse_log2 * LN2

    # Write output to descriptors; out of bounds writes are silently ignored by TMA
    o_desc.store([pid_m * BLOCK_M, 0], out.to(tl.bfloat16))

    # LSE is a simple 1D structure over sequence length - fallback to safe masked pointer store
    lse_base = LSE + pid_b * stride_lb + pid_h * stride_lh
    lse_ptrs = lse_base + offs_m * stride_ls
    tl.store(lse_ptrs, lse, mask=q_valid)


def run(Q, K, V, O, LSE):
    """Compute Causal Multi-Head Attention forward returning O and LSE."""
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    sm_scale = 1.0 / (D ** 0.5)

    # Blackwell dictates defining an allocator for device-side created tensor descriptors
    def alloc_fn(size: int, alignment: int, stream):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    BLOCK_M = 128
    BLOCK_N = 64
    
    # Tuning launch configurations specifically mapped to promote concurrent evaluation 
    # of blocks under the same batch and head logic to naturally utilize L2 cache reuse.
    grid = (triton.cdiv(S, BLOCK_M), B * H)

    _fwd_kernel[grid](
        Q, K, V, sm_scale,
        O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        B, H, S,
        D=D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=8,
        num_stages=3
    )