import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'LOOP_STAGES': 2}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'LOOP_STAGES': 3}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'LOOP_STAGES': 3}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'LOOP_STAGES': 4}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'LOOP_STAGES': 3}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'LOOP_STAGES': 3}, num_warps=4, num_stages=3),
    ],
    key=['S'],
)
@triton.jit
def _fwd_kernel(
    Q, K, V, scale_log2,
    O, LSE,
    stride_qb, stride_qh, stride_qs,
    stride_kb, stride_kh, stride_ks,
    stride_vb, stride_vh, stride_vs,
    stride_ob, stride_oh, stride_os,
    stride_lb, stride_lh, stride_ls,
    B, H, S,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    LOOP_STAGES: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    # Extract batch and head indices
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    # Device-side Descriptor Creation (TMA-backed via descriptors implicitly pipelined)
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

    LN2: tl.constexpr = 0.6931471805599453

    # Load Q block once
    q = q_desc.load([pid_m * BLOCK_M, 0])

    # Initialize softmax state efficiently configured for base-2
    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, D), tl.float32)

    # Masking setup
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_m_2d = offs_m[:, None]
    q_valid = offs_m < S
    
    # Calculate causal execution bounds to explicitly eliminate masking for strictly safe tiles
    max_kv_len = tl.minimum(S, (pid_m + 1) * BLOCK_M)
    num_kv_tiles = tl.cdiv(max_kv_len, BLOCK_N)
    
    # "Full tiles" are guaranteed strictly causal; eliminates the sequence and causal bounding instructions.
    num_full_tiles = (pid_m * BLOCK_M) // BLOCK_N
    num_full_tiles = tl.minimum(num_full_tiles, num_kv_tiles)

    # Loop 1: Fully valid sequence blocks (No `tl.where` limits internally)
    for kv_tile in tl.range(0, num_full_tiles, num_stages=LOOP_STAGES):
        offset_n = kv_tile * BLOCK_N
        k = k_desc.load([offset_n, 0])
        v = v_desc.load([offset_n, 0])

        scores = tl.dot(q, k.T)
        scores = scores * scale_log2
        
        # Eliminates conditional safe_m_ij since scores natively carry finite values in the safe bounds
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        
        alpha = tl.math.exp2(m_i - m_ij)
        p = tl.math.exp2(scores - m_ij[:, None])

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        m_i = m_ij

    # Loop 2: Partial/Boundary sequence blocks (Causal & sequence bounds safely verified)
    for kv_tile in tl.range(num_full_tiles, num_kv_tiles, num_stages=LOOP_STAGES):
        offset_n = kv_tile * BLOCK_N
        k = k_desc.load([offset_n, 0])
        v = v_desc.load([offset_n, 0])

        scores = tl.dot(q, k.T)
        scores = scores * scale_log2

        # Causal bounds implicitly protects out-of-bounds KV logic via padding properties
        offs_n_row = (offset_n + tl.arange(0, BLOCK_N))[None, :]
        valid_score = offs_m_2d >= offs_n_row
        scores = tl.where(valid_score, scores, -float("inf"))

        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        safe_m_ij = tl.where(m_ij == -float("inf"), 0.0, m_ij)
        
        alpha = tl.math.exp2(m_i - safe_m_ij)
        p = tl.math.exp2(scores - safe_m_ij[:, None])

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        m_i = m_ij

    # Epilogue & Division-free vector normalization (inverse multiplication mapping)
    safe_l_i = tl.where(l_i == 0.0, 1.0, l_i)
    inv_l_i = 1.0 / safe_l_i
    out = acc * inv_l_i[:, None]

    # Map LSE outputs back to natural-log scaling to correctly link mathematically
    lse_log2 = tl.where(l_i == 0.0, -float("inf"), m_i + tl.math.log2(safe_l_i))
    lse = lse_log2 * LN2

    # Descriptor native out-of-bounds omission explicitly honors `q_valid`
    o_desc.store([pid_m * BLOCK_M, 0], out.to(tl.bfloat16))

    # Masked physical pointer dump targeting sequential batch lengths
    lse_base = LSE + pid_b * stride_lb + pid_h * stride_lh
    lse_ptrs = lse_base + offs_m * stride_ls
    tl.store(lse_ptrs, lse, mask=q_valid)


def run(Q, K, V, O, LSE):
    """Compute Causal Multi-Head Attention forward returning O and LSE natively using standard Triton."""
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    # Hoisted computational float scale mapping pre-applied for instruction savings
    scale_log2 = (1.0 / (D ** 0.5)) * 1.4426950408889634

    # Target dictates an allocator specification for TMEM mappings globally
    def alloc_fn(size: int, alignment: int, stream):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)
    
    # Matrix partitioning logically promotes independent processing streams directly
    # mapped by L2 locality patterns spanning across (B*H) grids
    grid = lambda META: (triton.cdiv(S, META['BLOCK_M']), B * H)

    _fwd_kernel[grid](
        Q, K, V, scale_log2,
        O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        B, H, S,
        D=D
    )