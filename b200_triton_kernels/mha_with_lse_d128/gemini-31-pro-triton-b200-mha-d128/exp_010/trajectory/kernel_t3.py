import torch
import triton
import triton.language as tl

def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

@triton.autotune(
    configs=[
        # High occupancy configs (aiming for 2 CTAs per SM)
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        
        # Max tile configs (1 CTA per SM)
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
    ],
    key=['S'],
)
@triton.jit
def _fwd_kernel(
    Q, K, V, O, LSE,
    stride_qb, stride_qh, stride_qs,
    stride_kb, stride_kh, stride_ks,
    stride_vb, stride_vh, stride_vs,
    stride_ob, stride_oh, stride_os,
    stride_lseb, stride_lseh, stride_lses,
    S, H, scale_log2,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # Mapping pid_m to axis 0 schedules query blocks for the same batch/head concurrently, 
    # massively improving L2 cache hit rates for K and V tiles.
    pid_m = tl.program_id(0).to(tl.int64)
    pid_bh = tl.program_id(1).to(tl.int64)

    pid_b = pid_bh // H
    pid_h = pid_bh % H

    # Base pointers for the current batch and head
    q_base = Q + pid_b * stride_qb + pid_h * stride_qh
    k_base = K + pid_b * stride_kb + pid_h * stride_kh
    v_base = V + pid_b * stride_vb + pid_h * stride_vh
    o_base = O + pid_b * stride_ob + pid_h * stride_oh

    # Create Tensor descriptors inside the kernel but avoiding dynamic per-loop allocation 
    q_desc = tl.make_tensor_descriptor(
        q_base,
        shape=[S, BLOCK_D],
        strides=[stride_qs, 1],
        block_shape=[BLOCK_M, BLOCK_D],
        padding_option="zero"
    )
    k_desc = tl.make_tensor_descriptor(
        k_base,
        shape=[S, BLOCK_D],
        strides=[stride_ks, 1],
        block_shape=[BLOCK_N, BLOCK_D],
        padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        v_base,
        shape=[S, BLOCK_D],
        strides=[stride_vs, 1],
        block_shape=[BLOCK_N, BLOCK_D],
        padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        o_base,
        shape=[S, BLOCK_D],
        strides=[stride_os, 1],
        block_shape=[BLOCK_M, BLOCK_D],
        padding_option="zero"
    )

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    valid_m = offs_m < S

    # Load Q block via TMA
    offset_m = (pid_m * BLOCK_M).to(tl.int32)
    q = q_desc.load([offset_m, 0])

    # Initialize online softmax state
    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    offs_n = tl.arange(0, BLOCK_N)
    k_tiles = tl.cdiv(S, BLOCK_N)

    # Standard range loop to automatically utilize kernel launch `num_stages` software pipelining
    for kv_tile in range(0, k_tiles):
        offset_n = (kv_tile * BLOCK_N).to(tl.int32)
        
        # TMA Loads
        k = k_desc.load([offset_n, 0])
        v = v_desc.load([offset_n, 0])

        # Q @ K.T (scaled by base-2 scale)
        scores = tl.dot(q, k.T, out_dtype=tl.float32) * scale_log2
        
        # Masking sequence tails
        curr_offs_n = offset_n + offs_n
        valid_score = valid_m[:, None] & (curr_offs_n[None, :] < S)
        scores = tl.where(valid_score, scores, -float("inf"))

        # Online Softmax update
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        safe_m_ij = tl.where(m_ij == -float("inf"), 0.0, m_ij)
        alpha = tl.math.exp2(m_i - safe_m_ij)
        p = tl.math.exp2(scores - safe_m_ij[:, None])

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        
        # P @ V
        acc = tl.dot(p.to(tl.bfloat16), v, acc, out_dtype=tl.float32)
        
        m_i = m_ij

    # Finalize softmax and compute natural-log LSE
    safe_l_i = tl.where(l_i == 0.0, 1.0, l_i)
    out = acc / safe_l_i[:, None]
    lse_log2 = tl.where(l_i == 0.0, -float("inf"), m_i + tl.math.log2(safe_l_i))
    
    LN2 = 0.6931471805599453
    lse_ln = lse_log2 * LN2

    # Store output O seamlessly bounded via TMA out-of-bounds dropping
    o_desc.store([offset_m, 0], out.to(tl.bfloat16))

    # Store output LSE explicitly masked for the 1D dimension
    lse_ptrs = LSE + pid_b * stride_lseb + pid_h * stride_lseh + offs_m * stride_lses
    tl.store(lse_ptrs, lse_ln, mask=valid_m)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    
    # Configure the Triton descriptor allocator needed for device-created descriptors.
    triton.set_allocator(alloc_fn)
    
    B, H, S, D = Q.shape
    
    # Precompute standard scale, converting to base-2 mathematically 
    # to hit Triton's optimized `exp2` routine inline
    sm_scale = 1.0 / (D ** 0.5)
    RCP_LN2 = 1.4426950408889634
    scale_log2 = sm_scale * RCP_LN2
    
    # Grouping pid_m on the inner dimension to maximize L2 hit rates for K and V
    grid = lambda META: (
        triton.cdiv(S, META['BLOCK_M']),
        B * H,
    )
    
    _fwd_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        S, H, scale_log2,
        BLOCK_D=D,
    )