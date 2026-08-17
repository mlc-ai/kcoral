import torch
import triton
import triton.language as tl

# Pre-allocate device descriptors to enforce entirely zero-overhead TMA configurations on the hot path
_alloc_cache = {}
def alloc_fn(size: int, alignment: int, stream):
    if size not in _alloc_cache:
        _alloc_cache[size] = torch.empty(size, device="cuda", dtype=torch.int8)
    return _alloc_cache[size]

triton.set_allocator(alloc_fn)

def get_configs():
    return [
        # Aggressive pipeline configurations for optimal Tensor Core overlap on SM100 limits
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=5),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=4),
        
        # Larger footprint configurations
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4, num_stages=3),
        
        # Reduced resource footprints
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=4),
    ]


@triton.autotune(configs=get_configs(), key=["S"])
@triton.jit
def _attn_fwd_tma_kernel(
    Q, K, V, O, LSE,
    S, H,
    stride_qb, stride_qh, stride_qs,
    stride_kb, stride_kh, stride_ks,
    stride_vb, stride_vh, stride_vs,
    stride_ob, stride_oh, stride_os,
    stride_lseb, stride_lseh, stride_lses,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    offs_m_start = pid_m * BLOCK_M

    q_base = Q + pid_b * stride_qb + pid_h * stride_qh
    k_base = K + pid_b * stride_kb + pid_h * stride_kh
    v_base = V + pid_b * stride_vb + pid_h * stride_vh

    HEAD_DIM: tl.constexpr = 128

    # Zero-overhead 2D device side descriptors enforcing optimal native `tcgen05` mapping instructions
    q_desc = tl.make_tensor_descriptor(
        q_base, shape=[S, HEAD_DIM], strides=[stride_qs, 1], block_shape=[BLOCK_M, HEAD_DIM], padding_option="zero"
    )
    k_desc = tl.make_tensor_descriptor(
        k_base, shape=[S, HEAD_DIM], strides=[stride_ks, 1], block_shape=[BLOCK_N, HEAD_DIM], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        v_base, shape=[S, HEAD_DIM], strides=[stride_vs, 1], block_shape=[BLOCK_N, HEAD_DIM], padding_option="zero"
    )

    q = q_desc.load([offs_m_start, 0])

    # Folded FP32 pre-scale mapping minimizing scaling arithmetic natively using MUFU exponential logic rules 
    SCALE_LN2: tl.constexpr = 0.08838834764831843 * 1.4426950408889634
    q = (q * SCALE_LN2).to(tl.bfloat16)

    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)

    n_full_blocks = S // BLOCK_N

    for block_idx in range(n_full_blocks):
        start_n = block_idx * BLOCK_N
        k = k_desc.load([start_n, 0])
        v = v_desc.load([start_n, 0])
        
        scores = tl.dot(q, k.T)
        
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.math.exp2(m_i - m_ij)
        p = tl.math.exp2(scores - m_ij[:, None])
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        m_i = m_ij

    if not EVEN_N:
        start_n = n_full_blocks * BLOCK_N
        k = k_desc.load([start_n, 0])
        v = v_desc.load([start_n, 0])
        
        scores = tl.dot(q, k.T)
        
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S
        scores = tl.where(mask_n[None, :], scores, -float("inf"))
        
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.math.exp2(m_i - m_ij)
        p = tl.math.exp2(scores - m_ij[:, None])
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        m_i = m_ij

    # Safe row reduction natively skipping `tl.where(l_i == 0.0)` based on sequence fullness limits
    output = acc / l_i[:, None]
    LN2: tl.constexpr = 0.6931471805599453
    lse = (m_i + tl.math.log2(l_i)) * LN2

    offs_m = offs_m_start + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    
    # Write boundaries
    o_ptrs = O + pid_b * stride_ob + pid_h * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :]
    lse_ptrs = LSE + pid_b * stride_lseb + pid_h * stride_lseh + offs_m * stride_lses
    
    if EVEN_M:
        tl.store(o_ptrs, output.to(tl.bfloat16))
        tl.store(lse_ptrs, lse)
    else:
        mask_m = offs_m < S
        tl.store(o_ptrs, output.to(tl.bfloat16), mask=mask_m[:, None])
        tl.store(lse_ptrs, lse, mask=mask_m)


@triton.autotune(configs=get_configs(), key=["S"])
@triton.jit
def _attn_fwd_ptr_kernel(
    Q, K, V, O, LSE,
    S, H,
    stride_qb, stride_qh, stride_qs,
    stride_kb, stride_kh, stride_ks,
    stride_vb, stride_vh, stride_vs,
    stride_ob, stride_oh, stride_os,
    stride_lseb, stride_lseh, stride_lses,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    offs_m_start = pid_m * BLOCK_M
    offs_m = offs_m_start + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    
    HEAD_DIM: tl.constexpr = 128
    offs_d = tl.arange(0, HEAD_DIM)

    q_base = Q + pid_b * stride_qb + pid_h * stride_qh
    k_base = K + pid_b * stride_kb + pid_h * stride_kh
    v_base = V + pid_b * stride_vb + pid_h * stride_vh

    # Optimize inner-loop induction variables manually mapping `k_ptrs` + offset
    q_ptrs = q_base + offs_m[:, None] * stride_qs + offs_d[None, :]
    k_ptrs = k_base + offs_n[:, None] * stride_ks + offs_d[None, :]
    v_ptrs = v_base + offs_n[:, None] * stride_vs + offs_d[None, :]

    if EVEN_M:
        q = tl.load(q_ptrs)
    else:
        q = tl.load(q_ptrs, mask=offs_m[:, None] < S, other=0.0)

    SCALE_LN2: tl.constexpr = 0.08838834764831843 * 1.4426950408889634
    q = (q * SCALE_LN2).to(tl.bfloat16)

    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)

    n_full_blocks = S // BLOCK_N

    for block_idx in range(n_full_blocks):
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)
        
        scores = tl.dot(q, k.T)
        
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.math.exp2(m_i - m_ij)
        p = tl.math.exp2(scores - m_ij[:, None])
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        m_i = m_ij
        
        k_ptrs += BLOCK_N * stride_ks
        v_ptrs += BLOCK_N * stride_vs

    if not EVEN_N:
        start_n = n_full_blocks * BLOCK_N
        curr_n = start_n + offs_n
        mask_n = curr_n < S
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        
        scores = tl.dot(q, k.T)
        scores = tl.where(mask_n[None, :], scores, -float("inf"))
        
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.math.exp2(m_i - m_ij)
        p = tl.math.exp2(scores - m_ij[:, None])
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        m_i = m_ij

    output = acc / l_i[:, None]
    LN2: tl.constexpr = 0.6931471805599453
    lse = (m_i + tl.math.log2(l_i)) * LN2

    o_ptrs = O + pid_b * stride_ob + pid_h * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :]
    lse_ptrs = LSE + pid_b * stride_lseb + pid_h * stride_lseh + offs_m * stride_lses
    
    if EVEN_M:
        tl.store(o_ptrs, output.to(tl.bfloat16))
        tl.store(lse_ptrs, lse)
    else:
        mask_m = offs_m < S
        tl.store(o_ptrs, output.to(tl.bfloat16), mask=mask_m[:, None])
        tl.store(lse_ptrs, lse, mask=mask_m)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    grid = lambda META: (triton.cdiv(S, META["BLOCK_M"]), B * H)
    
    is_even = (S % 128 == 0)

    # Validating standard Native TMA alignment descriptors mathematically mappings.
    tma_supported = True
    for tensor in (Q, K, V):
        if tensor.stride(-1) != 1 or (tensor.stride(-2) * 2) % 16 != 0:
            tma_supported = False
            break

    if tma_supported:
        _attn_fwd_tma_kernel[grid](
            Q, K, V, O, LSE,
            S, H,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            LSE.stride(0), LSE.stride(1), LSE.stride(2),
            EVEN_M=is_even,
            EVEN_N=is_even,
        )
    else:
        _attn_fwd_ptr_kernel[grid](
            Q, K, V, O, LSE,
            S, H,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            LSE.stride(0), LSE.stride(1), LSE.stride(2),
            EVEN_M=is_even,
            EVEN_N=is_even,
        )