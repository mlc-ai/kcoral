import torch
import triton
import triton.language as tl

# Cache descriptor allocations to entirely remove host/device synchronization overhead during hot-path execution
_alloc_cache = {}
def alloc_fn(size: int, alignment: int, stream):
    if size not in _alloc_cache:
        _alloc_cache[size] = torch.empty(size, device="cuda", dtype=torch.int8)
    return _alloc_cache[size]

triton.set_allocator(alloc_fn)

def get_configs():
    return [
        # Large M blocks drastically minimize KV load traffic and maximize L2 sharing across operations.
        # B200 SMEM limit (228KB) accommodates these larger tiles securely with num_stages=2
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64}, num_warps=8, num_stages=4),
        
        # Standard configs
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=4),
        
        # Low footprint config
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=3),
    ]


@triton.autotune(configs=get_configs(), key=["S"])
@triton.jit
def _attn_fwd_tma_kernel(
    Q, K, V, O, LSE,
    S,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    offs_m_start = pid_m * BLOCK_M

    q_base = Q + pid_b * stride_qb + pid_h * stride_qh
    k_base = K + pid_b * stride_kb + pid_h * stride_kh
    v_base = V + pid_b * stride_vb + pid_h * stride_vh

    HEAD_DIM: tl.constexpr = 128

    # 2D TMA descriptors dynamically map to the appropriate head while guaranteeing native tcgen05 hardware pathways
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

    # Folded scale factor minimizing FP32 loop multiplications: 1 / sqrt(128) * log2(e)
    SCALE_LN2: tl.constexpr = 0.08838834764831843 * 1.4426950408889634
    q = (q * SCALE_LN2).to(tl.bfloat16)

    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)

    n_full_blocks = S // BLOCK_N

    for block_idx in range(0, n_full_blocks):
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

    # Masked block boundaries when the sequence does not align perfectly to a block multiple
    if S % BLOCK_N != 0:
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

    # Unnecessary sequence tail bound `tl.where` omitted natively utilizing non-causal attention rules
    output = acc / l_i[:, None]
    LN2: tl.constexpr = 0.6931471805599453
    lse = (m_i + tl.math.log2(l_i)) * LN2

    offs_m = offs_m_start + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    
    o_ptrs = O + pid_b * stride_ob + pid_h * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    lse_ptrs = LSE + pid_b * stride_lseb + pid_h * stride_lseh + offs_m * stride_lses
    
    if S % BLOCK_M == 0:
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
    S,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    offs_m_start = pid_m * BLOCK_M
    offs_m = offs_m_start + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    
    HEAD_DIM: tl.constexpr = 128
    offs_d = tl.arange(0, HEAD_DIM)

    q_ptrs = Q + pid_b * stride_qb + pid_h * stride_qh + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    k_ptrs = K + pid_b * stride_kb + pid_h * stride_kh + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = V + pid_b * stride_vb + pid_h * stride_vh + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd

    if S % BLOCK_M == 0:
        q = tl.load(q_ptrs)
    else:
        q = tl.load(q_ptrs, mask=offs_m[:, None] < S, other=0.0)

    SCALE_LN2: tl.constexpr = 0.08838834764831843 * 1.4426950408889634
    q = (q * SCALE_LN2).to(tl.bfloat16)

    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)

    n_full_blocks = S // BLOCK_N

    for block_idx in range(0, n_full_blocks):
        start_n = block_idx * BLOCK_N
        k = tl.load(k_ptrs + start_n * stride_ks)
        v = tl.load(v_ptrs + start_n * stride_vs)
        
        scores = tl.dot(q, k.T)
        
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.math.exp2(m_i - m_ij)
        p = tl.math.exp2(scores - m_ij[:, None])
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        m_i = m_ij

    if S % BLOCK_N != 0:
        start_n = n_full_blocks * BLOCK_N
        curr_n = start_n + offs_n
        mask_n = curr_n < S
        k = tl.load(k_ptrs + start_n * stride_ks, mask=mask_n[:, None], other=0.0)
        v = tl.load(v_ptrs + start_n * stride_vs, mask=mask_n[:, None], other=0.0)
        
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

    o_ptrs = O + pid_b * stride_ob + pid_h * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    lse_ptrs = LSE + pid_b * stride_lseb + pid_h * stride_lseh + offs_m * stride_lses
    
    if S % BLOCK_M == 0:
        tl.store(o_ptrs, output.to(tl.bfloat16))
        tl.store(lse_ptrs, lse)
    else:
        mask_m = offs_m < S
        tl.store(o_ptrs, output.to(tl.bfloat16), mask=mask_m[:, None])
        tl.store(lse_ptrs, lse, mask=mask_m)


def run(Q, K, V, O, LSE):
    """
    Computes non-causal multi-head attention forward using 2D block arrays guaranteeing native 
    tcgen05 mappings via TMA block alignments and tensor core footprints matching identically.
    """
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    grid = lambda META: (triton.cdiv(S, META["BLOCK_M"]), B, H)
    
    # Assert physical alignment/stride constraints to determine TMA descriptor memory alignment requirements
    tma_supported = True
    for tensor in (Q, K, V):
        if tensor.stride(-1) != 1 or (tensor.stride(-2) * 2) % 16 != 0:
            tma_supported = False
            break

    if tma_supported:
        _attn_fwd_tma_kernel[grid](
            Q, K, V, O, LSE,
            S,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            LSE.stride(0), LSE.stride(1), LSE.stride(2),
        )
    else:
        _attn_fwd_ptr_kernel[grid](
            Q, K, V, O, LSE,
            S,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            LSE.stride(0), LSE.stride(1), LSE.stride(2),
        )