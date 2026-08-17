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
        # Descriptor (TMA) configs
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "USE_TMA": True}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "USE_TMA": True}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "USE_TMA": True}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "USE_TMA": True}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "USE_TMA": True}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "USE_TMA": True}, num_warps=8, num_stages=2),
        
        # Pointer configs
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "USE_TMA": False}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "USE_TMA": False}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "USE_TMA": False}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "USE_TMA": False}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "USE_TMA": False}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "USE_TMA": False}, num_warps=8, num_stages=2),
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
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr, USE_TMA: tl.constexpr
):
    b = tl.program_id(1)
    h = tl.program_id(2)
    m_block = tl.program_id(0)
    
    offset_m = m_block * BLOCK_M

    # Pre-advance pointers to specific batch and head
    q_base = Q + b * stride_qb + h * stride_qh
    k_base = K + b * stride_kb + h * stride_kh
    v_base = V + b * stride_vb + h * stride_vh
    o_base = O + b * stride_ob + h * stride_oh
    
    offs_m = offset_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    mask_m = offs_m < S
    
    # Load Query block
    if USE_TMA:
        q_desc = tl.make_tensor_descriptor(q_base, shape=[S, D], strides=[stride_qs, 1], block_shape=[BLOCK_M, D], padding_option="zero")
        k_desc = tl.make_tensor_descriptor(k_base, shape=[S, D], strides=[stride_ks, 1], block_shape=[BLOCK_N, D], padding_option="zero")
        v_desc = tl.make_tensor_descriptor(v_base, shape=[S, D], strides=[stride_vs, 1], block_shape=[BLOCK_N, D], padding_option="zero")
        o_desc = tl.make_tensor_descriptor(o_base, shape=[S, D], strides=[stride_os, 1], block_shape=[BLOCK_M, D], padding_option="zero")
        q = q_desc.load([offset_m, 0])
    else:
        q_ptrs = q_base + offs_m[:, None] * stride_qs + offs_d[None, :]
        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
        
    # Apply fused Softmax scale and natural-to-base-2 scaling statically
    q = (q * scale).to(tl.bfloat16)
    
    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, D), tl.float32)
    
    num_n_blocks_full = S // BLOCK_N
    
    if not USE_TMA:
        offs_n = tl.arange(0, BLOCK_N)
        k_ptrs = k_base + offs_n[:, None] * stride_ks + offs_d[None, :]
        v_ptrs = v_base + offs_n[:, None] * stride_vs + offs_d[None, :]

    # HOT LOOP (Fully aligned sizes avoiding conditional tail checks inner computation)
    for n_block in range(num_n_blocks_full):
        if USE_TMA:
            offset_n = n_block * BLOCK_N
            k = k_desc.load([offset_n, 0])
            v = v_desc.load([offset_n, 0])
        else:
            k = tl.load(k_ptrs)
            v = tl.load(v_ptrs)
            k_ptrs += BLOCK_N * stride_ks
            v_ptrs += BLOCK_N * stride_vs
        
        scores = tl.dot(q, k.T)
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.math.exp2(m_i - m_ij)
        p = tl.math.exp2(scores - m_ij[:, None])
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        m_i = m_ij

    # EPILOGUE PARTIAL BLOCK
    if (S % BLOCK_N) != 0:
        if USE_TMA:
            offset_n = num_n_blocks_full * BLOCK_N
            k = k_desc.load([offset_n, 0])
            v = v_desc.load([offset_n, 0])
            k_offsets = offset_n + tl.arange(0, BLOCK_N)
            mask_n = k_offsets < S
        else:
            mask_n = (num_n_blocks_full * BLOCK_N + offs_n) < S
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

    # Guarantee validity to avoid NaN propagation from masked rows
    safe_l_i = tl.where(l_i == 0.0, 1.0, l_i)
    output = acc / safe_l_i[:, None]
    
    # Store output tile
    if USE_TMA:
        o_desc.store([offset_m, 0], output.to(tl.bfloat16))
    else:
        o_ptrs = o_base + offs_m[:, None] * stride_os + offs_d[None, :]
        tl.store(o_ptrs, output.to(tl.bfloat16), mask=mask_m[:, None])
    
    # Return LSE mathematically correctly translating back to standard Natural Log domain
    LN2: tl.constexpr = 0.6931471805599453
    lse = (m_i + tl.math.log2(safe_l_i)) * LN2
    
    lse_base = LSE + b * stride_lseb + h * stride_lseh
    tl.store(lse_base + offs_m * stride_lses, lse, mask=mask_m)


def run(Q, K, V, O, LSE):
    """Compute Non-Causal Multi-Head Attention forward with LSE."""
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    
    # Scale Q explicitly translating PyTorch SDPA `1 / sqrt(D)` into base-2 exponential properties
    RCP_LN2 = 1.4426950408889634
    scale = (1.0 / (D ** 0.5)) * RCP_LN2
    
    # High-Performance execution exploiting implicit L2 layout properties (M changes fastest)
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