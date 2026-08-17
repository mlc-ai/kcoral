import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

# Configure Triton's allocator for device-side tensor descriptors (for potential fallbacks)
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

try:
    triton.set_allocator(alloc_fn)
except Exception:
    pass


@triton.jit
def _fwd_kernel_2d_host_tma(
    q_desc, k_desc, v_desc, o_desc, LSE,
    stride_lseb, stride_lseh, stride_lses,
    B, H, S, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr
):
    b = tl.program_id(1)
    h = tl.program_id(2)
    m_block = tl.program_id(0)
    
    offset_m = m_block * BLOCK_M
    # Contiguous 2D mapping exactly matches physical flat storage layout
    base_idx = b * H * S + h * S
    
    # Hardware bounds-checked TMA load
    q = q_desc.load([base_idx + offset_m, 0])
    
    # Statically fuse Softmax scaling and base-2 exponential scaling 
    q = (q * scale).to(tl.bfloat16)
    
    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, D), tl.float32)
    
    num_n_blocks_full = S // BLOCK_N
    
    # --- HOT LOOP --- 
    # Processes complete blocks ensuring TMA loads, tensor cores, and exponentials pipeline perfectly
    for n_block in range(num_n_blocks_full):
        offset_n = n_block * BLOCK_N
        
        k = k_desc.load([base_idx + offset_n, 0])
        v = v_desc.load([base_idx + offset_n, 0])
        
        scores = tl.dot(q, k.T)
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.math.exp2(m_i - m_ij)
        p = tl.math.exp2(scores - m_ij[:, None])
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        m_i = m_ij

    # --- PARTIAL TAIL BOUNDARY (if applicable) ---
    if (S % BLOCK_N) != 0:
        offset_n = num_n_blocks_full * BLOCK_N
        
        k = k_desc.load([base_idx + offset_n, 0])
        v = v_desc.load([base_idx + offset_n, 0])
        
        scores = tl.dot(q, k.T)
        
        # Apply standard padding-mask boundary logic
        k_offsets = offset_n + tl.arange(0, BLOCK_N)
        mask = k_offsets[None, :] < S
        scores = tl.where(mask, scores, -float("inf"))
        
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.math.exp2(m_i - m_ij)
        p = tl.math.exp2(scores - m_ij[:, None])
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        m_i = m_ij

    safe_l_i = tl.where(l_i == 0.0, 1.0, l_i)
    output = acc / safe_l_i[:, None]
    
    # Store Output Tile using TMA
    o_desc.store([base_idx + offset_m, 0], output.to(tl.bfloat16))
    
    # Return LSE mathematically correctly transforming to standard PyTorch Natural Log convention 
    LN2: tl.constexpr = 0.6931471805599453
    lse = (m_i + tl.math.log2(safe_l_i)) * LN2
    
    m_offsets = offset_m + tl.arange(0, BLOCK_M)
    lse_mask = m_offsets < S
    lse_base = LSE + b * stride_lseb + h * stride_lseh
    tl.store(lse_base + m_offsets * stride_lses, lse, mask=lse_mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=4),
    ],
    key=["S"],
)
@triton.jit
def _fwd_kernel_pointers_fallback(
    Q, K, V, O, LSE,
    stride_qb, stride_qh, stride_qs,
    stride_kb, stride_kh, stride_ks,
    stride_vb, stride_vh, stride_vs,
    stride_ob, stride_oh, stride_os,
    stride_lseb, stride_lseh, stride_lses,
    S, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr
):
    b = tl.program_id(1)
    h = tl.program_id(2)
    m_block = tl.program_id(0)
    
    offset_m = m_block * BLOCK_M
    
    q_base = Q + b * stride_qb + h * stride_qh
    k_base = K + b * stride_kb + h * stride_kh
    v_base = V + b * stride_vb + h * stride_vh
    o_base = O + b * stride_ob + h * stride_oh
    
    offs_m = offset_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    mask_m = offs_m < S
    
    q_ptrs = q_base + offs_m[:, None] * stride_qs + offs_d[None, :]
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    q = (q * scale).to(tl.bfloat16)
    
    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, D), tl.float32)
    
    num_n_blocks_full = S // BLOCK_N
    offs_n = tl.arange(0, BLOCK_N)
    k_ptrs = k_base + offs_n[:, None] * stride_ks + offs_d[None, :]
    v_ptrs = v_base + offs_n[:, None] * stride_vs + offs_d[None, :]

    for n_block in range(num_n_blocks_full):
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

    if (S % BLOCK_N) != 0:
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

    safe_l_i = tl.where(l_i == 0.0, 1.0, l_i)
    output = acc / safe_l_i[:, None]
    
    o_ptrs = o_base + offs_m[:, None] * stride_os + offs_d[None, :]
    tl.store(o_ptrs, output.to(tl.bfloat16), mask=mask_m[:, None])
    
    LN2: tl.constexpr = 0.6931471805599453
    lse = (m_i + tl.math.log2(safe_l_i)) * LN2
    
    lse_base = LSE + b * stride_lseb + h * stride_lseh
    tl.store(lse_base + offs_m * stride_lses, lse, mask=mask_m)


def run(Q, K, V, O, LSE):
    """Compute Non-Causal Multi-Head Attention forward with LSE explicitly writing to O & LSE."""
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    # Premultiplying static scales translating PyTorch SDPA `1 / sqrt(D)` mapping into base-2 exponential properties
    RCP_LN2 = 1.4426950408889634
    scale = (1.0 / (D ** 0.5)) * RCP_LN2
    
    # Hot-Path execution: For structurally contiguous memory footprints bypass per-CTA descriptor creation.
    # Yields the highly performant Hopper/Blackwell hardware natively compiled TMA instructions.
    if Q.is_contiguous() and K.is_contiguous() and V.is_contiguous() and O.is_contiguous():
        Q_2d = Q.view(-1, D)
        K_2d = K.view(-1, D)
        V_2d = V.view(-1, D)
        O_2d = O.view(-1, D)
        
        BLOCK_M = 128
        BLOCK_N = 64
        
        q_desc = TensorDescriptor.from_tensor(Q_2d, [BLOCK_M, D])
        k_desc = TensorDescriptor.from_tensor(K_2d, [BLOCK_N, D])
        v_desc = TensorDescriptor.from_tensor(V_2d, [BLOCK_N, D])
        o_desc = TensorDescriptor.from_tensor(O_2d, [BLOCK_M, D])
        
        # M block varying fastest yields essentially contiguous hardware L2 cache hits for streamed memory footprints.
        grid = (triton.cdiv(S, BLOCK_M), B, H)
        
        _fwd_kernel_2d_host_tma[grid](
            q_desc, k_desc, v_desc, o_desc, LSE,
            LSE.stride(0), LSE.stride(1), LSE.stride(2),
            B, H, S, scale,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, D=D,
            num_warps=8, num_stages=4
        )
    else:
        grid = lambda META: (triton.cdiv(S, META["BLOCK_M"]), B, H)
        _fwd_kernel_pointers_fallback[grid](
            Q, K, V, O, LSE,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            LSE.stride(0), LSE.stride(1), LSE.stride(2),
            S, scale,
            D=D
        )