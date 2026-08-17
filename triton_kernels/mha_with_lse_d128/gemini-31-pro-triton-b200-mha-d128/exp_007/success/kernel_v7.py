import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=5),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64}, num_warps=8, num_stages=4),
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
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr,
    IS_CONTIGUOUS: tl.constexpr
):
    # Performance hint: informs compiler of dense inner memory layouts to optimize pointer vectorization
    if IS_CONTIGUOUS:
        tl.assume(stride_qs == D)
        tl.assume(stride_ks == D)
        tl.assume(stride_vs == D)
        tl.assume(stride_os == D)

    b = tl.program_id(1)
    h = tl.program_id(2)
    m_block = tl.program_id(0)
    
    offset_m = m_block * BLOCK_M
    
    q_base = Q + b * stride_qb + h * stride_qh
    k_base = K + b * stride_kb + h * stride_kh
    v_base = V + b * stride_vb + h * stride_vh
    o_base = O + b * stride_ob + h * stride_oh
    
    offs_m = offset_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    
    # Q is loaded once per M-block. We evict it first so it doesn't pollute the L2 cache, 
    # leaving maximum L2 space for K and V which are shared across all M-blocks.
    q_ptrs = q_base + offs_m[:, None] * stride_qs + offs_d[None, :]
    mask_m = offs_m < S
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0, eviction_policy="evict_first")
    
    # Pre-apply combined Softmax scaling and Log-Sum-Exp base conversion outside the hot loop
    q = (q * scale).to(tl.bfloat16)
    
    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, D), tl.float32)
    
    k_ptrs = k_base + offs_n[:, None] * stride_ks + offs_d[None, :]
    v_ptrs = v_base + offs_n[:, None] * stride_vs + offs_d[None, :]
    
    num_n_blocks_full = S // BLOCK_N
    
    # =========================================================================================
    # CORE HOT LOOP (Full Blocks)
    # =========================================================================================
    for n_block in range(num_n_blocks_full):
        # Explicit evict_last cache modifiers. 
        # By retaining K and V in L2 cache across all M-blocks, memory bandwidth is slashed natively.
        k = tl.load(k_ptrs, eviction_policy="evict_last")
        v = tl.load(v_ptrs, eviction_policy="evict_last")
        
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

    # =========================================================================================
    # TAIL BLOCK
    # =========================================================================================
    if (S % BLOCK_N) != 0:
        mask_n = (num_n_blocks_full * BLOCK_N + offs_n) < S
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0, eviction_policy="evict_last")
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0, eviction_policy="evict_last")
        
        scores = tl.dot(q, k.T)
        scores = tl.where(mask_n[None, :], scores, -float("inf"))
        
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        safe_m_ij = tl.where(m_ij == -float("inf"), 0.0, m_ij)
        
        alpha = tl.math.exp2(m_i - safe_m_ij)
        p = tl.math.exp2(scores - safe_m_ij[:, None])
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        m_i = m_ij

    # Safely guard completely masked rows
    safe_l_i = tl.where(l_i == 0.0, 1.0, l_i)
    output = acc / safe_l_i[:, None]
    
    # Store output tile, evicting first since we never read it again
    o_ptrs = o_base + offs_m[:, None] * stride_os + offs_d[None, :]
    tl.store(o_ptrs, output.to(tl.bfloat16), mask=mask_m[:, None], eviction_policy="evict_first")
    
    # Transform base-2 LSE back mathematically to classic Natural Log Log-Sum-Exp Format
    LN2: tl.constexpr = 0.6931471805599453
    lse = (m_i + tl.math.log2(safe_l_i)) * LN2
    
    lse_base = LSE + b * stride_lseb + h * stride_lseh
    tl.store(lse_base + offs_m * stride_lses, lse, mask=mask_m, eviction_policy="evict_first")


def run(Q, K, V, O, LSE):
    """Compute Non-Causal Multi-Head Attention Forward mapping PyTorch's SDPA behavior."""
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    # Mathematically fuse SDPA scale mapping onto optimal base-2 hardware scaling
    RCP_LN2 = 1.4426950408889634
    scale = (1.0 / (D ** 0.5)) * RCP_LN2
    
    is_contiguous = Q.is_contiguous() and K.is_contiguous() and V.is_contiguous() and O.is_contiguous()
    
    # Grouped Program Ordering: 
    # The grid sequentially launches sequential M blocks belonging to the exact same head.
    # Since these M blocks iterate over the identical K & V sequences, L2 Cache intercepts and reuses K & V naturally!
    grid = lambda META: (triton.cdiv(S, META["BLOCK_M"]), B, H)
    
    _fwd_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        S, scale,
        D=D,
        IS_CONTIGUOUS=is_contiguous
    )