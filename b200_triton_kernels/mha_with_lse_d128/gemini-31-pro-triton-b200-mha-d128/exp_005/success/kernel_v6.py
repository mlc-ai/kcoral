import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        # High occupancy configs optimizing for multiple CTAs resident per SM on Blackwell
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
        
        # Larger memory footprint targeting deeper pipeline stages overlapping natively
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=4, num_warps=4),
    ],
    key=['S']
)
@triton.jit
def _mha_fwd_kernel(
    Q, K, V, O, LSE,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    S, H, scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(S, BLOCK_M)
    
    # Establish block localization explicitly
    pid_m = pid % grid_m
    pid_bh = pid // grid_m
    b = pid_bh // H
    h = pid_bh % H
    
    start_m = pid_m * BLOCK_M
    
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    
    # Establish precise block pointer increments natively
    q_ptrs = Q + b * stride_qb + h * stride_qh + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    k_ptrs = K + b * stride_kb + h * stride_kh + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = V + b * stride_vb + h * stride_vh + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
    
    # Isolate boundary padding against non-evenly divisible sequences globally 
    m_mask = offs_m < S
    q = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0)
    
    # Pre-scale logical queries to negate sequential matrix multiplication overheads completely
    RCP_LN2: tl.constexpr = 1.4426950408889634
    scale_base2 = scale * RCP_LN2
    q = (q.to(tl.float32) * scale_base2).to(tl.bfloat16)
    
    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)
    
    # Compute guaranteed fully valid loop iterations devoid of bounding conditionals seamlessly
    num_blocks = S // BLOCK_N
    for _ in range(num_blocks):
        # Native unmasked retrieval mapping linearly to optimal vectorized ld.global
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)
        
        # WGMMA Matrix Products
        scores = tl.dot(q, k.T)
        
        # Stable Log-Sum Exp formulation directly utilizing MUFU EX2 natively without condition clamps
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.math.exp2(m_i - m_ij)
        p = tl.math.exp2(scores - m_ij[:, None])
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        
        m_i = m_ij
        k_ptrs += BLOCK_N * stride_ks
        v_ptrs += BLOCK_N * stride_vs

    # Tail conditional branch masking sequentially resolving partial end blocks optimally
    remainder = S % BLOCK_N
    if remainder > 0:
        n_mask = offs_n < remainder
        k = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)
        
        scores = tl.dot(q, k.T)
        scores = tl.where(n_mask[None, :], scores, -float("inf"))
        
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.math.exp2(m_i - m_ij)
        p = tl.math.exp2(scores - m_ij[:, None])
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        
        m_i = m_ij

    safe_l_i = tl.where(l_i == 0.0, 1.0, l_i)
    output = acc / safe_l_i[:, None]
    
    LN2: tl.constexpr = 0.6931471805599453
    lse = (m_i + tl.math.log2(safe_l_i)) * LN2
    
    # Unload context sequentially 
    o_ptrs = O + b * stride_ob + h * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(o_ptrs, output.to(tl.bfloat16), mask=m_mask[:, None])
    
    lse_ptrs = LSE + b * stride_lseb + h * stride_lseh + offs_m * stride_lses
    tl.store(lse_ptrs, lse, mask=m_mask)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    scale = D ** -0.5
    
    grid = lambda META: (B * H * triton.cdiv(S, META["BLOCK_M"]), )
    
    _mha_fwd_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        S, H, scale,
        D=128
    )