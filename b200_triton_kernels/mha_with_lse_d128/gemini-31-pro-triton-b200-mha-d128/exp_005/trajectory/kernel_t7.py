import torch
import triton
import triton.language as tl

# Provide Triton's allocator for creating device-side TMA descriptors
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

@triton.autotune(
    configs=[
        # High-utilization clustered configs designed to aggressively prefetch K and V via deep TMA pipelines 
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'LOOP_STAGES': 3, 'UNROLL': 2}, num_stages=3, num_warps=8, num_ctas=1),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'LOOP_STAGES': 3, 'UNROLL': 2}, num_stages=3, num_warps=8, num_ctas=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'LOOP_STAGES': 4, 'UNROLL': 2}, num_stages=4, num_warps=4, num_ctas=1),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'LOOP_STAGES': 4, 'UNROLL': 2}, num_stages=4, num_warps=4, num_ctas=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'LOOP_STAGES': 4, 'UNROLL': 2}, num_stages=4, num_warps=4, num_ctas=1),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'LOOP_STAGES': 4, 'UNROLL': 2}, num_stages=4, num_warps=4, num_ctas=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'LOOP_STAGES': 2, 'UNROLL': 1}, num_stages=2, num_warps=8, num_ctas=1),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'LOOP_STAGES': 2, 'UNROLL': 1}, num_stages=2, num_warps=8, num_ctas=1),
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
    LOOP_STAGES: tl.constexpr,
    UNROLL: tl.constexpr,
    DIVISIBLE_M: tl.constexpr,
    DIVISIBLE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(S, BLOCK_M)
    
    # Extract linear scheduling indices 
    pid_m = pid % grid_m
    pid_bh = pid // grid_m
    b = pid_bh // H
    h = pid_bh % H
    
    start_m = pid_m * BLOCK_M
    
    # Establish base evaluation pointers
    q_base = Q + b * stride_qb + h * stride_qh
    k_base = K + b * stride_kb + h * stride_kh
    v_base = V + b * stride_vb + h * stride_vh
    o_base = O + b * stride_ob + h * stride_oh
    
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    
    # Load Q using standard pointers. Bypassing TMA here keeps Q strictly inside registers rather 
    # than SMEM, saving 32-64KB per CTA allowing WGMMA to double available pipeline depth on SM100.
    q_ptrs = q_base + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    if DIVISIBLE_M:
        q = tl.load(q_ptrs)
    else:
        m_mask = offs_m < S
        q = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0)
        
    # Scale Q optimally in float registers via base-2 offsets
    RCP_LN2: tl.constexpr = 1.4426950408889634
    q = (q.to(tl.float32) * (scale * RCP_LN2)).to(tl.bfloat16)
    
    # Allocate TMA descriptors specifically for K and V enabling asynchronous multi-buffering 
    k_desc = tl.make_tensor_descriptor(k_base, shape=[S, D], strides=[stride_ks, 1], block_shape=[BLOCK_N, D], padding_option="zero")
    v_desc = tl.make_tensor_descriptor(v_base, shape=[S, D], strides=[stride_vs, 1], block_shape=[BLOCK_N, D], padding_option="zero")
    
    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)
    
    num_blocks = S // BLOCK_N
    limit = num_blocks * BLOCK_N
    
    # Advanced Unrolled Iterator strictly executing fully valid ranges utilizing hardware WGMMA overlap naturally
    for start_n in tl.range(0, limit, BLOCK_N, num_stages=LOOP_STAGES, loop_unroll_factor=UNROLL):
        k = k_desc.load([start_n, 0])
        v = v_desc.load([start_n, 0])
        
        scores = tl.dot(q, k.T, out_dtype=tl.float32)
        
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.math.exp2(m_i - m_ij)
        p = tl.math.exp2(scores - m_ij[:, None])
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc, out_dtype=tl.float32)
        
        m_i = m_ij

    # Precise tail-block handling seamlessly skipped dynamically via compile-time verification
    if not DIVISIBLE_N:
        if S % BLOCK_N != 0:
            k = k_desc.load([limit, 0])
            v = v_desc.load([limit, 0])
            
            scores = tl.dot(q, k.T, out_dtype=tl.float32)
            
            offs_n = limit + tl.arange(0, BLOCK_N)
            valid_score = offs_n[None, :] < S
            scores = tl.where(valid_score, scores, -float("inf"))
            
            m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
            alpha = tl.math.exp2(m_i - m_ij)
            p = tl.math.exp2(scores - m_ij[:, None])
            
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None]
            acc = tl.dot(p.to(tl.bfloat16), v, acc, out_dtype=tl.float32)
            
            m_i = m_ij

    # Fast division leveraging scalar multiplication natively without safe-clamp checks (because valid rows are guaranteed)
    inv_l_i = 1.0 / l_i
    output = acc * inv_l_i[:, None]
    
    # Emit WGMMA-computed registers directly avoiding SMEM capacity hits using pointer writes
    o_ptrs = o_base + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    lse_ptrs = LSE + b * stride_lseb + h * stride_lseh + offs_m * stride_lses
    
    LN2: tl.constexpr = 0.6931471805599453
    lse = (m_i + tl.math.log2(l_i)) * LN2
    
    if DIVISIBLE_M:
        tl.store(o_ptrs, output.to(tl.bfloat16))
        tl.store(lse_ptrs, lse)
    else:
        tl.store(o_ptrs, output.to(tl.bfloat16), mask=m_mask[:, None])
        tl.store(lse_ptrs, lse, mask=m_mask)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    scale = D ** -0.5
    
    # Establish strict sequence alignments resolving compile-time conditionals internally
    divisible_m = (S % 256 == 0)
    divisible_n = (S % 256 == 0)
    
    # Cluster schedule L2 hit priority ensuring linear head extraction
    grid = lambda META: (B * H * triton.cdiv(S, META["BLOCK_M"]), )
    
    _mha_fwd_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        S, H, scale,
        D=128,
        DIVISIBLE_M=divisible_m,
        DIVISIBLE_N=divisible_n,
    )