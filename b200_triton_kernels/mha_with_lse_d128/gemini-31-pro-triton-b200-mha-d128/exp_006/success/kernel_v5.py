import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        # Maximize Arithmetic Intensity (Favors large Shared Memory utilization)
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64},  num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 256}, num_warps=4, num_stages=3),
        
        # Balanced Standard Shapes
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4, num_stages=3),
        
        # Maximize Occupancy (Low register footprint per CTA)
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64},  num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64},  num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64},  num_warps=4, num_stages=4),
    ],
    key=["S"],
)
@triton.jit
def _attention_kernel_ptr(
    Q, K, V, O, LSE,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lb, stride_lh, stride_ls,
    S, softmax_scale_log2,
    EXACT_MULTIPLE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    q_base = Q + pid_b * stride_qb + pid_h * stride_qh
    k_base = K + pid_b * stride_kb + pid_h * stride_kh
    v_base = V + pid_b * stride_vb + pid_h * stride_vh
    o_base = O + pid_b * stride_ob + pid_h * stride_oh
    lse_base = LSE + pid_b * stride_lb + pid_h * stride_lh

    q_ptrs = q_base + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    
    # Q Load (Skip masking if geometry matches)
    if EXACT_MULTIPLE:
        q = tl.load(q_ptrs)
    else:
        is_full_m = (pid_m + 1) * BLOCK_M <= S
        if is_full_m:
            q = tl.load(q_ptrs)
        else:
            q = tl.load(q_ptrs, mask=(offs_m[:, None] < S), other=0.0)

    # Absorb temperature and LN2 conversions entirely into Q registers
    q_scaled = (q * softmax_scale_log2).to(q.dtype)

    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)

    offs_n = tl.arange(0, BLOCK_N)
    k_ptrs = k_base + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = v_base + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd

    num_full_blocks = S // BLOCK_N

    # Native Python range lets Triton compiler optimally stage and software-pipeline loop loads.
    for _ in range(0, num_full_blocks):
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)
        
        scores = tl.dot(q_scaled, k.T)
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        
        alpha = tl.math.exp2(m_i - m_ij)
        p = tl.math.exp2(scores - m_ij[:, None])
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(q.dtype), v, acc)
        m_i = m_ij
        
        k_ptrs += BLOCK_N * stride_ks
        v_ptrs += BLOCK_N * stride_vs

    # Tail block traversal pruned from trace entirely on strictly compatible dimensions
    if not EXACT_MULTIPLE:
        has_tail = (S % BLOCK_N != 0)
        if has_tail:
            offset_n = num_full_blocks * BLOCK_N
            offs_n_tail = offset_n + tl.arange(0, BLOCK_N)
            
            mask_k = (offs_n_tail[:, None] < S)
            k = tl.load(k_ptrs, mask=mask_k, other=0.0)
            v = tl.load(v_ptrs, mask=mask_k, other=0.0)
            
            scores = tl.dot(q_scaled, k.T)
            scores = tl.where(offs_n_tail[None, :] < S, scores, -float("inf"))
            
            m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
            alpha = tl.math.exp2(m_i - m_ij)
            p = tl.math.exp2(scores - m_ij[:, None])
            
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None]
            acc = tl.dot(p.to(q.dtype), v, acc)
            m_i = m_ij

    inv_l_i = 1.0 / l_i
    out = acc * inv_l_i[:, None]
    
    # Restore Log-Sum-Exp outputs from base-2 mathematical state back to natural log expectations
    LN2: tl.constexpr = 0.6931471805599453
    lse = (m_i + tl.math.log2(l_i)) * LN2
    
    o_ptrs = o_base + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    lse_ptrs = lse_base + offs_m * stride_ls

    if EXACT_MULTIPLE:
        tl.store(o_ptrs, out.to(q.dtype))
        tl.store(lse_ptrs, lse)
    else:
        if is_full_m:
            tl.store(o_ptrs, out.to(q.dtype))
            tl.store(lse_ptrs, lse)
        else:
            tl.store(o_ptrs, out.to(q.dtype), mask=(offs_m[:, None] < S))
            tl.store(lse_ptrs, lse, mask=(offs_m < S))


def run(Q, K, V, O, LSE):
    """
    Computes Non-Causal Multi-Head Attention forward returning output targets and LSE.
    Targeting natively optimized standard Triton loops and heuristics for SM100 limits.
    """
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    softmax_scale = 1.0 / (D ** 0.5)
    RCP_LN2 = 1.4426950408889634
    softmax_scale_log2 = softmax_scale * RCP_LN2

    # Maximum configuration BLOCK size hits ~256. Proving exact multiples skips dynamic branch traces.
    is_exact_multiple = bool(S % 256 == 0)

    # L2 Cache Broadcasting optimization: Flattening M across grid[0] ensures GPU schedulers distribute 
    # tasks belonging to identical batch components and attention heads over concurrent SM waves naturally. 
    grid = lambda META: (triton.cdiv(S, META["BLOCK_M"]), B, H)
    
    _attention_kernel_ptr[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        S, softmax_scale_log2,
        EXACT_MULTIPLE=is_exact_multiple,
        BLOCK_D=128
    )