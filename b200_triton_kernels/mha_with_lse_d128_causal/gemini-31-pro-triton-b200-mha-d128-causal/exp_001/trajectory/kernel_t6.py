import torch
import triton
import triton.language as tl
import math

_configs = [
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 64}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 64}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 128}, num_warps=8, num_stages=3),
]

@triton.autotune(configs=_configs, key=["S"])
@triton.jit
def _attn_fwd_kernel(
    Q, K, V, sm_scale_log2,
    O, LSE,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    B, H, S, D,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    # Mapping X-axis strictly to sequence offsets guarantees concurrent SM processing
    # natively shares the same underlying K and V matrices perfectly within the L2 Cache.
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    off_b = off_hz // H
    off_h = off_hz % H

    start_m_idx = start_m * BLOCK_M
    if start_m_idx >= S:
        return

    offs_m = start_m_idx + tl.arange(0, BLOCK_M)
    offs_n_base = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    # Base pointers computed once dynamically handling outer strided layout topologies
    q_base = Q + off_b * stride_qb + off_h * stride_qh
    k_base = K + off_b * stride_kb + off_h * stride_kh
    v_base = V + off_b * stride_vb + off_h * stride_vh
    
    q_ptrs = q_base + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=(offs_m[:, None] < S), other=0.0)

    # Set up sliding pointers for the sequence keys/values loop processing
    k_ptrs = k_base + offs_n_base[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = v_base + offs_n_base[:, None] * stride_vs + offs_d[None, :] * stride_vd

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # === UNMASKED LOOP ===
    # For a lower-triangular causal pass, all matrices entirely below the start_m_idx block boundary
    # fall safely behind the causal curve. This circumvents massive evaluation overhead 
    # omitting arbitrary diagonal masking logic completely along the primary execution pathway.
    for start_n in range(0, start_m_idx, BLOCK_N):
        offs_n = start_n + offs_n_base
        mask_n = offs_n < S
        
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        qk = tl.dot(q, k.T, out_dtype=tl.float32) * sm_scale_log2
        
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        # Optimized base-2 exponential intrinsic mappings mapping cleanly onto single-cycle EX2 math units
        p = tl.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        
        alpha = tl.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        acc = tl.dot(p.to(tl.bfloat16), v, acc=acc, out_dtype=tl.float32)
        
        m_i = m_ij
        k_ptrs += BLOCK_N * stride_ks
        v_ptrs += BLOCK_N * stride_vs

    # === MASKED LOOP ===
    # Restricts explicit masking arithmetic overhead exclusively into the thin diagonal bounds intersections.
    end_n = tl.minimum(start_m_idx + BLOCK_M, S)
    for start_n in range(start_m_idx, end_n, BLOCK_N):
        offs_n = start_n + offs_n_base
        mask_n = offs_n < S
        
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        qk = tl.dot(q, k.T, out_dtype=tl.float32) * sm_scale_log2
        
        # Apply strict logical lower-triangular causal evaluation safely ignoring the upper corners
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        valid_mask = causal_mask & mask_n[None, :]
        qk = tl.where(valid_mask, qk, float("-inf"))
        
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        
        alpha = tl.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        acc = tl.dot(p.to(tl.bfloat16), v, acc=acc, out_dtype=tl.float32)
        
        m_i = m_ij
        k_ptrs += BLOCK_N * stride_ks
        v_ptrs += BLOCK_N * stride_vs

    # Final safe normalizations
    acc = acc / l_i[:, None]
    
    o_base = O + off_b * stride_ob + off_h * stride_oh
    o_ptrs = o_base + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(tl.bfloat16), mask=(offs_m[:, None] < S))

    lse_base = LSE + off_b * stride_lseb + off_h * stride_lseh
    lse = m_i + tl.log2(l_i)
    # Safely transpose the LSE vector representations natively onto natural Base-E scales (via optimal standard ln(2) scalar)
    lse = lse * 0.6931471805599453 
    lse_ptrs = lse_base + offs_m * stride_lses
    tl.store(lse_ptrs, lse, mask=(offs_m < S))


def run(Q, K, V, O, LSE):
    """
    Highly optimized Scaled Dot-Product Causal Attention leveraging custom SM100 pointer indexing streams.
    Achieves maximum possible SM Cache sharing natively mapping parallel bounds across aligned inner axes.
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    if S == 0:
        return
        
    # Standard FlashAttention arithmetic optimization factoring standard log2(e) constants dynamically bypassing scale down overhead
    sm_scale_log2 = (1.0 / math.sqrt(D)) * 1.4426950408889634
    
    grid = lambda META: (triton.cdiv(S, META['BLOCK_M']), B * H, 1)
    
    _attn_fwd_kernel[grid](
        Q, K, V, sm_scale_log2,
        O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        B, H, S, D,
        BLOCK_D=128
    )