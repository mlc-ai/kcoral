import torch
import triton
import triton.language as tl
import math

@triton.jit
def _fwd_kernel(
    Q, K, V, O, LSE,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lb, stride_lh, stride_ls,
    S,
    SCALE,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    start_m = tl.program_id(0) * BLOCK_M
    b = tl.program_id(1)
    h = tl.program_id(2)

    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_ptrs = Q + b * stride_qb + h * stride_qh + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    k_ptrs = K + b * stride_kb + h * stride_kh + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = V + b * stride_vb + h * stride_vh + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # Load Q once
    q = tl.load(q_ptrs, mask=(offs_m[:, None] < S) & (offs_d[None, :] < BLOCK_D), other=0.0).to(tl.bfloat16)

    # Determine maximum number of steps required for causal attention for this query block
    max_n = tl.minimum(S, start_m + BLOCK_M)
    num_steps = (max_n + BLOCK_N - 1) // BLOCK_N

    for step in range(0, num_steps):
        start_n = step * BLOCK_N
        curr_offs_n = start_n + offs_n

        k = tl.load(k_ptrs, mask=(curr_offs_n[:, None] < S) & (offs_d[None, :] < BLOCK_D), other=0.0).to(tl.bfloat16)
        v = tl.load(v_ptrs, mask=(curr_offs_n[:, None] < S) & (offs_d[None, :] < BLOCK_D), other=0.0).to(tl.bfloat16)

        # Accumulate dot product in FP32
        qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32)
        
        # Scale in FP32 to maintain full precision
        qk = qk * SCALE

        # Causal mask and sequence boundary mask
        mask = (offs_m[:, None] >= curr_offs_n[None, :]) & (curr_offs_n[None, :] < S) & (offs_m[:, None] < S)
        qk = tl.where(mask, qk, -float("inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        
        # Safely handle completely masked rows to avoid inf - inf = nan
        safe_m_ij = tl.where(m_ij == -float("inf"), 0.0, m_ij)
        
        p = tl.math.exp2(qk - safe_m_ij[:, None])
        alpha = tl.math.exp2(m_i - safe_m_ij)
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        
        # Accumulate P @ V in FP32
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        
        m_i = m_ij
        k_ptrs += BLOCK_N * stride_ks
        v_ptrs += BLOCK_N * stride_vs

    safe_l_i = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / safe_l_i[:, None]
    
    # Convert LSE from base-2 back to natural logarithm
    LN2 = 0.6931471805599453
    lse_log2 = tl.where(l_i == 0.0, -float("inf"), m_i + tl.math.log2(safe_l_i))
    lse = lse_log2 * LN2

    o_ptrs = O + b * stride_ob + h * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(tl.bfloat16), mask=(offs_m[:, None] < S) & (offs_d[None, :] < BLOCK_D))

    lse_ptrs = LSE + b * stride_lb + h * stride_lh + offs_m * stride_ls
    tl.store(lse_ptrs, lse, mask=offs_m < S)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    # Base-2 scaling factor for log2 math
    RCP_LN2 = 1.4426950408889634
    scale = (1.0 / math.sqrt(D)) * RCP_LN2
    
    BLOCK_M = 128
    BLOCK_N = 64
    grid = (triton.cdiv(S, BLOCK_M), B, H)
    
    _fwd_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        S,
        scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=D,
        num_warps=8,
        num_stages=3
    )