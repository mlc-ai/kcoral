import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=4),
    ],
    key=["S"],
)
@triton.jit
def _fwd_kernel(
    Q, K, V, O, LSE,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    B, H, S, D,
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    # Q pointers
    q_ptrs = Q + (b * stride_qb + h * stride_qh + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd)
    mask_m = offs_m < S
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    # Accumulators and running statistics
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    m_i = tl.full((BLOCK_M,), -float('inf'), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)

    # Base pointers for K and V to avoid redundant calculations
    k_base = K + (b * stride_kb + h * stride_kh)
    v_base = V + (b * stride_vb + h * stride_vh)

    for start_n in range(0, S, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S
        
        # Load transposed K
        k_ptrs = k_base + (offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd)
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        k_t = tl.trans(k)
        
        # Q @ K^T / sqrt(D)
        qk = tl.dot(q, k_t, out_dtype=tl.float32)
        qk = qk * sm_scale
        
        # Apply attention mask for sequence boundary
        qk = tl.where(mask_n[None, :], qk, float('-inf'))
        
        # Running max
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        
        # Compute exponent and sum
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        
        # Update running max and scaling factor
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        
        # Load V and aggregate
        p_bf16 = p.to(tl.bfloat16)
        v_ptrs = v_base + (offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        
        acc = acc * alpha[:, None]
        acc = tl.dot(p_bf16, v, acc, out_dtype=tl.float32)

    # Normalize accumulator
    acc = acc / l_i[:, None]
    
    # Store output O
    out = acc.to(tl.bfloat16)
    o_base = O + (b * stride_ob + h * stride_oh)
    o_ptrs = o_base + (offs_m[:, None] * stride_os + offs_d[None, :] * stride_od)
    tl.store(o_ptrs, out, mask=mask_m[:, None])
    
    # Store LSE (log-sum-exp in natural-log)
    lse = m_i + tl.log(l_i)
    lse_base = LSE + (b * stride_lseb + h * stride_lseh)
    lse_ptrs = lse_base + offs_m
    tl.store(lse_ptrs, lse, mask=mask_m)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    if S == 0:
        return

    grid = lambda META: (triton.cdiv(S, META['BLOCK_M']), B * H)
    sm_scale = 1.0 / (D ** 0.5)

    _fwd_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        B, H, S, D,
        sm_scale,
        BLOCK_D=128
    )