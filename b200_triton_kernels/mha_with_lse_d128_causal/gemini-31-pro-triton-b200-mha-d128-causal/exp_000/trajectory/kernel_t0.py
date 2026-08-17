import torch
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=3, num_warps=8),
    ],
    key=['S']
)
@triton.jit
def _attn_fwd_kernel(
    Q, K, V, sm_scale,
    O, LSE,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lb, stride_lh, stride_ls,
    S,
    B: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    # Offset for current batch and head
    q_offset = pid_b * stride_qb + pid_h * stride_qh
    k_offset = pid_b * stride_kb + pid_h * stride_kh
    v_offset = pid_b * stride_vb + pid_h * stride_vh
    o_offset = pid_b * stride_ob + pid_h * stride_oh
    lse_offset = pid_b * stride_lb + pid_h * stride_lh

    # Tile index ranges
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    # Initialize pointers
    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    k_ptrs = K + k_offset + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = V + v_offset + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd

    # Load Q with boundary checks
    mask_m = offs_m < S
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    # Accumulator structures for softmax
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    # Limit K loop by sequence length and causal bounds
    max_n = tl.minimum(S, (pid_m + 1) * BLOCK_M)

    for start_n in range(0, max_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n_curr = start_n + offs_n
        
        mask_n = offs_n_curr < S
        
        # Load K and V
        k = tl.load(k_ptrs + start_n * stride_ks, mask=mask_n[:, None], other=0.0)
        v = tl.load(v_ptrs + start_n * stride_vs, mask=mask_n[:, None], other=0.0)
        
        # Compute Q * K^T
        qk = tl.dot(q, tl.trans(k)) * sm_scale
        
        # Causal mask formulation
        causal_mask = offs_m[:, None] >= offs_n_curr[None, :]
        qk = tl.where(causal_mask & mask_n[None, :], qk, float("-inf"))
        
        # Flash attention online softmax update
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        
        l_ij = tl.sum(p, 1)
        
        alpha = tl.exp(m_i - m_ij)
        acc = acc * alpha[:, None]
        
        p_bf16 = tl.cast(p, tl.bfloat16)
        acc = tl.dot(p_bf16, v, acc)
        
        m_i = m_ij
        l_i = l_i * alpha + l_ij

    # Final normalization
    acc = acc / l_i[:, None]
    
    # Calculate Natural Log-Sum-Exp (LSE) exactly mapping to reference mathematically
    lse = m_i + tl.log(l_i)

    # Store normalized output and log-sum-exp
    o_ptrs = O + o_offset + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(o_ptrs, tl.cast(acc, tl.bfloat16), mask=mask_m[:, None])
    
    lse_ptrs = LSE + lse_offset + offs_m * stride_ls
    tl.store(lse_ptrs, lse, mask=mask_m)


def run(Q, K, V, O, LSE):
    """
    Computes causal Multi-Head Attention forward pass returning Output and Log-Sum-Exp.
    All inputs and outputs are implicitly expected to have matching (B, H, S, D) dimensions
    except LSE which is expected to be shape (B, H, S).
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    sm_scale = 1.0 / math.sqrt(D)

    grid = lambda META: (triton.cdiv(S, META['BLOCK_M']), B * H)
    
    _attn_fwd_kernel[grid](
        Q, K, V, sm_scale,
        O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        S,
        B=B, H=H, D=D
    )