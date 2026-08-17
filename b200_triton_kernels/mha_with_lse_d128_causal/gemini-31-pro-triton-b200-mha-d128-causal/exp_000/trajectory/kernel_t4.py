import torch
import triton
import triton.language as tl
import math

# Provide device allocator for Triton's internal descriptor structures required by TMA on Blackwell
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'WARP_SPEC': True}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'WARP_SPEC': False}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'WARP_SPEC': True}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'WARP_SPEC': False}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'WARP_SPEC': True}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'WARP_SPEC': False}, num_stages=3, num_warps=8),
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
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, WARP_SPEC: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    # Evaluate memory offsets mapping to the current batch and head
    q_offset = pid_b * stride_qb + pid_h * stride_qh
    k_offset = pid_b * stride_kb + pid_h * stride_kh
    v_offset = pid_b * stride_vb + pid_h * stride_vh
    o_offset = pid_b * stride_ob + pid_h * stride_oh
    lse_offset = pid_b * stride_lb + pid_h * stride_lh

    # Device-side TensorDescriptors mapped cleanly for native TMA paths on SM100
    q_desc = tl.make_tensor_descriptor(
        Q + q_offset, shape=[S, D], strides=[stride_qs, stride_qd],
        block_shape=[BLOCK_M, D], padding_option="zero"
    )
    k_desc = tl.make_tensor_descriptor(
        K + k_offset, shape=[S, D], strides=[stride_ks, stride_kd],
        block_shape=[BLOCK_N, D], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        V + v_offset, shape=[S, D], strides=[stride_vs, stride_vd],
        block_shape=[BLOCK_N, D], padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        O + o_offset, shape=[S, D], strides=[stride_os, stride_od],
        block_shape=[BLOCK_M, D], padding_option="zero"
    )

    # Load block's full queries natively handling out of bounds padding explicitly via TMA
    q = q_desc.load([pid_m * BLOCK_M, 0])

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    # Establish loop boundaries mapped accurately to sequence limits
    hi_causal = tl.minimum((pid_m + 1) * BLOCK_M, S)
    n_steps = tl.cdiv(hi_causal, BLOCK_N)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    # Single pipeline loop structure maximizing compiler TMA and MMA overlapping
    # Integrates AWS natively granting compiler permissions for TMEM optimizations
    for step in tl.range(0, n_steps, warp_specialize=WARP_SPEC):
        start_n = step * BLOCK_N
        k = k_desc.load([start_n, 0])
        v = v_desc.load([start_n, 0])
        
        # Calculate raw correlations and strictly evaluate FP32 multiplicative constraint inside loop
        qk = tl.dot(q, tl.trans(k))
        qk = qk * sm_scale
        
        # Single causal validation cleanly maps over valid ranges implicitly via ALU masking overlapping MMAs
        offs_n_curr = start_n + offs_n
        causal_mask = offs_m[:, None] >= offs_n_curr[None, :]
        qk = tl.where(causal_mask, qk, float("-inf"))
        
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        
        alpha = tl.exp(m_i - m_ij)
        acc = acc * alpha[:, None]
        
        p_bf16 = tl.cast(p, tl.bfloat16)
        acc = tl.dot(p_bf16, v, acc)
        
        m_i = m_ij
        l_i = l_i * alpha + l_ij

    # Final Softmax Normalization
    acc = acc / l_i[:, None]
    
    # Calculate Natural Log-Sum-Exp natively matching exactly the standard mathematical layout mapping
    lse = m_i + tl.log(l_i)

    # Safely commit TMA memory avoiding writing logically discarded padding rows directly
    o_desc.store([pid_m * BLOCK_M, 0], tl.cast(acc, tl.bfloat16))
    
    # Conventional elementwise boundaries mapped exactly protecting specific layout sequence LSE tail limits
    mask_m = offs_m < S
    lse_ptrs = LSE + lse_offset + offs_m * stride_ls
    tl.store(lse_ptrs, lse, mask=mask_m)


def run(Q, K, V, O, LSE):
    """
    Computes causal Multi-Head Attention forward pass returning Output and Log-Sum-Exp.
    Architecturally binds to SM100 limits operating natively across Device TMA routines.
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    sm_scale = 1.0 / math.sqrt(D)

    # Schedule intrinsically maximizes identical K & V reuse internally within standard L2 caches
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