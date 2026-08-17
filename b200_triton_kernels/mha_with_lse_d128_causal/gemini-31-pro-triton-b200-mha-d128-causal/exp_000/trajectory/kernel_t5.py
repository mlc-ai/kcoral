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
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
    ],
    key=['S']
)
@triton.jit
def _attn_fwd_kernel(
    Q, K, V, sm_scale_log2,
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

    # Exit early if the assigned block covers queries entirely out of bounds
    if pid_m * BLOCK_M >= S:
        return
    
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

    # 1. Unmasked Execution Phase: Maximizes computational efficiency without bound checks
    hi_unmasked = tl.minimum(pid_m * BLOCK_M, S)
    n_unmasked_steps = hi_unmasked // BLOCK_N
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    for step in tl.range(0, n_unmasked_steps):
        start_n = step * BLOCK_N
        k = k_desc.load([start_n, 0])
        v = v_desc.load([start_n, 0])
        
        # Multiply strictly evaluated FP32 dot natively with mathematical log2 scale preserving full FP32 accuracy 
        qk = tl.dot(q, tl.trans(k)) * sm_scale_log2
        
        # Base-2 Online Softmax Updates
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        
        alpha = tl.exp2(m_i - m_ij)
        acc = acc * alpha[:, None]
        
        p_bf16 = tl.cast(p, tl.bfloat16)
        acc = tl.dot(p_bf16, v, acc)
        
        m_i = m_ij
        l_i = l_i * alpha + l_ij

    # 2. Causal Boundary Execution Phase: Safely governs sequence and causal limits via attention masks
    hi_causal = tl.minimum((pid_m + 1) * BLOCK_M, S)
    n_causal_steps = tl.cdiv(hi_causal, BLOCK_N)
    
    for step in tl.range(n_unmasked_steps, n_causal_steps):
        start_n = step * BLOCK_N
        k = k_desc.load([start_n, 0])
        v = v_desc.load([start_n, 0])
        
        qk = tl.dot(q, tl.trans(k)) * sm_scale_log2
        
        # Enforce Causal Limit Constraints (Safely masks Sequence bounds intrinsically as well)
        offs_n_curr = start_n + offs_n
        causal_mask = offs_m[:, None] >= offs_n_curr[None, :]
        qk = tl.where(causal_mask, qk, float("-inf"))
        
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        
        alpha = tl.exp2(m_i - m_ij)
        acc = acc * alpha[:, None]
        
        p_bf16 = tl.cast(p, tl.bfloat16)
        acc = tl.dot(p_bf16, v, acc)
        
        m_i = m_ij
        l_i = l_i * alpha + l_ij

    # Final Softmax Normalization
    inv_l = 1.0 / l_i
    acc = acc * inv_l[:, None]
    
    # Calculate Natural Log-Sum-Exp natively mirroring PyTorch exact formulation
    # Convert safely from computationally faster Base-2 representations back into Natural Log Base-e
    LN2 = 0.6931471805599453  # ln(2)
    lse = m_i * LN2 + tl.log(l_i)

    # Safely commit TMA memory avoiding writing logically discarded padding rows directly
    o_desc.store([pid_m * BLOCK_M, 0], tl.cast(acc, tl.bfloat16))
    
    # Conventional elementwise boundaries mapped strictly projecting final LSE outputs
    mask_m = offs_m < S
    lse_ptrs = LSE + lse_offset + offs_m * stride_ls
    tl.store(lse_ptrs, lse, mask=mask_m)


def run(Q, K, V, O, LSE):
    """
    Computes causal Multi-Head Attention forward pass returning Output and Log-Sum-Exp.
    Architecturally binds to SM100 limits operating natively across Device TMA descriptors mapping unmasked boundaries optimally.
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    sm_scale = 1.0 / math.sqrt(D)
    
    # Transform base multiplicative scalar scale efficiently shifting mathematical bases seamlessly towards base-2 computation limits
    # log2(e) approx 1.4426950408889634
    sm_scale_log2 = sm_scale * 1.4426950408889634

    # Scheduling maximizes intrinsic identical K & V structural reuse inherently within standard L2 block hierarchies
    grid = lambda META: (triton.cdiv(S, META['BLOCK_M']), B * H)
    
    _attn_fwd_kernel[grid](
        Q, K, V, sm_scale_log2,
        O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        S,
        B=B, H=H, D=D
    )