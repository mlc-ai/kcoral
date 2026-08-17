import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
    ],
    key=['S']
)
@triton.jit
def mha_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    B, H, S, sm_scale,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    # Offset base pointers to the current batch and head
    q_offset = pid_b * stride_qb + pid_h * stride_qh
    k_offset = pid_b * stride_kb + pid_h * stride_kh
    v_offset = pid_b * stride_vb + pid_h * stride_vh
    o_offset = pid_b * stride_ob + pid_h * stride_oh
    lse_offset = pid_b * stride_lseb + pid_h * stride_lseh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)

    q_ptrs = Q_ptr + q_offset + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    q_mask = (offs_m[:, None] < S)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # Initialize statistics for online softmax
    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    num_n_blocks = tl.cdiv(S, BLOCK_N)
    for start_n_idx in range(num_n_blocks):
        start_n = start_n_idx * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)
        
        k_ptrs = K_ptr + k_offset + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        v_ptrs = V_ptr + v_offset + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
        
        k_mask = (offs_n[:, None] < S)
        v_mask = (offs_n[:, None] < S)
        
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)
        
        # Compute dot: Q @ K^T
        qk = tl.dot(q, tl.trans(k))
        qk = qk * sm_scale
        
        # Mask out-of-bounds sequence elements
        qk = tl.where(offs_n[None, :] < S, qk, float("-inf"))
        
        # Max scaling step
        m_i_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])
        
        # Update running norm
        l_i_new = alpha * l_i + tl.sum(p, axis=1)
        
        # Scale and update output accumulator
        p_bf16 = p.to(tl.bfloat16)
        acc = acc * alpha[:, None]
        acc = tl.dot(p_bf16, v, acc)
        
        m_i = m_i_new
        l_i = l_i_new

    # Finalize O and Log-Sum-Exp
    out = acc / l_i[:, None]
    lse = m_i + tl.log(l_i)
    
    o_ptrs = O_ptr + o_offset + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    lse_ptrs = LSE_ptr + lse_offset + offs_m * stride_lses
    
    tl.store(o_ptrs, out.to(tl.bfloat16), mask=q_mask)
    tl.store(lse_ptrs, lse, mask=offs_m < S)


def run(Q, K, V, O, LSE):
    """
    Computes Non-Causal Multi-Head Attention using standard pointer arithmetic.
    
    Args:
        Q, K, V: bfloat16 input tensors of shape (B, H, S, D).
        O: preallocated bfloat16 output tensor of shape (B, H, S, D).
        LSE: preallocated float32 output tensor of shape (B, H, S).
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    sm_scale = 1.0 / (D ** 0.5)
    
    grid = lambda META: (triton.cdiv(S, META['BLOCK_M']), B, H)
    
    mha_fwd_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        B=B, H=H, S=S, sm_scale=sm_scale,
        D=128
    )