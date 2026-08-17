import torch
import triton
import triton.language as tl
import math

from triton.tools.tensor_descriptor import TensorDescriptor

@triton.jit
def _attn_fwd_kernel(
    q_desc, k_desc, v_desc, o_desc,
    LSE,
    stride_lb, stride_lh, stride_ls,
    sm_scale,
    S, H,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    # Calculate global batch and head indices
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    # TMA load natively manages bound padding (out of bounds returns 0 safely)
    q = q_desc.load([pid_b, pid_h, pid_m * BLOCK_M, 0])

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, 128], dtype=tl.float32)

    # Calculate exactly where causal overlaps begin bounding standard full processing
    hi_unmasked = tl.minimum(pid_m * BLOCK_M, S)
    n_unmasked_steps = hi_unmasked // BLOCK_N
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    # 1. Unmasked Pipeline Phase: Maximizes uninterrupted pipelined block processing internally
    for step in tl.range(0, n_unmasked_steps):
        start_n = step * BLOCK_N
        k = k_desc.load([pid_b, pid_h, start_n, 0])
        v = v_desc.load([pid_b, pid_h, start_n, 0])
        
        qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32)
        qk = qk * sm_scale
        
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        
        alpha = tl.exp(m_i - m_ij)
        acc = acc * alpha[:, None]
        
        p_bf16 = tl.cast(p, tl.bfloat16)
        acc = tl.dot(p_bf16, v, acc, out_dtype=tl.float32)
        
        m_i = m_ij
        l_i = l_i * alpha + l_ij

    # 2. Causal Boundary Phase: Bounds sequence safely masking causal offsets 
    hi_causal = tl.minimum((pid_m + 1) * BLOCK_M, S)
    n_causal_steps = tl.cdiv(hi_causal, BLOCK_N)
    
    for step in tl.range(n_unmasked_steps, n_causal_steps):
        start_n = step * BLOCK_N
        k = k_desc.load([pid_b, pid_h, start_n, 0])
        v = v_desc.load([pid_b, pid_h, start_n, 0])
        
        qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32)
        qk = qk * sm_scale
        
        offs_n_curr = start_n + offs_n
        causal_mask = offs_m[:, None] >= offs_n_curr[None, :]
        qk = tl.where(causal_mask, qk, float("-inf"))
        
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        
        alpha = tl.exp(m_i - m_ij)
        acc = acc * alpha[:, None]
        
        p_bf16 = tl.cast(p, tl.bfloat16)
        acc = tl.dot(p_bf16, v, acc, out_dtype=tl.float32)
        
        m_i = m_ij
        l_i = l_i * alpha + l_ij

    # Final Math Normalizations
    inv_l = 1.0 / l_i
    acc = acc * inv_l[:, None]
    
    # Nat-log mapping exactly aligns seamlessly against PyTorch reference outputs
    lse = m_i + tl.log(l_i)

    # Push to TMA cleanly dumping memory dynamically avoiding garbage sequence row tails automatically
    o_desc.store([pid_b, pid_h, pid_m * BLOCK_M, 0], tl.cast(acc, tl.bfloat16))
    
    # Store LSE standardly clamping trailing bound artifacts mapping explicit tails
    mask_m = offs_m < S
    lse_ptrs = LSE + pid_b * stride_lb + pid_h * stride_lh + offs_m * stride_ls
    tl.store(lse_ptrs, lse, mask=mask_m)


def run(Q, K, V, O, LSE):
    """
    Computes standard explicit causal Multi-Head Attention forward pass via Blackwell native TMAs.
    Bypasses inefficient device-side `tl.make_tensor_descriptor` using Host definitions driving bandwidth fully.
    """
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    sm_scale = 1.0 / math.sqrt(D)

    # Establish exceptionally stable execution dimensions preventing L2 trashing while maintaining pipeline depths
    BLOCK_M = 128
    BLOCK_N = 128
    
    # Instantiate TMA bounds structurally over Host APIs keeping device overhead completely absent
    # Fully resistant against differing non-contiguous external strides intrinsically
    q_desc = TensorDescriptor.from_tensor(Q, [1, 1, BLOCK_M, D])
    k_desc = TensorDescriptor.from_tensor(K, [1, 1, BLOCK_N, D])
    v_desc = TensorDescriptor.from_tensor(V, [1, 1, BLOCK_N, D])
    o_desc = TensorDescriptor.from_tensor(O, [1, 1, BLOCK_M, D])

    # Sequence loops inner blocks mapping identically cached K/V natively directly across consecutive CTAs 
    grid = (triton.cdiv(S, BLOCK_M), B * H)
    
    _attn_fwd_kernel[grid](
        q_desc, k_desc, v_desc, o_desc,
        LSE,
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        sm_scale,
        S, H,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=8,
        num_stages=2
    )