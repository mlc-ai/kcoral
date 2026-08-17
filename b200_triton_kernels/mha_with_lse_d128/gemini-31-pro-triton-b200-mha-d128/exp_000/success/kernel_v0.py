import math
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
    key=['S'],
)
@triton.jit
def _attn_fwd_kernel(
    Q, K, V, sm_scale,
    O, LSE,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    stride_lsez, stride_lseh, stride_lsem,
    S, H,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
):
    # Grid coordinates
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    batch = off_hz // H
    head = off_hz % H
    
    # Base offsets for this batch & head
    q_offset = batch * stride_qz + head * stride_qh
    k_offset = batch * stride_kz + head * stride_kh
    v_offset = batch * stride_vz + head * stride_vh
    o_offset = batch * stride_oz + head * stride_oh
    lse_offset = batch * stride_lsez + head * stride_lseh
    
    # Local element offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    # Block pointers
    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_ptrs = K + k_offset + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    v_ptrs = V + v_offset + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
    
    # Softmax running statistics
    # Initialize with 0.0 for fully masked rows to avoid NaN when computing exp(m_i - m_i_new)
    m_i = tl.where(offs_m < S, float("-inf"), 0.0)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    
    # Load Q
    q = tl.load(q_ptrs, mask=offs_m[:, None] < S, other=0.0)
    
    # Iterate over K and V blocks
    for start_n in range(0, S, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n_curr = start_n + offs_n
        
        # Load K
        k = tl.load(k_ptrs, mask=offs_n_curr[:, None] < S, other=0.0)
        
        # Compute q @ k.T
        qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32)
        qk = qk * sm_scale
        
        # Apply causal / padding mask
        qk = tl.where((offs_m[:, None] < S) & (offs_n_curr[None, :] < S), qk, float("-inf"))
        
        # Softmax inner step
        m_ij = tl.max(qk, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])
        p = tl.where((offs_m[:, None] < S) & (offs_n_curr[None, :] < S), p, 0.0)
        
        # Load V
        v = tl.load(v_ptrs, mask=offs_n_curr[:, None] < S, other=0.0)
        p_cast = p.to(tl.bfloat16)
        
        # Update accumulator
        acc = acc * alpha[:, None]
        acc = tl.dot(p_cast, v, acc, out_dtype=tl.float32)
        
        # Update running stats
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        
        # Advance K and V pointers
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn
        
    # Epilogue
    # Safeguard against l_i being exactly 0 for masked rows to avoid NaN
    l_i_safe = tl.where(offs_m < S, l_i, 1.0)
    acc = acc / l_i_safe[:, None]
    lse = m_i + tl.log(l_i_safe)
    
    # Store LSE
    lse_ptrs = LSE + lse_offset + offs_m * stride_lsem
    tl.store(lse_ptrs, lse, mask=offs_m < S)
    
    # Store O
    o_ptrs = O + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(tl.bfloat16), mask=offs_m[:, None] < S)


def run(Q, K, V, O, LSE):
    """
    Computes Multi-Head Attention forward pass and log-sum-exp (LSE).
    
    Args:
        Q: [B, H, S, D] dtype bfloat16
        K: [B, H, S, D] dtype bfloat16
        V: [B, H, S, D] dtype bfloat16
        O: [B, H, S, D] preallocated destination tensor dtype bfloat16
        LSE: [B, H, S] preallocated destination tensor dtype float32
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    sm_scale = 1.0 / math.sqrt(D)
    
    if S > 0:
        grid = lambda META: (triton.cdiv(S, META['BLOCK_M']), B * H, 1)
        _attn_fwd_kernel[grid](
            Q, K, V, sm_scale,
            O, LSE,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            LSE.stride(0), LSE.stride(1), LSE.stride(2),
            S, H,
            BLOCK_DMODEL=D,
        )