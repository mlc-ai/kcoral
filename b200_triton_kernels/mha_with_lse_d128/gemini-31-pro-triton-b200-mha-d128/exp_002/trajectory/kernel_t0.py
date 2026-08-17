import torch
import triton
import triton.language as tl
import math

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
    ],
    key=['N_CTX'],
)
@triton.jit
def _attn_fwd_kernel(
    Q, K, V, sm_scale,
    O, LSE,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    stride_lsez, stride_lseh, stride_lsem,
    H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    # Extract batch (Z) and head (H) coordinates
    off_z = off_hz // H
    off_h = off_hz % H
    
    q_offset = off_z * stride_qz + off_h * stride_qh
    k_offset = off_z * stride_kz + off_h * stride_kh
    v_offset = off_z * stride_vz + off_h * stride_vh
    o_offset = off_z * stride_oz + off_h * stride_oh
    lse_offset = off_z * stride_lsez + off_h * stride_lseh
    
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    mask_m = offs_m < N_CTX
    
    # Load Q
    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    
    # Initialize variables 
    # Use 0.0 for completely out-of-bounds rows to avoid -inf - (-inf) resulting in NaN
    m_i = tl.where(mask_m, float("-inf"), 0.0)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    
    # Process blocks over sequence (K, V)
    for start_n in range(0, N_CTX, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n_curr = start_n + offs_n
        
        mask_n = offs_n_curr < N_CTX
        
        k_ptrs = K + k_offset + offs_n_curr[:, None] * stride_kn + offs_d[None, :] * stride_kk
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        
        v_ptrs = V + v_offset + offs_n_curr[:, None] * stride_vk + offs_d[None, :] * stride_vn
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        
        # Compute QK^T scaled dot product
        qk = tl.dot(q, tl.trans(k))
        qk = qk * sm_scale
        
        # Mask out out-of-bound keys
        mask_qk = mask_m[:, None] & mask_n[None, :]
        qk = tl.where(mask_qk, qk, float("-inf"))
        
        # Update running max and scaling factor (FlashAttention formulation)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        
        # Update accumulator
        acc = acc * alpha[:, None]
        p = p.to(tl.bfloat16)
        acc = tl.dot(p, v, acc)
        
        m_i = m_ij
        
    # Finalize normalization for valid rows
    acc = acc / l_i[:, None]
    
    # Natural log-sum-exp
    lse = m_i + tl.log(l_i)
    
    # Write back output O
    o_ptrs = O + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on
    tl.store(o_ptrs, acc.to(tl.bfloat16), mask=mask_m[:, None])
    
    # Write back output LSE
    lse_ptrs = LSE + lse_offset + offs_m * stride_lsem
    tl.store(lse_ptrs, lse, mask=mask_m)


def run(Q, K, V, O, LSE):
    """
    Computes Non-Causal Multi-Head Attention forward.
    Q, K, V shapes: (B, H, S, D) in bfloat16.
    O shape: (B, H, S, D) in bfloat16.
    LSE shape: (B, H, S) in float32.
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    if S == 0:
        return
    
    # 1/sqrt(D) scale
    sm_scale = 1.0 / math.sqrt(D)
    
    # Config grid
    grid = lambda META: (
        triton.cdiv(S, META['BLOCK_M']),
        B * H,
    )
    
    # Kernel Launch
    _attn_fwd_kernel[grid](
        Q, K, V, sm_scale,
        O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        H, S,
        BLOCK_DMODEL=D,
    )