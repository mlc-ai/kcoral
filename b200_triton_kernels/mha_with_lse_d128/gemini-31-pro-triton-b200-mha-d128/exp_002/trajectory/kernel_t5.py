import torch
import triton
import triton.language as tl
import math

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
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
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_on,
    stride_lsez, stride_lseh, stride_lsem,
    H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    # Calculate batch (Z) and head (H) dimensions 
    off_z = off_hz // H
    off_h = off_hz % H
    
    # Compute base offsets
    q_offset = off_z * stride_qz + off_h * stride_qh
    k_offset = off_z * stride_kz + off_h * stride_kh
    v_offset = off_z * stride_vz + off_h * stride_vh
    o_offset = off_z * stride_oz + off_h * stride_oh
    lse_offset = off_z * stride_lsez + off_h * stride_lseh
    
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    # Intialize grid pointers. Note that K relies on standard row-major mappings since `tl.dot(Q, K.T)` natively shifts.
    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_ptrs = K + k_offset + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    v_ptrs = V + v_offset + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
    
    m_mask = offs_m < N_CTX
    q = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0)
    
    # Setup numerically stable FlashAttention accumulators
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    
    # Pipelined hardware loop over inner key/value contextual blocks
    # Employing `tl.range` over standard `range` is structurally required to assert compiler `num_stages` execution
    for start_n_idx in tl.range(0, tl.cdiv(N_CTX, BLOCK_N)):
        start_n = start_n_idx * BLOCK_N
        offs_n_curr = start_n + offs_n
        n_mask = offs_n_curr < N_CTX
        
        k = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)
        
        # Matrix multiplication Q @ K.T -> Accumulate to FP32
        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        qk = qk * sm_scale
        qk = tl.where(n_mask[None, :], qk, float("-inf"))
        
        # Standard stable Softmax max-shifting logic
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        
        # Multiply acc buffer and run reduction dot for final values
        acc = acc * alpha[:, None]
        p_bf16 = p.to(tl.bfloat16)
        acc = tl.dot(p_bf16, v, acc, out_dtype=tl.float32)
        
        m_i = m_ij
        
        # Uniformly traverse sliding pointers through the Sequence (S) elements
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn
        
    # Standardize final accumulation and resolve mathematically-exact Natural Log-Sum-Exp
    acc = acc / l_i[:, None]
    lse = m_i + tl.log(l_i)
    
    # Resolve and store final vectors out cleanly
    o_ptrs = O + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on
    tl.store(o_ptrs, acc.to(tl.bfloat16), mask=m_mask[:, None])
    
    lse_ptrs = LSE + lse_offset + offs_m * stride_lsem
    tl.store(lse_ptrs, lse, mask=m_mask)

def run(Q, K, V, O, LSE):
    """
    Computes Non-Causal Multi-Head Attention forward over preallocated Tensors.
    Inputs and Output (O) block shapes: (B, H, S, D) spanning in bfloat16.
    Output (LSE) vector lengths: (B, H, S) bound to float32 scalars.
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    if S == 0:
        return
    
    sm_scale = 1.0 / math.sqrt(D)
    
    # Construct an inherently swizzled execution grid that places M iteratively prior to heads
    grid = lambda META: (
        triton.cdiv(S, META['BLOCK_M']),
        B * H,
    )
    
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