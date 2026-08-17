import torch
import triton
import triton.language as tl
import math

def pre_hook(kwargs):
    # Determine at compile time whether sequence length leaves partial tiles
    kwargs['EVEN_M'] = (kwargs['N_CTX'] % kwargs['BLOCK_M'] == 0)
    kwargs['EVEN_N'] = (kwargs['N_CTX'] % kwargs['BLOCK_N'] == 0)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'NUM_STAGES': 3, 'WARP_SPECIALIZE': True}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'NUM_STAGES': 3, 'WARP_SPECIALIZE': False}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'NUM_STAGES': 4, 'WARP_SPECIALIZE': True}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'NUM_STAGES': 4, 'WARP_SPECIALIZE': False}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'NUM_STAGES': 4, 'WARP_SPECIALIZE': True}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'NUM_STAGES': 4, 'WARP_SPECIALIZE': False}, num_stages=4, num_warps=4),
    ],
    key=['N_CTX'],
    pre_hook=pre_hook,
)
@triton.jit
def _attn_fwd_kernel(
    Q, K, V, sm_scale, O, LSE,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_on,
    stride_lsez, stride_lseh, stride_lsem,
    H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    EVEN_M: tl.constexpr, EVEN_N: tl.constexpr,
    NUM_STAGES: tl.constexpr, WARP_SPECIALIZE: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    off_z = off_hz // H
    off_h = off_hz % H
    
    # Base offsets for the batch and head
    q_offset = off_z * stride_qz + off_h * stride_qh
    k_offset = off_z * stride_kz + off_h * stride_kh
    v_offset = off_z * stride_vz + off_h * stride_vh
    o_offset = off_z * stride_oz + off_h * stride_oh
    lse_offset = off_z * stride_lsez + off_h * stride_lseh
    
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    mask_m = offs_m < N_CTX
    
    # Initialize pointers to the first blocks along sequence (S) dimension
    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_ptrs = K + k_offset + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    v_ptrs = V + v_offset + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
    
    # Load query with compile-time branch-free optimization
    if EVEN_M:
        q = tl.load(q_ptrs)
    else:
        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    
    # Setup running FlashAttention accumulators
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    
    # Process sequential key and value blocks
    for start_n_idx in tl.range(0, tl.cdiv(N_CTX, BLOCK_N), num_stages=NUM_STAGES, warp_specialize=WARP_SPECIALIZE):
        if EVEN_N:
            k = tl.load(k_ptrs)
            v = tl.load(v_ptrs)
        else:
            start_n = start_n_idx * BLOCK_N
            offs_n_curr = start_n + offs_n
            mask_n = offs_n_curr < N_CTX
            k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
            v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
            
        # QK^T inner product
        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        qk = qk * sm_scale
        
        # Mask out-of-bounds keys (queries mask themselves implicitly on load/store boundaries)
        if not EVEN_N:
            qk = tl.where(mask_n[None, :], qk, float("-inf"))
            
        # FlashAttention formulation running stats
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        
        # Accumulate value outputs
        acc = acc * alpha[:, None]
        p = p.to(tl.bfloat16)
        acc = tl.dot(p, v, acc, out_dtype=tl.float32)
        
        m_i = m_ij
        
        # Advance sequence pointers uniformly avoiding vector logic inside loop
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn
        
    # Scale accumulated outputs and calculate LSE correctly for out-of-bounds padded elements
    acc = acc / l_i[:, None]
    lse = m_i + tl.log(l_i)
    
    # Resolve store pointers
    o_ptrs = O + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on
    lse_ptrs = LSE + lse_offset + offs_m * stride_lsem
    
    # Write back
    if EVEN_M:
        tl.store(o_ptrs, acc.to(tl.bfloat16))
        tl.store(lse_ptrs, lse)
    else:
        tl.store(o_ptrs, acc.to(tl.bfloat16), mask=mask_m[:, None])
        tl.store(lse_ptrs, lse, mask=mask_m)


def run(Q, K, V, O, LSE):
    """
    Computes Non-Causal Multi-Head Attention forward.
    Inputs/O shapes: (B, H, S, D) in bfloat16.
    LSE shape: (B, H, S) in float32.
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    if S == 0:
        return
    
    # Multiplier constant
    sm_scale = 1.0 / math.sqrt(D)
    
    # Mapping configuration
    grid = lambda META: (
        triton.cdiv(S, META['BLOCK_M']),
        B * H,
    )
    
    # Launch
    _attn_fwd_kernel[grid](
        Q, K, V, sm_scale, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        H, S,
        BLOCK_DMODEL=D,
        EVEN_M=False, EVEN_N=False,  # Replaced natively inside autotuner's pre_hook based on config shape mapping
    )