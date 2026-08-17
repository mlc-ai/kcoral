import math
import torch
import triton
import triton.language as tl

# Assign allocator to hold device-created tensor-descriptor storage
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'WARP_SPECIALIZE': True, 'PIPELINE_STAGES': 3}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'WARP_SPECIALIZE': True, 'PIPELINE_STAGES': 4}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'WARP_SPECIALIZE': True, 'PIPELINE_STAGES': 3}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'WARP_SPECIALIZE': True, 'PIPELINE_STAGES': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'WARP_SPECIALIZE': True, 'PIPELINE_STAGES': 3}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'WARP_SPECIALIZE': True, 'PIPELINE_STAGES': 4}, num_stages=4, num_warps=4),
        # Un-specialized baselines
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'WARP_SPECIALIZE': False, 'PIPELINE_STAGES': 3}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'WARP_SPECIALIZE': False, 'PIPELINE_STAGES': 3}, num_stages=3, num_warps=4),
    ],
    key=['S'],
)
@triton.jit
def _attn_fwd_kernel(
    Q, K, V, O, LSE,
    sm_scale,
    stride_qz, stride_qh, stride_qs,
    stride_kz, stride_kh, stride_ks,
    stride_vz, stride_vh, stride_vs,
    stride_oz, stride_oh, stride_os,
    stride_lsez, stride_lseh, stride_lses,
    S, H,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr, PIPELINE_STAGES: tl.constexpr
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    batch = off_hz // H
    head = off_hz % H
    
    # Navigate to proper batch and head base offset bounds
    q_offset = batch * stride_qz + head * stride_qh
    k_offset = batch * stride_kz + head * stride_kh
    v_offset = batch * stride_vz + head * stride_vh
    o_offset = batch * stride_oz + head * stride_oh
    lse_offset = batch * stride_lsez + head * stride_lseh
    
    q_ptr = Q + q_offset
    k_ptr = K + k_offset
    v_ptr = V + v_offset
    o_ptr = O + o_offset
    
    # Map physical dimensions into Tensor Memory Accelerators descriptors.
    q_desc = tl.make_tensor_descriptor(
        q_ptr, shape=[S, BLOCK_DMODEL], strides=[stride_qs, 1],
        block_shape=[BLOCK_M, BLOCK_DMODEL], padding_option="zero"
    )
    k_desc = tl.make_tensor_descriptor(
        k_ptr, shape=[S, BLOCK_DMODEL], strides=[stride_ks, 1],
        block_shape=[BLOCK_N, BLOCK_DMODEL], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        v_ptr, shape=[S, BLOCK_DMODEL], strides=[stride_vs, 1],
        block_shape=[BLOCK_N, BLOCK_DMODEL], padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        o_ptr, shape=[S, BLOCK_DMODEL], strides=[stride_os, 1],
        block_shape=[BLOCK_M, BLOCK_DMODEL], padding_option="zero"
    )
    
    # Running statistics (initializations gracefully resolve logic inside the TMA null-padded boundaries)
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    
    # Load Initial Q Block
    q = q_desc.load([start_m * BLOCK_M, 0])
    
    offs_n_base = tl.arange(0, BLOCK_N)
    
    # Unroll and warp-specialize over the contextual chunk sizes
    for start_n_idx in tl.range(0, tl.cdiv(S, BLOCK_N), num_stages=PIPELINE_STAGES, warp_specialize=WARP_SPECIALIZE):
        start_n = start_n_idx * BLOCK_N
        k = k_desc.load([start_n, 0])
        
        # Q @ K.T (MMA 5th generation resolution)
        qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32)
        qk = qk * sm_scale
        
        # Restrict causality to true token blocks, suppressing softmax spikes beyond constraints
        offs_n = start_n + offs_n_base
        qk = tl.where(offs_n[None, :] < S, qk, float("-inf"))
        
        m_ij = tl.max(qk, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        
        # Softmax step
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])
        
        acc = acc * alpha[:, None]
        
        p_cast = p.to(tl.bfloat16)
        v = v_desc.load([start_n, 0])
        acc = tl.dot(p_cast, v, acc, out_dtype=tl.float32)
        
        # Commit running aggregations
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < S
    
    # Epilogue safeguards strictly against empty 0-valued lanes avoiding subsequent log(0) NaN propagation
    l_i_safe = tl.where(m_mask, l_i, 1.0)
    lse = m_i + tl.log(l_i_safe)
    
    # Direct pointer stores are sufficient for flat independent statistics mappings 
    lse_ptrs = LSE + lse_offset + offs_m * stride_lses
    tl.store(lse_ptrs, lse, mask=m_mask)
    
    acc = acc / l_i_safe[:, None]
    
    # Store aggregated and formatted outputs through TMA mechanisms
    o_desc.store([start_m * BLOCK_M, 0], acc.to(tl.bfloat16))


def run(Q, K, V, O, LSE):
    """
    Computes Standard Attention forward pass and Log-Sum-Exp statistics (LSE) purely in Triton.
    
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
            Q, K, V, O, LSE,
            sm_scale,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            LSE.stride(0), LSE.stride(1), LSE.stride(2),
            S, H,
            BLOCK_DMODEL=D,
        )