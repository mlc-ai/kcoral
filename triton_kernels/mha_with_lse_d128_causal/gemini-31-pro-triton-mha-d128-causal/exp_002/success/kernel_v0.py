import torch
import triton
import triton.language as tl
import math

def get_configs():
    return [
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
    ]

@triton.autotune(
    configs=get_configs(),
    key=['S']
)
@triton.jit
def _attn_fwd_kernel(
    Q, K, V, sm_scale, O, LSE,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    B, H, S,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    batch_idx = off_hz // H
    head_idx = off_hz % H
    
    q_offset = batch_idx * stride_qb + head_idx * stride_qh
    k_offset = batch_idx * stride_kb + head_idx * stride_kh
    v_offset = batch_idx * stride_vb + head_idx * stride_vh
    
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    
    # Q pointers
    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qs + tl.arange(0, BLOCK_D)[None, :] * stride_qd
    # K and V pointers initialized to start of sequence
    k_ptrs = K + k_offset + tl.arange(0, BLOCK_N)[:, None] * stride_ks + tl.arange(0, BLOCK_D)[None, :] * stride_kd
    v_ptrs = V + v_offset + tl.arange(0, BLOCK_N)[:, None] * stride_vs + tl.arange(0, BLOCK_D)[None, :] * stride_vd
    
    valid_m = offs_m[:, None] < S
    q = tl.load(q_ptrs, mask=valid_m, other=0.0)
    
    # Initialize online softmax statistics
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    # Calculate the upper bound for the key blocks based on causal restriction
    max_n = tl.minimum(S, (start_m + 1) * BLOCK_M)
    num_steps = tl.cdiv(max_n, BLOCK_N)
    
    for start_n_idx in range(0, num_steps):
        start_n = start_n_idx * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)
        
        valid_n = offs_n[:, None] < S
        k = tl.load(k_ptrs + start_n * stride_ks, mask=valid_n, other=0.0)
        
        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        qk = qk * sm_scale
        
        # Apply bounds and causal mask conditionally
        valid = valid_m & (offs_n[None, :] < S)
        if start_n + BLOCK_N > start_m * BLOCK_M:
            causal_mask = offs_m[:, None] >= offs_n[None, :]
            valid = valid & causal_mask
            
        qk = tl.where(valid, qk, float("-inf"))
        
        # Softmax and scale accumulator
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        
        alpha = tl.exp(m_i - m_ij)
        alpha = tl.where(m_ij == float("-inf"), 0.0, alpha)
        
        l_i = l_i * alpha + l_ij
        
        v = tl.load(v_ptrs + start_n * stride_vs, mask=valid_n, other=0.0)
        
        # WGMMA FP32 accumulation
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        
        m_i = m_ij
        
    acc = acc / l_i[:, None]
    lse = m_i + tl.log(l_i)
    
    # Write back output O
    o_offset = batch_idx * stride_ob + head_idx * stride_oh
    o_ptrs = O + o_offset + offs_m[:, None] * stride_os + tl.arange(0, BLOCK_D)[None, :] * stride_od
    tl.store(o_ptrs, acc.to(tl.bfloat16), mask=valid_m)
    
    # Write back output LSE
    lse_offset = batch_idx * stride_lseb + head_idx * stride_lseh
    lse_ptrs = LSE + lse_offset + offs_m * stride_lses
    tl.store(lse_ptrs, lse, mask=offs_m < S)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    grid = lambda META: (
        triton.cdiv(S, META['BLOCK_M']),
        B * H
    )
    
    sm_scale = 1.0 / math.sqrt(D)
    
    _attn_fwd_kernel[grid](
        Q, K, V, sm_scale, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        B, H, S,
        BLOCK_D=128
    )