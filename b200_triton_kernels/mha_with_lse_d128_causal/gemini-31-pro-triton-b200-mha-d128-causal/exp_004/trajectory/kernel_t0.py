import torch
import triton
import triton.language as tl
import math

@triton.jit
def _mha_fwd_kernel(
    Q, K, V, O, LSE,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    S, sm_scale,
    D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    batch_id = tl.program_id(1)
    head_id = tl.program_id(2)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    
    q_base = Q + batch_id * stride_qb + head_id * stride_qh
    k_base = K + batch_id * stride_kb + head_id * stride_kh
    v_base = V + batch_id * stride_vb + head_id * stride_vh
    
    q_ptrs = q_base + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    k_ptrs = k_base + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = v_base + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
    
    m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    
    q_mask = offs_m[:, None] < S
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)
    
    # Fold natural-to-base-2 conversion into Q for faster tl.math.exp2 execution
    RCP_LN2 = 1.4426950408889634
    q_scaled = (q * sm_scale * RCP_LN2).to(q.dtype)
    
    # Causal sequence mask bound
    end_n = tl.minimum(S, (start_m + 1) * BLOCK_M)
    num_n_blocks = tl.cdiv(end_n, BLOCK_N)
    
    for block_n_id in range(num_n_blocks):
        start_n = block_n_id * BLOCK_N
        k_mask = (start_n + offs_n[:, None]) < end_n
        v_mask = (start_n + offs_n[:, None]) < end_n
        
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)
        
        qk = tl.dot(q_scaled, tl.trans(k))
        
        n_indices = start_n + offs_n[None, :]
        causal_mask = offs_m[:, None] >= n_indices
        valid_score = causal_mask & (offs_m[:, None] < S) & (n_indices < S)
        
        qk = tl.where(valid_score, qk, -float('inf'))
        
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        safe_m_ij = tl.where(m_ij == -float('inf'), 0.0, m_ij)
        
        alpha = tl.math.exp2(m_i - safe_m_ij)
        p = tl.math.exp2(qk - safe_m_ij[:, None])
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(q.dtype), v, acc)
        m_i = m_ij
        
        k_ptrs += BLOCK_N * stride_ks
        v_ptrs += BLOCK_N * stride_vs
        
    safe_l_i = tl.where(l_i == 0.0, 1.0, l_i)
    output = acc / safe_l_i[:, None]
    
    # Convert base-2 logsumexp back to base-e natural log
    LN2 = 0.6931471805599453
    lse = tl.where(l_i == 0.0, -float('inf'), (m_i + tl.math.log2(safe_l_i)) * LN2)
    
    o_base = O + batch_id * stride_ob + head_id * stride_oh
    o_ptrs = o_base + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(o_ptrs, output.to(O.dtype.element_ty), mask=q_mask)
    
    lse_base = LSE + batch_id * stride_lseb + head_id * stride_lseh
    lse_ptrs = lse_base + offs_m * stride_lses
    tl.store(lse_ptrs, lse, mask=(offs_m < S))

def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    sm_scale = 1.0 / math.sqrt(D)
    
    BLOCK_M = 128
    BLOCK_N = 64
    
    grid = (triton.cdiv(S, BLOCK_M), B, H)
    
    _mha_fwd_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        S, sm_scale,
        D=128, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=4, num_stages=3
    )