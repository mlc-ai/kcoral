import torch
import triton
import triton.language as tl


@triton.jit
def _mha_fwd_kernel(
    Q, K, V, sm_scale,
    O, LSE,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    B, H, S,
    max_n_blocks,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    off_b = off_hz // H
    off_h = off_hz % H
    
    # Block Offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    
    # Initialize Pointers
    q_ptrs = Q + off_b * stride_qb + off_h * stride_qh + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    k_ptrs = K + off_b * stride_kb + off_h * stride_kh + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = V + off_b * stride_vb + off_h * stride_vh + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
    
    # Load Query Block
    q_mask = (offs_m[:, None] < S) & (offs_d[None, :] < BLOCK_D)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)
    
    # Initialize Softmax Tracking & Output Accumulator
    is_valid_m = offs_m < S
    m_i = tl.where(is_valid_m, float("-inf"), 0.0)
    l_i = tl.where(is_valid_m, 0.0, 1.0)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    # Bounded Loop logic handling causal mask ranges natively without `break` statements.
    # Assumes BLOCK_M is evenly divisible by BLOCK_N
    limit_n = tl.minimum((start_m + 1) * (BLOCK_M // BLOCK_N), max_n_blocks)
    
    for start_n in range(0, limit_n):
        start_n_offs = start_n * BLOCK_N
        offs_n_curr = start_n_offs + offs_n
        
        # Load Key Block
        k_mask = (offs_n_curr[:, None] < S) & (offs_d[None, :] < BLOCK_D)
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)
        
        # Q @ K^T
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, tl.trans(k), qk)
        qk = qk * sm_scale
        
        # Masking limits
        causal_mask = offs_m[:, None] >= offs_n_curr[None, :]
        valid_mask = causal_mask & (offs_m[:, None] < S) & (offs_n_curr[None, :] < S)
        
        qk = tl.where(valid_mask, qk, float("-inf"))
        
        # Softmax Online Update Tracking
        m_ij = tl.max(qk, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])
        p = tl.where(valid_mask, p, 0.0)
        
        l_i_new = alpha * l_i + tl.sum(p, 1)
        
        acc = acc * alpha[:, None]
        
        # Load Value Block
        v_mask = (offs_n_curr[:, None] < S) & (offs_d[None, :] < BLOCK_D)
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)
        
        # P @ V
        p_bf16 = p.to(tl.bfloat16)
        acc = tl.dot(p_bf16, v, acc)
        
        # Step Tracking variables
        m_i = m_i_new
        l_i = l_i_new
        
        # Advance K/V Pointers for Next Block Iteration
        k_ptrs += BLOCK_N * stride_ks
        v_ptrs += BLOCK_N * stride_vs

    # Final Softmax Normalization and Log-Sum-Exp Representation Output
    acc = acc / l_i[:, None]
    lse = m_i + tl.log(l_i)
    
    # Store LSE Block
    lse_ptrs = LSE + off_b * stride_lseb + off_h * stride_lseh + offs_m * stride_lses
    tl.store(lse_ptrs, lse, mask=is_valid_m)
    
    # Store Output Block
    o_ptrs = O + off_b * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    o_mask = is_valid_m[:, None] & (offs_d[None, :] < BLOCK_D)
    tl.store(o_ptrs, acc.to(tl.bfloat16), mask=o_mask)


def run(Q, K, V, O, LSE):
    """
    Compute Causal Multi-Head Attention in BFloat16 returning O and LSE.
    Results are directly written onto output destination tensors.
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    
    # Edge Case Avoidance
    if S == 0:
        return
        
    sm_scale = 1.0 / (D ** 0.5)
    
    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_D = 128
    
    max_n_blocks = (S + BLOCK_N - 1) // BLOCK_N
    
    # Set thread/block limits to scale accordingly
    grid = (triton.cdiv(S, BLOCK_M), B * H, 1)
    
    _mha_fwd_kernel[grid](
        Q, K, V, sm_scale,
        O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        B, H, S,
        max_n_blocks,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=4, num_stages=3
    )