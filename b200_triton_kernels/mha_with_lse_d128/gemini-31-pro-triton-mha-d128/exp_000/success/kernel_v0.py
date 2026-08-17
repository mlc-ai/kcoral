import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=4),
    ],
    key=["S"],
)
@triton.jit
def _fwd_kernel(
    Q, K, V, O, LSE,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    S, H,
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    
    # Map the linear 1D batch-head index back to batch and head identifiers
    b = off_bh // H
    h = off_bh % H
    
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    
    # Base pointers for the current batch and head
    q_ptrs = Q + b * stride_qb + h * stride_qh + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    k_ptrs = K + b * stride_kb + h * stride_kh + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = V + b * stride_vb + h * stride_vh + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
    
    mask_q = (offs_m[:, None] < S) & (offs_d[None, :] < BLOCK_D)
    q = tl.load(q_ptrs, mask=mask_q, other=0.0)
    
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    for start_n in range(0, S, BLOCK_N):
        current_offs_n = start_n + offs_n
        mask_n = current_offs_n < S
        mask_kv = mask_n[:, None] & (offs_d[None, :] < BLOCK_D)
        
        # Load and compute Q @ K^T
        k = tl.load(k_ptrs, mask=mask_kv, other=0.0)
        qk = tl.dot(q, k.T) * sm_scale
        
        # Mask out-of-bounds keys for the current query block
        qk = tl.where(mask_n[None, :], qk, float('-inf'))
        
        # Update running max (m_i) and running denominator (l_i) using the Flash Attention formulation
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        
        alpha = tl.exp(m_i - m_ij)
        acc = acc * alpha[:, None]
        
        # Load V and aggregate
        v = tl.load(v_ptrs, mask=mask_kv, other=0.0)
        p_bfloat16 = p.to(tl.bfloat16)
        acc += tl.dot(p_bfloat16, v)
        
        # Commit running statistics
        m_i = m_ij
        l_i = l_i * alpha + l_ij
        
        # Advance pointers for K and V
        k_ptrs += BLOCK_N * stride_ks
        v_ptrs += BLOCK_N * stride_vs
        
    # Finalize attention scaling
    acc = acc / l_i[:, None]
    
    # LSE (LogSumExp) equals max + natural-log(sum) 
    lse = m_i + tl.log(l_i)
    
    # Pointers and masks to commit outputs safely 
    o_ptrs = O + b * stride_ob + h * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    lse_ptrs = LSE + b * stride_lseb + h * stride_lseh + offs_m * stride_lses
    
    out_mask = (offs_m[:, None] < S) & (offs_d[None, :] < BLOCK_D)
    tl.store(o_ptrs, acc.to(tl.bfloat16), mask=out_mask)
    tl.store(lse_ptrs, lse, mask=offs_m < S)


def run(Q, K, V, O, LSE):
    """
    Computes multi-head attention natively.
    Inputs:
        Q, K, V: [B, H, S, D] bfloat16 inputs 
    Outputs:
        O: [B, H, S, D] bfloat16 preallocated result
        LSE: [B, H, S] float32 preallocated result
    """
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    # 1.0 / sqrt(D) scale factor
    sm_scale = 1.0 / (D ** 0.5)
    
    # 2D Grid separating parallelization spaces (M-dimension tiles by batch*head items)
    grid = lambda META: (triton.cdiv(S, META["BLOCK_M"]), B * H)
    
    _fwd_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        S, H,
        sm_scale,
        BLOCK_D=128
    )