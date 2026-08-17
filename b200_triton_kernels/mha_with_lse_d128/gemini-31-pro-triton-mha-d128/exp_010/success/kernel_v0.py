import torch
import triton
import triton.language as tl
import math

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=4),
    ],
    key=["S"]
)
@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale,
    O, LSE,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    S,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    b_idx = pid_bh // H
    h_idx = pid_bh % H
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    
    q_ptrs = Q + b_idx * stride_qb + h_idx * stride_qh + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    k_ptrs = K + b_idx * stride_kb + h_idx * stride_kh + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = V + b_idx * stride_vb + h_idx * stride_vh + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
    
    mask_m = offs_m < S
    
    # Load Q and apply scaling
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    q = (q * sm_scale).to(Q.dtype.element_ty)
    
    # Initialize online softmax statistics and accumulator
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    
    n_blocks = tl.cdiv(S, BLOCK_N)
    for start_n in range(0, n_blocks):
        offs_n_curr = start_n * BLOCK_N + offs_n
        mask_n = offs_n_curr < S
        
        # Load K and V for the current block
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        
        # Q @ K.T (accumulates in FP32)
        qk = tl.dot(q, tl.trans(k))
        qk = tl.where(mask_n[None, :], qk, float('-inf'))
        
        # Update running maximum and compute scaled exponent
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        
        # Update normalization factors
        l_ij = tl.sum(p, 1)
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        
        # Scale previously accumulated values by alpha and compute P @ V
        acc = acc * alpha[:, None]
        p = p.to(V.dtype.element_ty)
        acc = tl.dot(p, v, acc)
        
        # Set running maximum to the updated maximum
        m_i = m_ij
        
        # Advance KV pointers to the next sequence block
        k_ptrs += BLOCK_N * stride_ks
        v_ptrs += BLOCK_N * stride_vs

    # Normalize final output and compute LogSumExp
    acc = acc / l_i[:, None]
    lse = m_i + tl.log(l_i)
    
    # Store results
    o_ptrs = O + b_idx * stride_ob + h_idx * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    lse_ptrs = LSE + b_idx * stride_lseb + h_idx * stride_lseh + offs_m * stride_lses
    
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=mask_m[:, None])
    tl.store(lse_ptrs, lse, mask=mask_m)

def run(Q, K, V, O, LSE):
    """
    Computes a batched multi-head attention forward pass dynamically scaling for varying sequence lengths.
    
    Inputs:
        Q, K, V: bfloat16 tensors of shape (B, H, S, D)
    Outputs:
        O: Preallocated bfloat16 output tensor of shape (B, H, S, D)
        LSE: Preallocated float32 LogSumExp tensor of shape (B, H, S)
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    sm_scale = 1.0 / math.sqrt(D)
    
    grid = lambda META: (
        triton.cdiv(S, META['BLOCK_M']),
        B * H
    )
    
    _fwd_kernel[grid](
        Q, K, V, sm_scale,
        O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        S,
        H=H,
        D=D,
    )