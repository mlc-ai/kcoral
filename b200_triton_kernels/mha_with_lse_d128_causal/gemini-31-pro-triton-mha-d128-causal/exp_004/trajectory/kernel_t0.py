import math
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
    ],
    key=["S"],
)
@triton.jit
def _attn_fwd_kernel(
    Q, K, V, sm_scale,
    O, LSE,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    B, H, S,
    BLOCK_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    b_idx = pid_bh // H
    h_idx = pid_bh % H
    
    start_m = pid_m * BLOCK_M
    # Return early if this block is fully out of sequence bounds
    if start_m >= S:
        return
        
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    
    # Initialize pointers
    q_ptrs = Q + b_idx * stride_qb + h_idx * stride_qh + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    k_ptrs = K + b_idx * stride_kb + h_idx * stride_kh + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = V + b_idx * stride_vb + h_idx * stride_vh + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
    
    # Initialize running max, sum, and accumulator
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    # Load Query block
    q_mask = offs_m[:, None] < S
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)
    
    # For causal attention, queries only attend to keys up to their own index.
    # Therefore, we only need to iterate over key blocks up to min(S, start_m + BLOCK_M).
    end_n_limit = tl.minimum(S, start_m + BLOCK_M)
    num_n_blocks = tl.cdiv(end_n_limit, BLOCK_N)
    
    for start_n_idx in range(0, num_n_blocks):
        start_n = start_n_idx * BLOCK_N
        
        # Load Key and Value blocks
        k_mask = (start_n + offs_n[:, None]) < S
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)
        v_mask = (start_n + offs_n[:, None]) < S
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)
        
        # Compute dot product
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, k.T, acc=qk)
        qk = qk * sm_scale
        
        # Apply causal mask and sequence boundary mask
        valid_mask = (offs_m[:, None] >= (start_n + offs_n[None, :])) & ((start_n + offs_n[None, :]) < S)
        qk = tl.where(valid_mask, qk, float("-inf"))
        
        # Online softmax implementation
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        p = tl.where(valid_mask, p, 0.0)
        
        l_ij = tl.sum(p, axis=1)
        alpha = tl.exp(m_i - m_ij)
        
        # Update normalizer and accumulated values
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        
        p_bf16 = p.to(V.dtype.element_ty)
        acc = tl.dot(p_bf16, v, acc=acc)
        
        m_i = m_ij
        
        # Advance KV pointers
        k_ptrs += BLOCK_N * stride_ks
        v_ptrs += BLOCK_N * stride_vs

    # Normalize accumulated outputs
    acc = acc / l_i[:, None]
    
    # Compute Log-Sum-Exp (LSE) for the block
    lse = m_i + tl.log(l_i)
    
    # Store results to output pointers (guarding with sequence bounds)
    out_ptrs = O + b_idx * stride_ob + h_idx * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(out_ptrs, acc.to(O.dtype.element_ty), mask=q_mask)
    
    lse_ptrs = LSE + b_idx * stride_lseb + h_idx * stride_lseh + offs_m * stride_lses
    lse_mask = offs_m < S
    tl.store(lse_ptrs, lse, mask=lse_mask)

def run(Q, K, V, O, LSE):
    """
    Computes causal multi-head attention forward pass.
    
    Inputs:
        Q, K, V: bfloat16 tensors of shape [B, H, S, D]
    Outputs:
        O: bfloat16 tensor of shape [B, H, S, D]
        LSE: float32 tensor of shape [B, H, S]
    """
    B, H, S, D = Q.shape
    sm_scale = 1.0 / math.sqrt(D)
    
    torch.cuda.set_device(Q.device)
    
    # Ensure optimal 3D grid layout resolving queries per batch and head 
    grid = lambda META: (
        triton.cdiv(S, META["BLOCK_M"]),
        B * H,
        1
    )
    
    _attn_fwd_kernel[grid](
        Q, K, V, sm_scale,
        O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        B, H, S,
        BLOCK_D=128
    )