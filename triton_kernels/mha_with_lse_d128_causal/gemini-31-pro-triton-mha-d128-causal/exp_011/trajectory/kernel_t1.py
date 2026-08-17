import torch
import triton
import triton.language as tl
import math

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=3),
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
    S, sm_scale,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    start_m = pid_m * BLOCK_M
    if start_m >= S:
        return
        
    batch_id = pid_bh // H
    head_id = pid_bh % H
    
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    
    # Pointers to current blocks
    q_ptrs = Q + batch_id * stride_qb + head_id * stride_qh + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    k_ptrs = K + batch_id * stride_kb + head_id * stride_kh + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = V + batch_id * stride_vb + head_id * stride_vh + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
    
    # Load query block
    mask_m = offs_m < S
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    
    # Initialize accumulator and states
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    
    # Compute loop bounds
    # For a causal mask, the maximum valid key index for query i is i.
    # The max query index in this block is start_m + BLOCK_M - 1.
    max_k_len = tl.minimum(start_m + BLOCK_M, S)
    num_n_blocks = tl.cdiv(max_k_len, BLOCK_N)
    
    # Full blocks where start_n + BLOCK_N - 1 <= start_m, so offs_m >= offs_n_cur is always True
    num_full_blocks = tl.minimum(start_m // BLOCK_N, num_n_blocks)
    
    # 1. Iterate over full blocks where no causal or sequence bounds masking is required
    for start_n_idx in range(0, num_full_blocks):
        k = tl.load(k_ptrs)
        qk = tl.dot(q, k.T) * sm_scale
        
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])
        
        v = tl.load(v_ptrs)
        
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        
        m_i = m_i_new
        k_ptrs += BLOCK_N * stride_ks
        v_ptrs += BLOCK_N * stride_vs

    # 2. Iterate over partial blocks where we must check causal mask and/or sequence boundary
    for start_n_idx in range(num_full_blocks, num_n_blocks):
        start_n = start_n_idx * BLOCK_N
        offs_n_cur = start_n + offs_n
        
        mask_n = offs_n_cur < S
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        qk = tl.dot(q, k.T) * sm_scale
        
        # Apply causal mask and sequence boundary mask
        is_valid = (offs_m[:, None] >= offs_n_cur[None, :]) & mask_n[None, :]
        qk = tl.where(is_valid, qk, float('-inf'))
        
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])
        
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        
        m_i = m_i_new
        k_ptrs += BLOCK_N * stride_ks
        v_ptrs += BLOCK_N * stride_vs

    # Epilogue: scale and compute Log-Sum-Exp
    acc = acc / l_i[:, None]
    lse = m_i + tl.log(l_i)
    
    # Store outputs
    o_ptrs = O + batch_id * stride_ob + head_id * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(tl.bfloat16), mask=mask_m[:, None])
    
    lse_ptrs = LSE + batch_id * stride_lseb + head_id * stride_lseh + offs_m * stride_lses
    tl.store(lse_ptrs, lse, mask=mask_m)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    if S == 0:
        return

    grid = lambda META: (
        triton.cdiv(S, META['BLOCK_M']),
        B * H,
        1
    )
    
    sm_scale = 1.0 / math.sqrt(D)
    
    _fwd_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        S, sm_scale,
        H=H, D=D
    )