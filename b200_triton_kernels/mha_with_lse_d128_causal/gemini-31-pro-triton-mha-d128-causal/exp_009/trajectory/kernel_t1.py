import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=4),
    ],
    key=["S"],
)
@triton.jit
def _mha_fwd_kernel(
    Q, K, V, sm_scale,
    O, LSE,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    H, S,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    start_m = tl.program_id(0)
    
    # Early escape when the program handles completely out-of-bounds sequences
    if start_m * BLOCK_M >= S:
        return
        
    off_hz = tl.program_id(1)
    off_b = off_hz // H
    off_h = off_hz % H
    
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    
    # Establish Pointers
    q_ptrs = Q + off_b * stride_qb + off_h * stride_qh + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    k_ptrs = K + off_b * stride_kb + off_h * stride_kh + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = V + off_b * stride_vb + off_h * stride_vh + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
    
    # Query is safely loaded applying sequence boundaries once
    q = tl.load(q_ptrs, mask=(offs_m[:, None] < S), other=0.0)
    
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    # Strictly define maximum lengths and blocks resolving Causal interactions & sequence endpoints
    max_k_idx = tl.minimum(start_m * BLOCK_M + BLOCK_M, S)
    n_blocks = (max_k_idx + BLOCK_N - 1) // BLOCK_N
    
    limit_n = tl.minimum(start_m * BLOCK_M, S)
    n_unmasked_blocks = limit_n // BLOCK_N
    
    # ==========================
    # 1. Unmasked Sequence Loop
    # ==========================
    for i in range(0, n_unmasked_blocks):
        # All conditions strictly validate boundaries, disabling tl.where evaluations entirely.
        k = tl.load(k_ptrs)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, k.T, qk)
        qk = qk * sm_scale
        
        m_ij = tl.max(qk, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])
        
        l_i_new = alpha * l_i + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        
        v = tl.load(v_ptrs)
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        
        m_i = m_i_new
        l_i = l_i_new
        
        k_ptrs += BLOCK_N * stride_ks
        v_ptrs += BLOCK_N * stride_vs
        
    # ==========================
    # 2. Masked Sequence Loop
    # ==========================
    for i in range(n_unmasked_blocks, n_blocks):
        start_n = i * BLOCK_N
        offs_n_curr = start_n + offs_n
        
        k = tl.load(k_ptrs, mask=(offs_n_curr[:, None] < S), other=0.0)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, k.T, qk)
        qk = qk * sm_scale
        
        causal_mask = offs_m[:, None] >= offs_n_curr[None, :]
        valid_mask = causal_mask & (offs_m[:, None] < S) & (offs_n_curr[None, :] < S)
        qk = tl.where(valid_mask, qk, float("-inf"))
        
        m_ij = tl.max(qk, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])
        p = tl.where(valid_mask, p, 0.0)
        
        l_i_new = alpha * l_i + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        
        v = tl.load(v_ptrs, mask=(offs_n_curr[:, None] < S), other=0.0)
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        
        m_i = m_i_new
        l_i = l_i_new
        
        k_ptrs += BLOCK_N * stride_ks
        v_ptrs += BLOCK_N * stride_vs

    # Final Softmax Normalization
    acc = acc / l_i[:, None]
    lse = m_i + tl.log(l_i)
    
    is_valid_m = offs_m < S
    lse_ptrs = LSE + off_b * stride_lseb + off_h * stride_lseh + offs_m * stride_lses
    tl.store(lse_ptrs, lse, mask=is_valid_m)
    
    o_ptrs = O + off_b * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(tl.bfloat16), mask=is_valid_m[:, None])


def run(Q, K, V, O, LSE):
    """
    Compute Causal Multi-Head Attention forward returning `O` and LogSumExp (`LSE`).
    Results are exclusively deposited in preallocated output tensors inplace avoiding new allocations.
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    
    if S == 0:
        return
        
    sm_scale = 1.0 / (D ** 0.5)
    
    grid = lambda META: (triton.cdiv(S, META["BLOCK_M"]), B * H, 1)
    
    _mha_fwd_kernel[grid](
        Q, K, V, sm_scale,
        O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        H, S,
        BLOCK_D=128
    )