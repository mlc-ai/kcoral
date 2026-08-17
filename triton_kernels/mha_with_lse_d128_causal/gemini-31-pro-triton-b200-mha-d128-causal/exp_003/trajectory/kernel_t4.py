import torch
import triton
import triton.language as tl
import math

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "PIPELINE_STAGES": 3}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "PIPELINE_STAGES": 4}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "PIPELINE_STAGES": 3}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "PIPELINE_STAGES": 4}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "PIPELINE_STAGES": 3}, num_warps=4, num_stages=3),
    ],
    key=["S"],
)
@triton.jit
def _attn_fwd_kernel(
    Q, K, V, O, LSE,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    sm_scale,
    B, H, S,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
):
    tl.static_assert(BLOCK_M % BLOCK_N == 0, "BLOCK_M must be a multiple of BLOCK_N")
    
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    # Base pointers
    Q_ptr = Q + pid_b * stride_qb + pid_h * stride_qh
    K_ptr = K + pid_b * stride_kb + pid_h * stride_kh
    V_ptr = V + pid_b * stride_vb + pid_h * stride_vh
    O_ptr = O + pid_b * stride_ob + pid_h * stride_oh
    LSE_ptr = LSE + pid_b * stride_lseb + pid_h * stride_lseh

    # Q block pointer mapping
    Q_ptrs = Q_ptr + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    q = tl.load(Q_ptrs, mask=offs_m[:, None] < S, other=0.0)

    # Pre-scale Q with base-2 scaling factor enabling hardware exp2/log2 execution
    LOG2E = 1.4426950408889634
    scale = sm_scale * LOG2E
    q = (q * scale).to(tl.bfloat16)

    # K, V block initial pointer setups
    K_ptrs = K_ptr + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    V_ptrs = V_ptr + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd

    m_start = pid_m * BLOCK_M
    
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    
    # 1. Fully Unmasked Loop (Strictly Below Causal Diagonal)
    # Allows Triton compiler to perfectly streamline TMA / async execution unbounded
    for start_n in tl.range(0, m_start, BLOCK_N, num_stages=PIPELINE_STAGES):
        k = tl.load(K_ptrs)
        v = tl.load(V_ptrs)
        
        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        
        m_ij = tl.max(qk, 1)
        m_new = tl.maximum(m_i, m_ij)
        
        alpha = tl.exp2(m_i - m_new)
        beta = tl.exp2(qk - m_new[:, None])
        
        acc = acc * alpha[:, None]
        
        beta_bf16 = beta.to(tl.bfloat16)
        acc += tl.dot(beta_bf16, v, out_dtype=tl.float32)
        
        l_i = l_i * alpha + tl.sum(beta, 1)
        m_i = m_new
        
        K_ptrs += BLOCK_N * stride_ks
        V_ptrs += BLOCK_N * stride_vs

    # 2. Diagonal Loop (Partially Masked by Causal Constraint and Boundary Length)
    diag_end = tl.minimum(S, m_start + BLOCK_M)
    for start_n in range(m_start, diag_end, BLOCK_N):
        offs_n_curr = start_n + offs_n
        k_mask = offs_n_curr[:, None] < S
        v_mask = offs_n_curr[:, None] < S
        
        k = tl.load(K_ptrs, mask=k_mask, other=0.0)
        v = tl.load(V_ptrs, mask=v_mask, other=0.0)
        
        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        
        causal_mask = offs_m[:, None] >= offs_n_curr[None, :]
        qk = tl.where(causal_mask, qk, float("-inf"))
        
        m_ij = tl.max(qk, 1)
        m_new = tl.maximum(m_i, m_ij)
        
        alpha = tl.exp2(m_i - m_new)
        beta = tl.exp2(qk - m_new[:, None])
        
        acc = acc * alpha[:, None]
        
        beta_bf16 = beta.to(tl.bfloat16)
        acc += tl.dot(beta_bf16, v, out_dtype=tl.float32)
        
        l_i = l_i * alpha + tl.sum(beta, 1)
        m_i = m_new
        
        K_ptrs += BLOCK_N * stride_ks
        V_ptrs += BLOCK_N * stride_vs

    # Final Softmax Projection
    out = acc / l_i[:, None]
    
    O_ptrs = O_ptr + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(O_ptrs, out.to(tl.bfloat16), mask=offs_m[:, None] < S)
    
    # Mathematical conversion from base-2 log back to natural log for LSE
    LSE_ptrs = LSE_ptr + offs_m * stride_lses
    LN2 = 0.6931471805599453
    lse_val = (m_i + tl.log2(l_i)) * LN2
    tl.store(LSE_ptrs, lse_val, mask=offs_m < S)


def run(Q, K, V, O, LSE):
    """
    Computes causal scaled dot product attention and its log-sum-exp (LSE).
    
    Args:
        Q: Query tensor of shape (B, H, S, D) and dtype bfloat16.
        K: Key tensor of shape (B, H, S, D) and dtype bfloat16.
        V: Value tensor of shape (B, H, S, D) and dtype bfloat16.
        O: Preallocated output tensor (B, H, S, D) for the attention output, dtype bfloat16.
        LSE: Preallocated output tensor (B, H, S) for the log-sum-exp, dtype float32.
    """
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    sm_scale = 1.0 / math.sqrt(D)

    # Grid mapping: (Sequence Blocks, Batch * Heads)
    # The layout inherently processes matching heads sequentially. Because dispatch resolves continuously,
    # threads belonging to the same head map together to contiguous SMs heavily re-using their collective L2 fetches!
    grid = lambda META: (triton.cdiv(S, META["BLOCK_M"]), B * H)
    
    _attn_fwd_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        sm_scale,
        B, H, S,
        D,
    )