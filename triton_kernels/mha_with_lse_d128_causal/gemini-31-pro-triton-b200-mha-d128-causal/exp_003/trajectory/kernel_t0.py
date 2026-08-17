import torch
import triton
import triton.language as tl
import math

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_stages=4, num_warps=4),
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
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    Q_ptr = Q + pid_b * stride_qb + pid_h * stride_qh
    K_ptr = K + pid_b * stride_kb + pid_h * stride_kh
    V_ptr = V + pid_b * stride_vb + pid_h * stride_vh
    O_ptr = O + pid_b * stride_ob + pid_h * stride_oh
    LSE_ptr = LSE + pid_b * stride_lseb + pid_h * stride_lseh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)

    Q_ptrs = Q_ptr + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    q = tl.load(Q_ptrs, mask=offs_m[:, None] < S, other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    # Determine maximum key sequence length to loop over based on causal constraint
    limit = tl.minimum(S, (pid_m + 1) * BLOCK_M)
    
    for start_n in range(0, limit, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        
        K_ptrs = K_ptr + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        V_ptrs = V_ptr + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
        
        k = tl.load(K_ptrs, mask=offs_n[:, None] < S, other=0.0)
        v = tl.load(V_ptrs, mask=offs_n[:, None] < S, other=0.0)
        
        # Q @ K^T
        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        qk = qk * sm_scale
        
        # Apply causal mask
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        qk = tl.where(causal_mask, qk, float("-inf"))
        
        # Online softmax components
        m_ij = tl.max(qk, 1)
        m_new = tl.maximum(m_i, m_ij)
        
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(qk - m_new[:, None])
        
        acc = acc * alpha[:, None]
        
        # Cast beta (activations) appropriately for dot product
        beta_bf16 = beta.to(tl.bfloat16)
        acc += tl.dot(beta_bf16, v, out_dtype=tl.float32)
        
        # Update running factors
        l_i = l_i * alpha + tl.sum(beta, 1)
        m_i = m_new

    # Final outputs
    out = acc / l_i[:, None]
    
    O_ptrs = O_ptr + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(O_ptrs, out.to(tl.bfloat16), mask=offs_m[:, None] < S)
    
    LSE_ptrs = LSE_ptr + offs_m * stride_lses
    lse_val = m_i + tl.log(l_i)
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

    grid = lambda META: (triton.cdiv(S, META["BLOCK_M"]), B * H)
    
    _attn_fwd_kernel[grid](
        Q, K, V, sm_scale,
        O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        B, H, S,
        D,
    )