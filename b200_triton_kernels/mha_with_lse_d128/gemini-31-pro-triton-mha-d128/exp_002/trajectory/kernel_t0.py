import torch
import triton
import triton.language as tl
import math

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
    ],
    key=['S']
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
    B, H, S, D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    pid_b = pid_bh // H
    pid_h = pid_bh % H
    
    q_offset = pid_b * stride_qb + pid_h * stride_qh
    k_offset = pid_b * stride_kb + pid_h * stride_kh
    v_offset = pid_b * stride_vb + pid_h * stride_vh
    o_offset = pid_b * stride_ob + pid_h * stride_oh
    lse_offset = pid_b * stride_lseb + pid_h * stride_lseh
    
    Q_ptr = Q + q_offset
    K_ptr = K + k_offset
    V_ptr = V + v_offset
    O_ptr = O + o_offset
    LSE_ptr = LSE + lse_offset
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    
    mask_q = offs_m[:, None] < S
    mask_q_1d = offs_m < S
    
    q_ptrs = Q_ptr + (offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd)
    q = tl.load(q_ptrs, mask=mask_q, other=0.0)
    
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    
    for start_n in range(0, S, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K_ptr + (offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd)
        v_ptrs = V_ptr + (offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd)
        
        mask_k = offs_n[:, None] < S
        mask_v = offs_n[:, None] < S
        
        k = tl.load(k_ptrs, mask=mask_k, other=0.0)
        v = tl.load(v_ptrs, mask=mask_v, other=0.0)
        
        # Q @ K.T
        qk = tl.dot(q, k.T, out_dtype=tl.float32) * sm_scale
        
        # Mask out-of-bounds keys and queries
        mask_qk = mask_q_1d[:, None] & (offs_n[None, :] < S)
        qk = tl.where(mask_qk, qk, float("-inf"))
        
        m_ij = tl.max(qk, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        # Avoid NaN values for completely out-of-bounds queries 
        m_i_new = tl.where(mask_q_1d, m_i_new, 0.0)
        
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(qk - m_i_new[:, None])
        
        l_i_new = alpha * l_i + tl.sum(beta, 1)
        
        # Scale and accumulate
        p = beta.to(v.dtype)
        acc = acc * alpha[:, None] + tl.dot(p, v, out_dtype=tl.float32)
        
        m_i = m_i_new
        l_i = l_i_new
        
    acc = acc / l_i[:, None]
    O_val = acc.to(q.dtype)
    
    o_ptrs = O_ptr + (offs_m[:, None] * stride_os + offs_d[None, :] * stride_od)
    tl.store(o_ptrs, O_val, mask=mask_q)
    
    lse_ptrs = LSE_ptr + (offs_m * stride_lses)
    lse_val = m_i + tl.log(l_i)
    tl.store(lse_ptrs, lse_val, mask=mask_q_1d)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    
    if S == 0:
        return
        
    sm_scale = 1.0 / math.sqrt(D)
    
    grid = lambda META: (triton.cdiv(S, META['BLOCK_M']), B * H)
    
    _attn_fwd_kernel[grid](
        Q, K, V, sm_scale,
        O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        B, H, S, D=D,
    )