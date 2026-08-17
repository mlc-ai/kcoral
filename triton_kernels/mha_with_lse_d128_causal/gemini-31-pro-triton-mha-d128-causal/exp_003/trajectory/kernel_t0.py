import math
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=4),
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
    H, S,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    off_b = off_hz // H
    off_h = off_hz % H
    
    q_base = Q + off_b * stride_qb + off_h * stride_qh
    k_base = K + off_b * stride_kb + off_h * stride_kh
    v_base = V + off_b * stride_vb + off_h * stride_vh
    o_base = O + off_b * stride_ob + off_h * stride_oh
    lse_base = LSE + off_b * stride_lseb + off_h * stride_lseh
    
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    
    q_ptrs = q_base + (offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd)
    q_mask = offs_m < S
    q = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)
    
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    
    k_ptrs = k_base + (offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd)
    v_ptrs = v_base + (offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd)
    
    hi = (start_m + 1) * BLOCK_M
    if hi > S:
        hi = S
        
    num_steps = tl.cdiv(hi, BLOCK_N)
    for step in range(0, num_steps):
        start_n = step * BLOCK_N
        k_offs_n = start_n + offs_n
        k_mask = k_offs_n < S
        
        k = tl.load(k_ptrs, mask=k_mask[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0)
        
        qk = tl.dot(q, k.T)
        qk = qk * sm_scale
        
        # Causal mask: strictly limit attending to future tokens.
        is_valid = k_mask[None, :] & (offs_m[:, None] >= k_offs_n[None, :])
        qk = tl.where(is_valid, qk, float("-inf"))
        
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        
        alpha = tl.exp(m_i - m_ij)
        acc = acc * alpha[:, None]
        
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        
        p = p.to(tl.bfloat16)
        acc = tl.dot(p, v, acc)
        
        k_ptrs += BLOCK_N * stride_ks
        v_ptrs += BLOCK_N * stride_vs
        
    acc = acc / l_i[:, None]
    lse = m_i + tl.log(l_i)
    
    o_ptrs = o_base + (offs_m[:, None] * stride_os + offs_d[None, :] * stride_od)
    lse_ptrs = lse_base + offs_m * stride_lses
    
    acc_out = acc.to(tl.bfloat16)
    
    tl.store(o_ptrs, acc_out, mask=q_mask[:, None])
    tl.store(lse_ptrs, lse, mask=q_mask)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    sm_scale = 1.0 / math.sqrt(D)
    
    if S == 0:
        return
        
    grid = lambda META: (triton.cdiv(S, META["BLOCK_M"]), B * H)
    
    _attn_fwd_kernel[grid](
        Q, K, V, sm_scale,
        O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        H, S,
        D=128,
    )