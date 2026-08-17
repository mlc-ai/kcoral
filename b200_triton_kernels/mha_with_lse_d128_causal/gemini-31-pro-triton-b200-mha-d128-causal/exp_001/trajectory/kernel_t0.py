import torch
import triton
import triton.language as tl
import math

@triton.jit
def _attn_fwd_kernel(
    Q, K, V, sm_scale,
    O, LSE,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_km, stride_kk,
    stride_vz, stride_vh, stride_vm, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    stride_lsez, stride_lseh, stride_lsem,
    S,
    H: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    # Grid is mapped as (B * H, cdiv(S, BLOCK_M), 1)
    off_hz = tl.program_id(0)
    off_z = off_hz // H
    off_h = off_hz % H
    
    start_m = tl.program_id(1) * BLOCK_M
    
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_D)
    
    # Pointers
    q_ptrs = Q + off_z * stride_qz + off_h * stride_qh + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    o_ptrs = O + off_z * stride_oz + off_h * stride_oh + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
    lse_ptrs = LSE + off_z * stride_lsez + off_h * stride_lseh + offs_m * stride_lsem
    
    # Initial accumulator state
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    mask_m = offs_m < S
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    
    # Causal attention boundary
    end_n = tl.minimum(S, start_m + BLOCK_M)
    
    for start_n in range(0, end_n, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        
        # Load K transposed logically by loading [BLOCK_N, BLOCK_D] and applying .T
        k_ptrs = K + off_z * stride_kz + off_h * stride_kh + offs_n[:, None] * stride_km + offs_k[None, :] * stride_kk
        k = tl.load(k_ptrs, mask=offs_n[:, None] < S, other=0.0)
        
        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        qk = qk * sm_scale
        
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        valid_mask = causal_mask & (offs_n[None, :] < S)
        qk = tl.where(valid_mask, qk, float("-inf"))
        
        # Max update
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        
        # Rescaling factors
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        
        # Load V
        v_ptrs = V + off_z * stride_vz + off_h * stride_vh + offs_n[:, None] * stride_vm + offs_k[None, :] * stride_vk
        v = tl.load(v_ptrs, mask=offs_n[:, None] < S, other=0.0)
        
        # Cast P and accumulate
        p_bf16 = p.to(tl.bfloat16)
        acc = tl.dot(p_bf16, v, acc=acc, out_dtype=tl.float32)
        
        m_i = m_ij

    # Write output
    out = acc / l_i[:, None]
    out = out.to(tl.bfloat16)
    tl.store(o_ptrs, out, mask=mask_m[:, None])
    
    # Write natural-log sum-exp
    lse = m_i + tl.log(l_i)
    tl.store(lse_ptrs, lse, mask=mask_m)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    if S == 0:
        return
        
    sm_scale = 1.0 / math.sqrt(D)
    
    grid = (B * H, triton.cdiv(S, 128), 1)
    
    _attn_fwd_kernel[grid](
        Q, K, V, sm_scale,
        O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        S,
        H=H,
        BLOCK_M=128,
        BLOCK_N=64,
        BLOCK_D=128,
        num_warps=4,
        num_stages=3
    )