import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=8, num_stages=4),
    ],
    key=['S'],
)
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
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_z = tl.program_id(1)
    pid_h = tl.program_id(2)

    # Early exit if the entire block is out of sequence bounds
    if pid_m * BLOCK_M >= S:
        return

    q_offset = pid_z * stride_qz + pid_h * stride_qh
    k_offset = pid_z * stride_kz + pid_h * stride_kh
    v_offset = pid_z * stride_vz + pid_h * stride_vh
    o_offset = pid_z * stride_oz + pid_h * stride_oh
    lse_offset = pid_z * stride_lsez + pid_h * stride_lseh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_ptrs = K + k_offset + offs_n[:, None] * stride_km + offs_d[None, :] * stride_kk
    v_ptrs = V + v_offset + offs_n[:, None] * stride_vm + offs_d[None, :] * stride_vk

    # Load Q block once
    mask_m = offs_m < S
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # 1. Unmasked Full Blocks
    # For start_n < pid_m * BLOCK_M, all m >= n is guaranteed, eliminating the need for causal masks.
    # Since we early exit if pid_m * BLOCK_M >= S, we also know that start_n + BLOCK_N <= S.
    # Thus, both causal and sequence bound masks can be entirely skipped here for peak performance.
    end_n_full = pid_m * BLOCK_M
    
    for start_n in range(0, end_n_full, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        
        k = tl.load(k_ptrs + start_n * stride_km)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, k.T, qk)
        qk = qk * sm_scale
        
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])
        l_i_new = alpha * l_i + tl.sum(p, 1)
        
        v = tl.load(v_ptrs + start_n * stride_vm)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        
        m_i = m_i_new
        l_i = l_i_new

    # 2. Masked Causal Blocks
    # For start_n >= pid_m * BLOCK_M, we evaluate the precise causal boundary and potential sequence tail.
    end_n_causal = tl.minimum((pid_m + 1) * BLOCK_M, S)
    
    for start_n in range(end_n_full, end_n_causal, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n_curr = start_n + offs_n
        mask_n = offs_n_curr < S
        
        k = tl.load(k_ptrs + start_n * stride_km, mask=mask_n[:, None], other=0.0)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, k.T, qk)
        qk = qk * sm_scale
        
        mask = offs_m[:, None] >= offs_n_curr[None, :]
        qk = tl.where(mask, qk, float("-inf"))
        
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])
        l_i_new = alpha * l_i + tl.sum(p, 1)
        
        v = tl.load(v_ptrs + start_n * stride_vm, mask=mask_n[:, None], other=0.0)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        
        m_i = m_i_new
        l_i = l_i_new

    # Final Epilogue
    acc = acc / l_i[:, None]
    lse = m_i + tl.log(l_i)
    
    # Store Attention Output
    o_ptrs = O + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(tl.bfloat16), mask=mask_m[:, None])
    
    # Store Log-Sum-Exp
    lse_ptrs = LSE + lse_offset + offs_m * stride_lsem
    tl.store(lse_ptrs, lse, mask=mask_m)

def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    sm_scale = 1.0 / (D ** 0.5)

    grid = lambda META: (
        triton.cdiv(S, META['BLOCK_M']),
        B,
        H,
    )
    
    _attn_fwd_kernel[grid](
        Q, K, V, sm_scale,
        O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        S,
        BLOCK_D=128,
    )