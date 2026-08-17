import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def mha_fwd_kernel_tma_host(
    q_desc, k_desc, v_desc, o_desc,
    LSE_ptr, stride_lseb, stride_lseh, stride_lses,
    B, H, S, sm_scale_log2,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN_S: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)
    
    start_m = pid_m * BLOCK_M
    
    # 1. 4D Bulk TMA Load directly mapped to shared memory
    q_4d = q_desc.load([pid_b, pid_h, start_m, 0])
    q = tl.reshape(q_4d, [BLOCK_M, 128])
    
    # 2. Scale Q once outside the K/V loop to eliminate ~16k floating-point instructions per step
    q = (q * sm_scale_log2).to(tl.bfloat16)
    
    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, 128], dtype=tl.float32)
    
    num_n_blocks = tl.cdiv(S, BLOCK_N)
    
    for start_n_idx in range(num_n_blocks):
        start_n = start_n_idx * BLOCK_N
        
        # TMA Hardware manages the loading pipeline directly
        k_4d = k_desc.load([pid_b, pid_h, start_n, 0])
        k = tl.reshape(k_4d, [BLOCK_N, 128])
        
        v_4d = v_desc.load([pid_b, pid_h, start_n, 0])
        v = tl.reshape(v_4d, [BLOCK_N, 128])
        
        # Trigger native Hopper WGMMA FP16/BF16 -> FP32 Accumulation
        qk = tl.dot(q, k.T)
        
        if not EVEN_S:
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask = offs_n[None, :] < S
            qk = tl.where(mask, qk, float("-inf"))
            
        m_i_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp2(m_i - m_i_new)
        p = tl.exp2(qk - m_i_new[:, None])
        l_i_new = alpha * l_i + tl.sum(p, axis=1)
        
        # FlashAttention Accumulator Scaling & Projection
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        
        m_i = m_i_new
        l_i = l_i_new
        
    # Scale inverse norm and store outputs
    l_i_safe = tl.where(l_i == 0.0, 1.0, l_i)
    out = acc * (1.0 / l_i_safe[:, None])
    
    out_4d = tl.reshape(out.to(tl.bfloat16), [1, 1, BLOCK_M, 128])
    o_desc.store([pid_b, pid_h, start_m, 0], out_4d)
    
    # Format running softmax scale back to natural log (Log Sum Exp Output requirement)
    lse = (m_i + tl.log2(l_i)) * 0.6931471805599453
    lse_offset = pid_b * stride_lseb + pid_h * stride_lseh + start_m * stride_lses
    offs_m = tl.arange(0, BLOCK_M)
    lse_ptrs = LSE_ptr + lse_offset + offs_m * stride_lses
    
    if EVEN_S:
        tl.store(lse_ptrs, lse)
    else:
        tl.store(lse_ptrs, lse, mask=(start_m + offs_m) < S)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=4, num_warps=4),
    ],
    key=['S']
)
@triton.jit
def mha_fwd_kernel_ptr(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    B, H, S, sm_scale_log2,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN_S: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    q_offset = pid_b * stride_qb + pid_h * stride_qh
    k_offset = pid_b * stride_kb + pid_h * stride_kh
    v_offset = pid_b * stride_vb + pid_h * stride_vh
    o_offset = pid_b * stride_ob + pid_h * stride_oh
    lse_offset = pid_b * stride_lseb + pid_h * stride_lseh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    q_ptrs = Q_ptr + q_offset + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    k_ptrs = K_ptr + k_offset + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = V_ptr + v_offset + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd

    if EVEN_S:
        q = tl.load(q_ptrs)
    else:
        q = tl.load(q_ptrs, mask=offs_m[:, None] < S, other=0.0)
        
    q = (q * sm_scale_log2).to(tl.bfloat16)

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    num_n_blocks = tl.cdiv(S, BLOCK_N)
    k_step = BLOCK_N * stride_ks
    v_step = BLOCK_N * stride_vs

    for start_n_idx in range(num_n_blocks):
        start_n = start_n_idx * BLOCK_N
        
        if EVEN_S:
            k = tl.load(k_ptrs)
            v = tl.load(v_ptrs)
        else:
            k_mask = (start_n + offs_n[:, None] < S)
            k = tl.load(k_ptrs, mask=k_mask, other=0.0)
            v = tl.load(v_ptrs, mask=k_mask, other=0.0)
        
        qk = tl.dot(q, tl.trans(k))
        
        if not EVEN_S:
            mask_n = start_n + offs_n < S
            qk = tl.where(mask_n[None, :], qk, float("-inf"))
            
        m_i_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp2(m_i - m_i_new)
        p = tl.exp2(qk - m_i_new[:, None])
        l_i_new = alpha * l_i + tl.sum(p, axis=1)
        
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        
        m_i = m_i_new
        l_i = l_i_new
        k_ptrs += k_step
        v_ptrs += v_step

    l_i_inv = 1.0 / tl.where(l_i == 0.0, 1.0, l_i)
    out = acc * l_i_inv[:, None]
    
    lse = (m_i + tl.log2(l_i)) * 0.6931471805599453
    
    o_ptrs = O_ptr + o_offset + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    lse_ptrs = LSE_ptr + lse_offset + offs_m * stride_lses
    
    if EVEN_S:
        tl.store(o_ptrs, out.to(tl.bfloat16))
        tl.store(lse_ptrs, lse)
    else:
        out_mask = offs_m < S
        tl.store(o_ptrs, out.to(tl.bfloat16), mask=out_mask[:, None])
        tl.store(lse_ptrs, lse, mask=out_mask)


def run(Q, K, V, O, LSE):
    """
    Computes Non-Causal Multi-Head Attention strictly using high-performance TMA patterns where structurally valid.
    
    Args:
        Q, K, V: bfloat16 input tensors of shape (B, H, S, D).
        O: preallocated bfloat16 output tensor of shape (B, H, S, D).
        LSE: preallocated float32 output tensor of shape (B, H, S).
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    
    # Blend base scaling standard with log2 transformation constant on the CPU 
    sm_scale_log2 = (1.0 / (D ** 0.5)) * 1.4426950408889634
    
    EVEN_S = (S % 128 == 0)
    
    # Hardware check assessing safe contiguous bounds mapping for direct Host Descriptors 
    use_tma = (
        Q.stride(-1) == 1 and K.stride(-1) == 1 and V.stride(-1) == 1 and O.stride(-1) == 1 and
        (Q.stride(-2) * 2) % 16 == 0 and
        (K.stride(-2) * 2) % 16 == 0 and
        (V.stride(-2) * 2) % 16 == 0 and
        (O.stride(-2) * 2) % 16 == 0 and 
        EVEN_S
    )
    
    if use_tma:
        # Ideal Hopper settings bypassing Autotune to retain fully unified TMA descriptors
        BLOCK_M, BLOCK_N = 128, 128
        
        q_desc = TensorDescriptor.from_tensor(Q, [1, 1, BLOCK_M, D])
        k_desc = TensorDescriptor.from_tensor(K, [1, 1, BLOCK_N, D])
        v_desc = TensorDescriptor.from_tensor(V, [1, 1, BLOCK_N, D])
        o_desc = TensorDescriptor.from_tensor(O, [1, 1, BLOCK_M, D])
        
        # Grid X dimension handles sequence splits enabling L2 cache reuse across subsequent CTA iterations per Head 
        grid = (triton.cdiv(S, BLOCK_M), B, H)
        mha_fwd_kernel_tma_host[grid](
            q_desc, k_desc, v_desc, o_desc,
            LSE, LSE.stride(0), LSE.stride(1), LSE.stride(2),
            B, H, S, sm_scale_log2,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, EVEN_S=EVEN_S,
            num_stages=3, num_warps=8
        )
    else:
        grid = lambda META: (triton.cdiv(S, META['BLOCK_M']), B, H)
        mha_fwd_kernel_ptr[grid](
            Q, K, V, O, LSE,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            LSE.stride(0), LSE.stride(1), LSE.stride(2),
            B, H, S, sm_scale_log2,
            D=128, EVEN_S=EVEN_S
        )