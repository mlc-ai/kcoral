import torch
import triton
import triton.language as tl

# Required for device-created tensor descriptors on Hopper (TMA setup)
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

@triton.autotune(
    configs=[
        # High occupancy, best bandwidth utilization
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64}, num_warps=8, num_stages=4),
        
        # Balanced configurations
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        
        # Lower register pressure fallback
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=4),
        
        # More fallback options
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
    ],
    key=['S'],
)
@triton.jit
def _attn_fwd_kernel(
    Q, K, V, sm_scale_log2,
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
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    # Early exit for entirely out-of-bounds blocks
    if pid_m * BLOCK_M >= S:
        return

    # Base offsets for the current batch and head
    q_offset = pid_b * stride_qz + pid_h * stride_qh
    k_offset = pid_b * stride_kz + pid_h * stride_kh
    v_offset = pid_b * stride_vz + pid_h * stride_vh
    o_offset = pid_b * stride_oz + pid_h * stride_oh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    # Load Q block directly into registers (bypasses shared memory, saving BW and space)
    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    mask_m = offs_m < S
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    # Create TMA descriptors for K and V to exploit asynchronous hardware loads to shared memory
    k_desc = tl.make_tensor_descriptor(
        K + k_offset,
        shape=[S, BLOCK_D],
        strides=[stride_km, stride_kk],
        block_shape=[BLOCK_N, BLOCK_D],
        padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        V + v_offset,
        shape=[S, BLOCK_D],
        strides=[stride_vm, stride_vk],
        block_shape=[BLOCK_N, BLOCK_D],
        padding_option="zero"
    )

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # 1. Fully Unmasked Blocks
    # Since pid_m * BLOCK_M < S, all sequence positions up to pid_m * BLOCK_M are strictly < S.
    # Therefore, TMA will never load out of bounds here, and causal masks are mathematically unnecessary.
    end_n_full = pid_m * BLOCK_M
    
    for start_n in tl.range(0, end_n_full, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        
        # Load K via TMA and compute WGMMA dot product directly from shared memory
        k = k_desc.load([start_n, 0])
        qk = tl.dot(q, k.T)
        qk = qk * sm_scale_log2
        
        # FlashAttention numerically-stable softmax (base 2)
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp2(m_i - m_i_new)
        p = tl.exp2(qk - m_i_new[:, None])
        l_i_new = alpha * l_i + tl.sum(p, 1)
        
        # Load V via TMA and accumulate values via WGMMA
        v = v_desc.load([start_n, 0])
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        
        m_i = m_i_new
        l_i = l_i_new

    # 2. Masked Causal Blocks
    end_n_causal = tl.minimum((pid_m + 1) * BLOCK_M, S)
    
    for start_n in tl.range(end_n_full, end_n_causal, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        
        k = k_desc.load([start_n, 0])
        qk = tl.dot(q, k.T)
        qk = qk * sm_scale_log2
        
        # Apply causal masking
        offs_n_curr = start_n + tl.arange(0, BLOCK_N)
        mask = offs_m[:, None] >= offs_n_curr[None, :]
        
        qk = tl.where(mask, qk, float("-inf"))
        
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp2(m_i - m_i_new)
        p = tl.exp2(qk - m_i_new[:, None])
        l_i_new = alpha * l_i + tl.sum(p, 1)
        
        v = v_desc.load([start_n, 0])
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        
        m_i = m_i_new
        l_i = l_i_new

    # Epilogue operations
    # Multiplication by reciprocal is generally faster than division
    l_i_rcp = 1.0 / l_i
    acc = acc * l_i_rcp[:, None]
    
    # LSE computation: revert log2 scale (m_i * ln(2)) and add ln(l_i) to match exact mathematical reference
    lse = m_i * 0.6931471805599453 + tl.log(l_i)
    
    # Store Attention Output directly from registers to global memory
    o_ptrs = O + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(tl.bfloat16), mask=mask_m[:, None])
    
    # Store Log-Sum-Exp 
    lse_offset = pid_b * stride_lsez + pid_h * stride_lseh
    lse_ptrs = LSE + lse_offset + offs_m * stride_lsem
    tl.store(lse_ptrs, lse, mask=mask_m)

def run(Q, K, V, O, LSE):
    """
    Computes Causal Multi-Head Attention forward pass and stores the results directly
    into pre-allocated destination tensors without re-allocating.
    """
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    # Precompute base-2 scaling factor on the host
    sm_scale_log2 = (1.0 / (D ** 0.5)) * 1.4426950408889634

    # Grid strictly structured such that pid_m varies fastest.
    # This natively maximizes L2 cache locality across blocks of the same batch/head.
    grid = lambda META: (
        triton.cdiv(S, META['BLOCK_M']),
        B,
        H,
    )
    
    _attn_fwd_kernel[grid](
        Q, K, V, sm_scale_log2,
        O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        S,
        BLOCK_D=D,
    )