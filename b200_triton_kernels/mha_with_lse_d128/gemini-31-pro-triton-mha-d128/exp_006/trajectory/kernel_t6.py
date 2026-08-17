import torch
import triton
import triton.language as tl

# Set Triton's allocator for device-created tensor descriptors
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=3, num_warps=8, num_ctas=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=3, num_warps=4),
    ],
    key=['S']
)
@triton.jit
def mha_fwd_kernel_tma(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    stride_qb, stride_qh, stride_qs,
    stride_kb, stride_kh, stride_ks,
    stride_vb, stride_vh, stride_vs,
    stride_ob, stride_oh, stride_os,
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
    
    # Base offsets for the current Batch and Head
    q_offset = pid_b * stride_qb + pid_h * stride_qh
    k_offset = pid_b * stride_kb + pid_h * stride_kh
    v_offset = pid_b * stride_vb + pid_h * stride_vh
    o_offset = pid_b * stride_ob + pid_h * stride_oh
    lse_offset = pid_b * stride_lseb + pid_h * stride_lseh
    
    # Extremely lightweight 2D TMA Descriptors instantiated cleanly on device 
    # to avoid heavy Python Host execution overheads during repeated kernel launches
    q_desc = tl.make_tensor_descriptor(
        Q_ptr + q_offset,
        shape=[S, D],
        strides=[stride_qs, 1],
        block_shape=[BLOCK_M, D],
        padding_option="zero"
    )
    k_desc = tl.make_tensor_descriptor(
        K_ptr + k_offset,
        shape=[S, D],
        strides=[stride_ks, 1],
        block_shape=[BLOCK_N, D],
        padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        V_ptr + v_offset,
        shape=[S, D],
        strides=[stride_vs, 1],
        block_shape=[BLOCK_N, D],
        padding_option="zero"
    )
    # Note: no padding_option for the store descriptor, out-of-bounds writes are safely ignored implicitly
    o_desc = tl.make_tensor_descriptor(
        O_ptr + o_offset,
        shape=[S, D],
        strides=[stride_os, 1],
        block_shape=[BLOCK_M, D]
    )
    
    start_m = pid_m * BLOCK_M
    q = tl.load(q_desc, [start_m, 0])
    
    # KEY OPTIMIZATION: Pre-scale Q to force it securely into Registers.
    # This transforms the subsequent hardware WGMMA calls into RS-GEMM (Register-Shared) operations,
    # completely bypassing ~16,000 FLOPs per inner-loop iteration that would've occurred inside `qk = qk * sm_scale_log2`.
    q = (q * sm_scale_log2).to(tl.bfloat16)
    
    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    
    num_n_blocks = tl.cdiv(S, BLOCK_N)
    
    # Leverage robust range mechanics compatible with standard Triton pipelining heuristics
    for start_n_idx in range(num_n_blocks):
        start_n = start_n_idx * BLOCK_N
        
        # Async TMA hardware issues fetches bypassing traditional LDG instructions
        k = tl.load(k_desc, [start_n, 0])
        v = tl.load(v_desc, [start_n, 0])
        
        # Hardware executes Ping-Pong scheduling of RS-GEMMs directly natively on 8 warps
        qk = tl.dot(q, k.T)
        
        # Eliminate masking conditionals completely if sequence structurally maps powers of two cleanly 
        if not EVEN_S:
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask = offs_n[None, :] < S
            qk = tl.where(mask, qk, float("-inf"))
        
        # Mathematically optimal robust online softmax utilizing base-2 exponential properties
        m_i_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp2(m_i - m_i_new)
        p = tl.exp2(qk - m_i_new[:, None])
        l_i_new = alpha * l_i + tl.sum(p, axis=1)
        
        # Scale accumulated values and overlap projections
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        
        m_i = m_i_new
        l_i = l_i_new
        
    # Scale inverse norms at completion and store out bulk sequences natively via TMA
    l_i_safe = tl.where(l_i == 0.0, 1.0, l_i)
    out = acc * (1.0 / l_i_safe[:, None])
    tl.store(o_desc, [start_m, 0], out.to(tl.bfloat16))
    
    offs_m = start_m + tl.arange(0, BLOCK_M)
    lse = (m_i + tl.log2(l_i)) * 0.6931471805599453
    lse_ptrs = LSE_ptr + lse_offset + offs_m * stride_lses
    
    if EVEN_S:
        tl.store(lse_ptrs, lse)
    else:
        tl.store(lse_ptrs, lse, mask=offs_m < S)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=3, num_warps=4),
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
        q_mask = (offs_m[:, None] < S)
        q = tl.load(q_ptrs, mask=q_mask, other=0.0)

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
    Computes Non-Causal Multi-Head Attention efficiently mapping WGMMA and TMA async capabilities targeting Hopper SM90 Architecture.
    
    Args:
        Q, K, V: bfloat16 input tensors of shape (B, H, S, D).
        O: preallocated bfloat16 output tensor of shape (B, H, S, D).
        LSE: preallocated float32 output tensor of shape (B, H, S).
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    
    # Standardize scale transitions cleanly into log2 base representations prior to kernel invocation
    sm_scale_log2 = (1.0 / (D ** 0.5)) * 1.4426950408889634
    
    # Assess mathematical guarantees enabling complete masking eliminations
    EVEN_S = (S % 128 == 0)
    
    grid = lambda META: (triton.cdiv(S, META['BLOCK_M']), B, H)
    
    # Preflight integrity checks to ensure alignment safety for the asynchronous TMA engine bindings.
    use_tma = (
        Q.stride(-1) == 1 and K.stride(-1) == 1 and V.stride(-1) == 1 and O.stride(-1) == 1 and
        (Q.stride(-2) * 2) % 16 == 0 and
        (K.stride(-2) * 2) % 16 == 0 and
        (V.stride(-2) * 2) % 16 == 0 and
        (O.stride(-2) * 2) % 16 == 0
    )
    
    if use_tma:
        mha_fwd_kernel_tma[grid](
            Q, K, V, O, LSE,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            LSE.stride(0), LSE.stride(1), LSE.stride(2),
            B, H, S, sm_scale_log2,
            D=D,
            EVEN_S=EVEN_S
        )
    else:
        mha_fwd_kernel_ptr[grid](
            Q, K, V, O, LSE,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            LSE.stride(0), LSE.stride(1), LSE.stride(2),
            B, H, S, sm_scale_log2,
            D=D,
            EVEN_S=EVEN_S
        )