import torch
import triton
import triton.language as tl

# Standard Blackwell device allocation hook for TMAs
def _tma_alloc(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(_tma_alloc)

def get_autotune_configs():
    return [
        # Blackwell tailored configs heavily leveraging TMEM and larger memory footprints.
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'LOOP_STAGES': 3, 'WARP_SPECIALIZE': True}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'LOOP_STAGES': 3, 'WARP_SPECIALIZE': True}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'LOOP_STAGES': 3, 'WARP_SPECIALIZE': True}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'LOOP_STAGES': 4, 'WARP_SPECIALIZE': True}, num_warps=8, num_stages=4),
        
        # Unspecialized fallbacks 
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'LOOP_STAGES': 3, 'WARP_SPECIALIZE': False}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'LOOP_STAGES': 3, 'WARP_SPECIALIZE': False}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'LOOP_STAGES': 3, 'WARP_SPECIALIZE': False}, num_warps=8, num_stages=3),
        
        # Lower footprint options
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'LOOP_STAGES': 3, 'WARP_SPECIALIZE': True}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'LOOP_STAGES': 3, 'WARP_SPECIALIZE': True}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'LOOP_STAGES': 3, 'WARP_SPECIALIZE': False}, num_warps=4, num_stages=3),
    ]

@triton.autotune(configs=get_autotune_configs(), key=['S'])
@triton.jit
def _fwd_kernel(
    Q, K, V, O, LSE,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    B, H, S, D,
    sm_scale_log2,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    LOOP_STAGES: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
    S_MOD_N_ZERO: tl.constexpr,
):
    # Logically mapping grid to process matching heads consecutively (promotes native KV L2 caching hits).
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    start_m = pid_m * BLOCK_M

    q_ptr = Q + b * stride_qb + h * stride_qh
    k_ptr = K + b * stride_kb + h * stride_kh
    v_ptr = V + b * stride_vb + h * stride_vh
    o_ptr = O + b * stride_ob + h * stride_oh

    # Hardware optimized 2D TMA descriptors bounding logic strictly mapping the device offsets
    q_desc = tl.make_tensor_descriptor(
        q_ptr, shape=[S, D], strides=[stride_qs, stride_qd],
        block_shape=[BLOCK_M, BLOCK_D], padding_option="zero"
    )
    k_desc = tl.make_tensor_descriptor(
        k_ptr, shape=[S, D], strides=[stride_ks, stride_kd],
        block_shape=[BLOCK_N, BLOCK_D], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        v_ptr, shape=[S, D], strides=[stride_vs, stride_vd],
        block_shape=[BLOCK_N, BLOCK_D], padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        o_ptr, shape=[S, D], strides=[stride_os, stride_od],
        block_shape=[BLOCK_M, BLOCK_D], padding_option="zero"
    )

    q = q_desc.load([start_m, 0])

    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    m_i = tl.full((BLOCK_M,), -float('inf'), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)

    num_full_steps = S // BLOCK_N

    # Mask-Free pipelined loop (opt-in warp specialization segregating TMEM bound operations)
    for n_idx in tl.range(0, num_full_steps, num_stages=LOOP_STAGES, warp_specialize=WARP_SPECIALIZE):
        start_n = n_idx * BLOCK_N
        k = k_desc.load([start_n, 0])
        v = v_desc.load([start_n, 0])

        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        qk = qk * sm_scale_log2
        
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)

        alpha = tl.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        m_i = m_ij

        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc, out_dtype=tl.float32)

    # Strictly isolated remainder processing (compiled out completely if matching dimensions are passed)
    if not S_MOD_N_ZERO:
        if S % BLOCK_N != 0:
            start_n = num_full_steps * BLOCK_N
            k = k_desc.load([start_n, 0])
            v = v_desc.load([start_n, 0])

            qk = tl.dot(q, k.T, out_dtype=tl.float32)
            qk = qk * sm_scale_log2
            
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < S
            qk = tl.where(mask_n[None, :], qk, float('-inf'))
            
            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp2(qk - m_ij[:, None])
            l_ij = tl.sum(p, axis=1)

            alpha = tl.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            m_i = m_ij

            acc = acc * alpha[:, None]
            acc = tl.dot(p.to(tl.bfloat16), v, acc, out_dtype=tl.float32)

    l_i_inv = 1.0 / l_i
    acc = acc * l_i_inv[:, None]
    out = acc.to(tl.bfloat16)
    
    o_desc.store([start_m, 0], out)

    # Structure naturally tracking log structures mathematically matched back to natural bases
    LN_2 = 0.6931471805599453
    lse = (m_i + tl.log2(l_i)) * LN_2
    
    offs_m = start_m + tl.arange(0, BLOCK_M)
    lse_base = LSE + b * stride_lseb + h * stride_lseh
    lse_ptrs = lse_base + offs_m
    
    if S_MOD_N_ZERO:
        tl.store(lse_ptrs, lse)
    else:
        mask_m = offs_m < S
        tl.store(lse_ptrs, lse, mask=mask_m)

def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    if S == 0:
        return

    grid = lambda META: (triton.cdiv(S, META['BLOCK_M']), B * H)
    sm_scale = 1.0 / (D ** 0.5)
    
    # Accelerated scale passing dynamically tracking Base 2 optimizations
    LOG2_E = 1.4426950408889634
    sm_scale_log2 = sm_scale * LOG2_E
    
    # Informs compiler that tensor offsets intrinsically match hardware padding limitations 
    S_MOD_N_ZERO = (S % 256 == 0)

    _fwd_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        B, H, S, D,
        sm_scale_log2,
        BLOCK_D=128,
        S_MOD_N_ZERO=S_MOD_N_ZERO
    )