import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

@triton.jit
def _attn_fwd_tma_host_desc_kernel(
    q_desc, k_desc, v_desc,
    O, LSE,
    S, H,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    STAGE_COUNT: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    offs_m_start = pid_m * BLOCK_M

    # Standard TMA descriptors dynamically map to up to 4/5 dimensions. Squeeze to our 2D Block
    q_4d = q_desc.load([pid_b, pid_h, offs_m_start, 0])
    q = tl.reshape(q_4d, (BLOCK_M, 128))

    # Fold 1/sqrt(D) scaling and log2(e) for tl.math.exp2 instruction fastpaths
    RCP_LN2: tl.constexpr = 1.4426950408889634
    scale: tl.constexpr = 0.08838834764831843
    q = (q * scale * RCP_LN2).to(tl.bfloat16)

    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, 128), tl.float32)

    n_full_blocks = S // BLOCK_N

    for block_idx in tl.range(0, n_full_blocks, num_stages=STAGE_COUNT):
        start_n = block_idx * BLOCK_N
        k_4d = k_desc.load([pid_b, pid_h, start_n, 0])
        v_4d = v_desc.load([pid_b, pid_h, start_n, 0])
        
        k = tl.reshape(k_4d, (BLOCK_N, 128))
        v = tl.reshape(v_4d, (BLOCK_N, 128))
        
        scores = tl.dot(q, k.T)
        
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.math.exp2(m_i - m_ij)
        p = tl.math.exp2(scores - m_ij[:, None])
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        m_i = m_ij

    if not EVEN_N:
        start_n = n_full_blocks * BLOCK_N
        k_4d = k_desc.load([pid_b, pid_h, start_n, 0])
        v_4d = v_desc.load([pid_b, pid_h, start_n, 0])
        
        k = tl.reshape(k_4d, (BLOCK_N, 128))
        v = tl.reshape(v_4d, (BLOCK_N, 128))
        
        scores = tl.dot(q, k.T)
        
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S
        scores = tl.where(mask_n[None, :], scores, -float("inf"))
        
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.math.exp2(m_i - m_ij)
        p = tl.math.exp2(scores - m_ij[:, None])
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        m_i = m_ij

    output = acc / l_i[:, None]
    LN2: tl.constexpr = 0.6931471805599453
    lse = (m_i + tl.math.log2(l_i)) * LN2

    offs_m = offs_m_start + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, 128)
    
    o_ptrs = O + pid_b * stride_ob + pid_h * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    lse_ptrs = LSE + pid_b * stride_lseb + pid_h * stride_lseh + offs_m * stride_lses
    
    if EVEN_M:
        tl.store(o_ptrs, output.to(tl.bfloat16))
        tl.store(lse_ptrs, lse)
    else:
        mask_m = offs_m < S
        tl.store(o_ptrs, output.to(tl.bfloat16), mask=mask_m[:, None])
        tl.store(lse_ptrs, lse, mask=mask_m)


@triton.jit
def _attn_fwd_ptr_kernel(
    Q, K, V, O, LSE,
    S, H,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    STAGE_COUNT: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    offs_m_start = pid_m * BLOCK_M
    offs_m = offs_m_start + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, 128)

    q_ptrs = Q + pid_b * stride_qb + pid_h * stride_qh + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    k_ptrs = K + pid_b * stride_kb + pid_h * stride_kh + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = V + pid_b * stride_vb + pid_h * stride_vh + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd

    if EVEN_M:
        q = tl.load(q_ptrs)
    else:
        mask_m = offs_m < S
        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    RCP_LN2: tl.constexpr = 1.4426950408889634
    scale: tl.constexpr = 0.08838834764831843
    q = (q * scale * RCP_LN2).to(tl.bfloat16)

    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, 128), tl.float32)

    n_full_blocks = S // BLOCK_N

    for block_idx in tl.range(0, n_full_blocks, num_stages=STAGE_COUNT):
        start_n = block_idx * BLOCK_N
        k = tl.load(k_ptrs + start_n * stride_ks)
        v = tl.load(v_ptrs + start_n * stride_vs)
        
        scores = tl.dot(q, k.T)
        
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.math.exp2(m_i - m_ij)
        p = tl.math.exp2(scores - m_ij[:, None])
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        m_i = m_ij

    if not EVEN_N:
        start_n = n_full_blocks * BLOCK_N
        curr_n = start_n + offs_n
        mask = curr_n < S
        k = tl.load(k_ptrs + start_n * stride_ks, mask=mask[:, None], other=0.0)
        v = tl.load(v_ptrs + start_n * stride_vs, mask=mask[:, None], other=0.0)
        
        scores = tl.dot(q, k.T)
        scores = tl.where(mask[None, :], scores, -float("inf"))
        
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.math.exp2(m_i - m_ij)
        p = tl.math.exp2(scores - m_ij[:, None])
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        m_i = m_ij

    output = acc / l_i[:, None]
    LN2: tl.constexpr = 0.6931471805599453
    lse = (m_i + tl.math.log2(l_i)) * LN2

    o_ptrs = O + pid_b * stride_ob + pid_h * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    lse_ptrs = LSE + pid_b * stride_lseb + pid_h * stride_lseh + offs_m * stride_lses
    
    if EVEN_M:
        tl.store(o_ptrs, output.to(tl.bfloat16))
        tl.store(lse_ptrs, lse)
    else:
        mask_m = offs_m < S
        tl.store(o_ptrs, output.to(tl.bfloat16), mask=mask_m[:, None])
        tl.store(lse_ptrs, lse, mask=mask_m)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    BLOCK_M = 128
    BLOCK_N = 64
    num_warps = 4
    num_stages = 4

    # Narrow to a safe footprint for shorter sequences avoiding overly sized tail shapes bounds
    if S < 128:
        BLOCK_M = 64
        BLOCK_N = 64
        num_warps = 4
        num_stages = 2
    
    grid = (triton.cdiv(S, BLOCK_M), B * H)
    EVEN_M = (S % BLOCK_M == 0)
    EVEN_N = (S % BLOCK_N == 0)
    
    # Assert conditions for physical alignment needed for standard Native TMA TensorDescriptors
    tma_supported = True
    for tensor in (Q, K, V):
        if tensor.stride(-1) != 1 or (tensor.stride(-2) * 2) % 16 != 0:
            tma_supported = False
            break

    if tma_supported:
        q_desc = TensorDescriptor.from_tensor(Q, [1, 1, BLOCK_M, 128])
        k_desc = TensorDescriptor.from_tensor(K, [1, 1, BLOCK_N, 128])
        v_desc = TensorDescriptor.from_tensor(V, [1, 1, BLOCK_N, 128])
        
        _attn_fwd_tma_host_desc_kernel[grid](
            q_desc, k_desc, v_desc, O, LSE,
            S, H,
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            LSE.stride(0), LSE.stride(1), LSE.stride(2),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            EVEN_M=EVEN_M,
            EVEN_N=EVEN_N,
            STAGE_COUNT=num_stages,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        _attn_fwd_ptr_kernel[grid](
            Q, K, V, O, LSE,
            S, H,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            LSE.stride(0), LSE.stride(1), LSE.stride(2),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            EVEN_M=EVEN_M,
            EVEN_N=EVEN_N,
            STAGE_COUNT=num_stages,
            num_warps=num_warps,
            num_stages=num_stages,
        )