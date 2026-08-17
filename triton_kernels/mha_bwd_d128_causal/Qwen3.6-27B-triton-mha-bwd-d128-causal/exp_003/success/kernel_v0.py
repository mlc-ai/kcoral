import torch
import triton
import triton.language as tl


@triton.jit
def _bwd_dq_kernel(
    Q, K, V, O, dO, L, dQ,
    S,
    inv_sqrt_d,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    HEAD_COUNT: tl.constexpr,
    D: tl.constexpr,
):
    """Compute dQ for causal MHA backward.
    
    Formula: dQ[m,:] = (Σₙ P[m,n]*dP̄[m,n]*K[n,:] - pD[m]*Σₙ P[m,n]*K[n,:]) / √d
    where pD[m] = dO[m,:]·O[m,:] (dot-product identity).
    """
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    b = pid_bh // HEAD_COUNT
    h = pid_bh % HEAD_COUNT

    stride_B = HEAD_COUNT * S * D
    stride_H = S * D
    stride_L_B = HEAD_COUNT * S
    stride_L_H = S

    q_base = Q + b * stride_B + h * stride_H
    k_base = K + b * stride_B + h * stride_H
    v_base = V + b * stride_B + h * stride_H
    o_base = O + b * stride_B + h * stride_H
    do_base = dO + b * stride_B + h * stride_H
    l_base = L + b * stride_L_B + h * stride_L_H
    dq_base = dQ + b * stride_B + h * stride_H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < S
    offs_d = tl.arange(0, D)
    m_idx = offs_m[:, None]
    d_idx = offs_d[None, :]

    lse = tl.load(l_base + offs_m, mask=mask_m, other=0.0)

    q_ptrs = q_base + m_idx * D + d_idx
    q_block = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)

    do_ptrs = do_base + m_idx * D + d_idx
    do_block = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)

    o_ptrs = o_base + m_idx * D + d_idx
    o_block = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)

    pD = tl.sum(do_block * o_block, axis=1)

    acc_t1 = tl.zeros((BLOCK_M, D), dtype=tl.float32)
    acc_t3 = tl.zeros((BLOCK_M, D), dtype=tl.float32)

    num_n_tiles = tl.cdiv(S, BLOCK_N)

    for pi in range(num_n_tiles):
        offs_n = pi * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S
        n_idx = offs_n[:, None]

        k_ptrs = k_base + n_idx * D + d_idx
        k_block = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)

        v_ptrs = v_base + n_idx * D + d_idx
        v_block = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)

        attn = tl.dot(q_block, k_block.T) * inv_sqrt_d

        causal = offs_m[:, None] >= offs_n[None, :]
        mask = causal & mask_m[:, None] & mask_n[None, :]

        attn_safe = tl.where(mask, attn, float('-inf'))
        P = tl.exp(attn_safe - lse[:, None])
        P = tl.where(mask, P, 0.0)

        dP_bar = tl.dot(do_block, v_block.T)

        PdP = P * dP_bar
        acc_t1 = tl.dot(PdP, k_block, acc_t1)
        acc_t3 = tl.dot(P, k_block, acc_t3)

    dQ_val = (acc_t1 - pD[:, None] * acc_t3) * inv_sqrt_d

    dq_ptrs = dq_base + m_idx * D + d_idx
    tl.store(dq_ptrs, dQ_val.to(tl.bfloat16), mask=mask_m[:, None])


@triton.jit
def _bwd_dkv_kernel(
    Q, K, V, O, dO, L, dK, dV,
    S,
    inv_sqrt_d,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    HEAD_COUNT: tl.constexpr,
    D: tl.constexpr,
):
    """Compute dK and dV for causal MHA backward (fused).
    
    dV[n,:] = Σₘ P[m,n] * dO[m,:]
    dK[n,:] = Σₘ dscore[m,n] * Q[m,:] / √d
    where dscore[m,n] = P[m,n] * (dP̄[m,n] - pD[m])
    """
    pid_bh = tl.program_id(0)
    pid_n = tl.program_id(1)

    b = pid_bh // HEAD_COUNT
    h = pid_bh % HEAD_COUNT

    stride_B = HEAD_COUNT * S * D
    stride_H = S * D
    stride_L_B = HEAD_COUNT * S
    stride_L_H = S

    k_base = K + b * stride_B + h * stride_H
    v_base = V + b * stride_B + h * stride_H
    l_base = L + b * stride_L_B + h * stride_L_H
    dk_base = dK + b * stride_B + h * stride_H
    dv_base = dV + b * stride_B + h * stride_H

    Q_base = Q + b * stride_B + h * stride_H
    do_base = dO + b * stride_B + h * stride_H
    o_base = O + b * stride_B + h * stride_H

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < S
    offs_d = tl.arange(0, D)
    n_idx = offs_n[:, None]
    d_idx = offs_d[None, :]

    k_ptrs = k_base + n_idx * D + d_idx
    k_block = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)

    v_ptrs = v_base + n_idx * D + d_idx
    v_block = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)

    acc_dk = tl.zeros((BLOCK_N, D), dtype=tl.float32)
    acc_dv = tl.zeros((BLOCK_N, D), dtype=tl.float32)

    num_m_tiles = tl.cdiv(S, BLOCK_M)

    for pi in range(num_m_tiles):
        offs_m = pi * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S
        m_idx = offs_m[:, None]

        q_ptrs = Q_base + m_idx * D + d_idx
        q_block = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)

        do_ptrs = do_base + m_idx * D + d_idx
        do_block = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)

        o_ptrs = o_base + m_idx * D + d_idx
        o_block = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)

        lse = tl.load(l_base + offs_m, mask=mask_m, other=0.0)

        pD = tl.sum(do_block * o_block, axis=1)

        attn = tl.dot(q_block, k_block.T) * inv_sqrt_d

        causal = offs_m[:, None] >= offs_n[None, :]
        mask = causal & mask_m[:, None] & mask_n[None, :]

        attn_safe = tl.where(mask, attn, float('-inf'))
        P = tl.exp(attn_safe - lse[:, None])
        P = tl.where(mask, P, 0.0)

        dP_bar = tl.dot(do_block, v_block.T)

        dscore = P * (dP_bar - pD[:, None])

        acc_dk = tl.dot(dscore.T, q_block, acc_dk)
        acc_dv = tl.dot(P.T, do_block, acc_dv)

    dk_ptrs = dk_base + n_idx * D + d_idx
    tl.store(dk_ptrs, (acc_dk * inv_sqrt_d).to(tl.bfloat16), mask=mask_n[:, None])

    dv_ptrs = dv_base + n_idx * D + d_idx
    tl.store(dv_ptrs, acc_dv.to(tl.bfloat16), mask=mask_n[:, None])


def run(Q, K, V, O, dO, L, dQ_out, dK_out, dV_out):
    """Causal multi-head attention backward.
    
    Arguments are received in definition order: all inputs then all preallocated outputs.
    Writes dQ, dK, dV into the supplied output tensors.
    """
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape

    import math
    inv_sqrt_d = 1.0 / math.sqrt(D)

    BLOCK_M = 64
    BLOCK_N = 64

    num_bh = B * H
    num_m_tiles = triton.cdiv(S, BLOCK_M)
    num_n_tiles = triton.cdiv(S, BLOCK_N)

    grid_dq = (num_bh, num_m_tiles)
    _bwd_dq_kernel[grid_dq](
        Q, K, V, O, dO, L, dQ_out,
        S,
        inv_sqrt_d,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        HEAD_COUNT=H, D=D,
        num_warps=4, num_stages=2,
    )

    grid_dkv = (num_bh, num_n_tiles)
    _bwd_dkv_kernel[grid_dkv](
        Q, K, V, O, dO, L, dK_out, dV_out,
        S,
        inv_sqrt_d,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        HEAD_COUNT=H, D=D,
        num_warps=4, num_stages=2,
    )