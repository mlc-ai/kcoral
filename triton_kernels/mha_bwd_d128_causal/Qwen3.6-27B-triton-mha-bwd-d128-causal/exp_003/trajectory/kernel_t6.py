import torch
import triton
import triton.language as tl


@triton.jit
def _bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dQ_out_ptr,
    S,
    inv_sqrt_d,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    HEAD_COUNT: tl.constexpr,
    D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    b = pid_bh // HEAD_COUNT
    h = pid_bh % HEAD_COUNT

    stride_B_Q = HEAD_COUNT * S * D
    stride_H_Q = S * D
    stride_B_L = HEAD_COUNT * S
    stride_H_L = S

    q_base = Q_ptr + b * stride_B_Q + h * stride_H_Q
    k_base = K_ptr + b * stride_B_Q + h * stride_H_Q
    v_base = V_ptr + b * stride_B_Q + h * stride_H_Q
    o_base = O_ptr + b * stride_B_Q + h * stride_H_Q
    do_base = dO_ptr + b * stride_B_Q + h * stride_H_Q
    l_base = L_ptr + b * stride_B_L + h * stride_H_L
    dq_base = dQ_out_ptr + b * stride_B_Q + h * stride_H_Q

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
        valid = causal & mask_m[:, None] & mask_n[None, :]

        P = tl.where(valid, tl.exp(attn - lse[:, None]), 0.0)

        dP_bar = tl.dot(do_block, v_block.T)

        PdP = P * dP_bar
        acc_t1 = tl.dot(PdP, k_block, acc_t1)
        acc_t3 = tl.dot(P, k_block, acc_t3)

    dQ_val = (acc_t1 - pD[:, None] * acc_t3) * inv_sqrt_d

    dq_ptrs = dq_base + m_idx * D + d_idx
    tl.store(dq_ptrs, dQ_val.to(tl.bfloat16), mask=mask_m[:, None])


@triton.jit
def _bwd_dkv_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dK_out_ptr, dV_out_ptr,
    S,
    inv_sqrt_d,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    HEAD_COUNT: tl.constexpr,
    D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_n = tl.program_id(1)

    b = pid_bh // HEAD_COUNT
    h = pid_bh % HEAD_COUNT

    stride_B_Q = HEAD_COUNT * S * D
    stride_H_Q = S * D
    stride_B_L = HEAD_COUNT * S
    stride_H_L = S

    k_base = K_ptr + b * stride_B_Q + h * stride_H_Q
    v_base = V_ptr + b * stride_B_Q + h * stride_H_Q
    l_base = L_ptr + b * stride_B_L + h * stride_H_L
    dk_base = dK_out_ptr + b * stride_B_Q + h * stride_H_Q
    dv_base = dV_out_ptr + b * stride_B_Q + h * stride_H_Q

    Q_base = Q_ptr + b * stride_B_Q + h * stride_H_Q
    do_base = dO_ptr + b * stride_B_Q + h * stride_H_Q
    o_base = O_ptr + b * stride_B_Q + h * stride_H_Q

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
        valid = causal & mask_m[:, None] & mask_n[None, :]

        P = tl.where(valid, tl.exp(attn - lse[:, None]), 0.0)

        dP_bar = tl.dot(do_block, v_block.T)

        dscore = P * (dP_bar - pD[:, None])

        acc_dk = tl.dot(dscore.T, q_block, acc_dk)
        acc_dv = tl.dot(P.T, do_block, acc_dv)

    dk_ptrs = dk_base + n_idx * D + d_idx
    tl.store(dk_ptrs, (acc_dk * inv_sqrt_d).to(tl.bfloat16), mask=mask_n[:, None])

    dv_ptrs = dv_base + n_idx * D + d_idx
    tl.store(dv_ptrs, acc_dv.to(tl.bfloat16), mask=mask_n[:, None])


def run(Q, K, V, O, dO, L, dQ_out, dK_out, dV_out):
    """Causal multi-head attention backward."""
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape

    import math
    inv_sqrt_d = 1.0 / math.sqrt(D)

    # Try multiple configurations; pick based on size
    if S <= 1024:
        BLOCK_M = 128
        BLOCK_N = 128
        NUM_WARPS = 8
        NUM_STAGES = 3
    elif S <= 2048:
        BLOCK_M = 128
        BLOCK_N = 64
        NUM_WARPS = 8
        NUM_STAGES = 4
    else:
        BLOCK_M = 128
        BLOCK_N = 64
        NUM_WARPS = 8
        NUM_STAGES = 3

    num_bh = B * H
    num_m_tiles = triton.cdiv(S, BLOCK_M)
    num_n_tiles = triton.cdiv(S, BLOCK_N)

    # Phase 1: compute dQ
    grid_dq = (num_bh, num_m_tiles)
    _bwd_dq_kernel[grid_dq](
        Q, K, V, O, dO, L, dQ_out,
        S,
        inv_sqrt_d,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        HEAD_COUNT=H, D=D,
        num_warps=NUM_WARPS, num_stages=NUM_STAGES,
    )

    # Phase 2: compute dK and dV
    grid_dkv = (num_bh, num_n_tiles)
    _bwd_dkv_kernel[grid_dkv](
        Q, K, V, O, dO, L, dK_out, dV_out,
        S,
        inv_sqrt_d,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        HEAD_COUNT=H, D=D,
        num_warps=NUM_WARPS, num_stages=NUM_STAGES,
    )