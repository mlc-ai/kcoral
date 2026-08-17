import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


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

    stride_B_BHD = HEAD_COUNT * S * D
    stride_H_SD = S * D
    stride_L_BHS = HEAD_COUNT * S
    stride_L_HS = S

    q_base = Q_ptr + b * stride_B_BHD + h * stride_H_SD
    k_base = K_ptr + b * stride_B_BHD + h * stride_H_SD
    v_base = V_ptr + b * stride_B_BHD + h * stride_H_SD
    o_base = O_ptr + b * stride_B_BHD + h * stride_H_SD
    do_base = dO_ptr + b * stride_B_BHD + h * stride_H_SD
    l_base = L_ptr + b * stride_L_BHS + h * stride_L_HS
    dq_base = dQ_out_ptr + b * stride_B_BHD + h * stride_H_SD

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

    for pi in range(tl.cdiv(S, BLOCK_N)):
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

        attn_safe = tl.where(valid, attn, float('-inf'))
        P = tl.exp(attn_safe - lse[:, None])
        P = tl.where(valid, P, 0.0)

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

    stride_B_BHD = HEAD_COUNT * S * D
    stride_H_SD = S * D
    stride_L_BHS = HEAD_COUNT * S
    stride_L_HS = S

    k_base = K_ptr + b * stride_B_BHD + h * stride_H_SD
    v_base = V_ptr + b * stride_B_BHD + h * stride_H_SD
    l_base = L_ptr + b * stride_L_BHS + h * stride_L_HS
    dk_base = dK_out_ptr + b * stride_B_BHD + h * stride_H_SD
    dv_base = dV_out_ptr + b * stride_B_BHD + h * stride_H_SD

    Q_base = Q_ptr + b * stride_B_BHD + h * stride_H_SD
    do_base = dO_ptr + b * stride_B_BHD + h * stride_H_SD
    o_base = O_ptr + b * stride_B_BHD + h * stride_H_SD

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

    for pi in range(tl.cdiv(S, BLOCK_M)):
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

        attn_safe = tl.where(valid, attn, float('-inf'))
        P = tl.exp(attn_safe - lse[:, None])
        P = tl.where(valid, P, 0.0)

        dP_bar = tl.dot(do_block, v_block.T)

        dscore = P * (dP_bar - pD[:, None])

        acc_dk = tl.dot(dscore.T, q_block, acc_dk)
        acc_dv = tl.dot(P.T, do_block, acc_dv)

    dk_ptrs = dk_base + n_idx * D + d_idx
    tl.store(dk_ptrs, (acc_dk * inv_sqrt_d).to(tl.bfloat16), mask=mask_n[:, None])

    dv_ptrs = dv_base + n_idx * D + d_idx
    tl.store(dv_ptrs, acc_dv.to(tl.bfloat16), mask=mask_n[:, None])


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
    ],
    key=["S"],
)
@triton.jit
def _bwd_dq_kernel_tuned(
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

    stride_B_BHD = HEAD_COUNT * S * D
    stride_H_SD = S * D
    stride_L_BHS = HEAD_COUNT * S
    stride_L_HS = S

    q_base = Q_ptr + b * stride_B_BHD + h * stride_H_SD
    k_base = K_ptr + b * stride_B_BHD + h * stride_H_SD
    v_base = V_ptr + b * stride_B_BHD + h * stride_H_SD
    o_base = O_ptr + b * stride_B_BHD + h * stride_H_SD
    do_base = dO_ptr + b * stride_B_BHD + h * stride_H_SD
    l_base = L_ptr + b * stride_L_BHS + h * stride_L_HS
    dq_base = dQ_out_ptr + b * stride_B_BHD + h * stride_H_SD

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

    for pi in range(tl.cdiv(S, BLOCK_N)):
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

        attn_safe = tl.where(valid, attn, float('-inf'))
        P = tl.exp(attn_safe - lse[:, None])
        P = tl.where(valid, P, 0.0)

        dP_bar = tl.dot(do_block, v_block.T)

        PdP = P * dP_bar
        acc_t1 = tl.dot(PdP, k_block, acc_t1)
        acc_t3 = tl.dot(P, k_block, acc_t3)

    dQ_val = (acc_t1 - pD[:, None] * acc_t3) * inv_sqrt_d

    dq_ptrs = dq_base + m_idx * D + d_idx
    tl.store(dq_ptrs, dQ_val.to(tl.bfloat16), mask=mask_m[:, None])


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
    ],
    key=["S"],
)
@triton.jit
def _bwd_dkv_kernel_tuned(
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

    stride_B_BHD = HEAD_COUNT * S * D
    stride_H_SD = S * D
    stride_L_BHS = HEAD_COUNT * S
    stride_L_HS = S

    k_base = K_ptr + b * stride_B_BHD + h * stride_H_SD
    v_base = V_ptr + b * stride_B_BHD + h * stride_H_SD
    l_base = L_ptr + b * stride_L_BHS + h * stride_L_HS
    dk_base = dK_out_ptr + b * stride_B_BHD + h * stride_H_SD
    dv_base = dV_out_ptr + b * stride_B_BHD + h * stride_H_SD

    Q_base = Q_ptr + b * stride_B_BHD + h * stride_H_SD
    do_base = dO_ptr + b * stride_B_BHD + h * stride_H_SD
    o_base = O_ptr + b * stride_B_BHD + h * stride_H_SD

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

    for pi in range(tl.cdiv(S, BLOCK_M)):
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

        attn_safe = tl.where(valid, attn, float('-inf'))
        P = tl.exp(attn_safe - lse[:, None])
        P = tl.where(valid, P, 0.0)

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

    num_bh = B * H
    num_tiles_m = triton.cdiv(S, 128)
    num_tiles_n = triton.cdiv(S, 128)

    grid_dq = lambda META: (num_bh, triton.cdiv(S, META["BLOCK_M"]))
    _bwd_dq_kernel_tuned[grid_dq](
        Q, K, V, O, dO, L, dQ_out,
        S,
        inv_sqrt_d,
        HEAD_COUNT=H, D=D,
    )

    grid_dkv = lambda META: (num_bh, triton.cdiv(S, META["BLOCK_N"]))
    _bwd_dkv_kernel_tuned[grid_dkv](
        Q, K, V, O, dO, L, dK_out, dV_out,
        S,
        inv_sqrt_d,
        HEAD_COUNT=H, D=D,
    )