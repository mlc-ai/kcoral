import torch
import triton
import triton.language as tl


@triton.jit
def _dkdv_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
    dK_ptr, dV_ptr,
    B, H, S, d,
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_n  = tl.program_id(0)
    pid_bh = tl.program_id(1)

    bh_base = pid_bh * S * d

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < S
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < d
    nk_mask = mask_n[:, None] & mask_d[None, :]

    kv_base = bh_base + offs_n[:, None] * d + offs_d[None, :]
    K_reg = tl.load(K_ptr + kv_base, mask=nk_mask, other=0.0).to(tl.float32)
    V_reg = tl.load(V_ptr + kv_base, mask=nk_mask, other=0.0).to(tl.float32)

    acc_dk = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    acc_dv = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)

    num_m_tiles = tl.cdiv(S, BLOCK_M)

    for pm in range(num_m_tiles):
        offs_m = pm * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S
        mr_mask = mask_m[:, None]

        row_base = bh_base + offs_m[:, None] * d + offs_d[None, :]

        Q_reg  = tl.load(Q_ptr  + row_base, mask=mr_mask, other=0.0).to(tl.float32)
        dO_reg = tl.load(dO_ptr + row_base, mask=mr_mask, other=0.0).to(tl.float32)
        O_reg  = tl.load(O_ptr  + row_base, mask=mr_mask, other=0.0).to(tl.float32)

        L_reg = tl.load(L_ptr + pid_bh * S + offs_m, mask=mask_m, other=0.0)

        S_mat = tl.dot(Q_reg, K_reg.T) * sm_scale
        P     = tl.exp(S_mat - L_reg[:, None])
        dP    = tl.dot(dO_reg, V_reg.T)
        D     = tl.sum(dO_reg * O_reg, axis=1)
        dS    = P * (dP - D[:, None]) * sm_scale

        acc_dk = tl.dot(dS.T, Q_reg, acc=acc_dk)
        acc_dv = tl.dot(P.T, dO_reg, acc=acc_dv)

    tl.store(dK_ptr + kv_base, acc_dk.to(tl.bfloat16), mask=nk_mask)
    tl.store(dV_ptr + kv_base, acc_dv.to(tl.bfloat16), mask=nk_mask)


@triton.jit
def _dq_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
    dQ_ptr,
    B, H, S, d,
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m  = tl.program_id(0)
    pid_bh = tl.program_id(1)

    bh_base = pid_bh * S * d

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < S
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < d
    mk_mask = mask_m[:, None] & mask_d[None, :]

    row_base = bh_base + offs_m[:, None] * d + offs_d[None, :]

    Q_reg  = tl.load(Q_ptr  + row_base, mask=mk_mask, other=0.0).to(tl.float32)
    dO_reg = tl.load(dO_ptr + row_base, mask=mk_mask, other=0.0).to(tl.float32)
    O_reg  = tl.load(O_ptr  + row_base, mask=mk_mask, other=0.0).to(tl.float32)

    L_reg = tl.load(L_ptr + pid_bh * S + offs_m, mask=mask_m, other=0.0)
    D = tl.sum(dO_reg * O_reg, axis=1)

    acc_dq = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    num_n_tiles = tl.cdiv(S, BLOCK_N)

    for pn in range(num_n_tiles):
        offs_n = pn * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S
        nk_mask = mask_n[:, None] & mask_d[None, :]

        kv_base = bh_base + offs_n[:, None] * d + offs_d[None, :]
        K_reg = tl.load(K_ptr + kv_base, mask=nk_mask, other=0.0).to(tl.float32)
        V_reg = tl.load(V_ptr + kv_base, mask=nk_mask, other=0.0).to(tl.float32)

        S_mat = tl.dot(Q_reg, K_reg.T) * sm_scale
        P     = tl.exp(S_mat - L_reg[:, None])
        dP    = tl.dot(dO_reg, V_reg.T)
        dS    = P * (dP - D[:, None]) * sm_scale

        acc_dq = tl.dot(dS, K_reg, acc=acc_dq)

    tl.store(dQ_ptr + row_base, acc_dq.to(tl.bfloat16), mask=mk_mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64},  num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64},  num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64},  num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64},  num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128},  num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64},  num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64},  num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=8, num_stages=4),
    ],
    key=["S", "d"],
)
@triton.heuristics(values={"BLOCK_D": lambda args: args["d"]})
@triton.jit
def _dkdv_autotune(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
    dK_ptr, dV_ptr,
    B, H, S, d,
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    _dkdv_kernel(
        Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
        dK_ptr, dV_ptr, B, H, S, d, sm_scale,
        BLOCK_M, BLOCK_N, BLOCK_D,
    )


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64},  num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64},  num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64},  num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64},  num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128},  num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64},  num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64},  num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=8, num_stages=4),
    ],
    key=["S", "d"],
)
@triton.heuristics(values={"BLOCK_D": lambda args: args["d"]})
@triton.jit
def _dq_autotune(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
    dQ_ptr,
    B, H, S, d,
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    _dq_kernel(
        Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
        dQ_ptr, B, H, S, d, sm_scale,
        BLOCK_M, BLOCK_N, BLOCK_D,
    )


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    sm_scale = 1.0 / (d ** 0.5)
    BH = B * H

    # Autotuner will pick actual block sizes; grid extents use worst-case bounds
    max_bn = 256
    max_bm = 256
    num_kv_tiles = triton.cdiv(S, max_bn)
    num_q_tiles  = triton.cdiv(S, max_bm)

    grid_dkdv = (num_kv_tiles, BH)
    _dkdv_autotune[grid_dkdv](
        Q, K, V, O, dO, L,
        dK, dV,
        B, H, S, d, sm_scale,
        BLOCK_D=d,
    )

    grid_dq = (num_q_tiles, BH)
    _dq_autotune[grid_dq](
        Q, K, V, O, dO, L,
        dQ,
        B, H, S, d, sm_scale,
        BLOCK_D=d,
    )