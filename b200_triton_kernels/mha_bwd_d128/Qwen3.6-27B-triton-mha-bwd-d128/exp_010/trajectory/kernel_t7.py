import math
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Small tiles, fewer warps
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=8, num_stages=4),
        # Standard size
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=5),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        # Larger M tiles
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=5),
        # Smaller N tiles
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 16}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 16}, num_warps=8, num_stages=3),
        # Very large tiles
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 32}, num_warps=8, num_stages=3),
    ],
    key=["S", "d"],
)
@triton.jit
def _dKV_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
    dK_ptr, dV_ptr,
    S, d, scale, num_m_blocks,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Computes dK and dV for one (batch*head, kv_sequence_block)."""
    pid_bh = tl.program_id(0)
    pid_n = tl.program_id(1)

    base = pid_bh * S * d

    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = off_n < S

    off_d = tl.arange(0, 128)
    mask_d = off_d < d

    # Load K and V once: [BLOCK_N, 128] bf16
    kv_ptrs = base + off_n[:, None] * d + off_d[None, :]
    kv_mask = mask_n[:, None] & mask_d[None, :]
    K = tl.load(K_ptr + kv_ptrs, mask=kv_mask, other=0.0)
    V = tl.load(V_ptr + kv_ptrs, mask=kv_mask, other=0.0)

    K_T = K.T
    V_T = V.T

    dK_acc = tl.zeros((BLOCK_N, 128), dtype=tl.float32)
    dV_acc = tl.zeros((BLOCK_N, 128), dtype=tl.float32)

    L_base = pid_bh * S

    for pid_m in range(num_m_blocks):
        off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = off_m < S

        q_ptrs = base + off_m[:, None] * d + off_d[None, :]
        q_mask = mask_m[:, None] & mask_d[None, :]

        Q_m = tl.load(Q_ptr + q_ptrs, mask=q_mask, other=0.0)
        dO_m = tl.load(dO_ptr + q_ptrs, mask=q_mask, other=0.0)
        O_m = tl.load(O_ptr + q_ptrs, mask=q_mask, other=0.0)

        D_m = tl.sum(dO_m.to(tl.float32) * O_m.to(tl.float32), axis=1)

        S_mat = tl.dot(Q_m, K_T) * scale

        L_m = tl.load(L_ptr + L_base + off_m, mask=mask_m, other=0.0).to(tl.float32)
        P = tl.exp(S_mat - L_m[:, None])
        dP = tl.dot(dO_m, V_T)
        dS = P * (dP - D_m[:, None]) * scale

        dV_acc = tl.dot(P.to(tl.bfloat16).T, dO_m, acc=dV_acc)
        dK_acc = tl.dot(dS.to(tl.bfloat16).T, Q_m, acc=dK_acc)

    tl.store(dK_ptr + kv_ptrs, dK_acc.to(tl.bfloat16), mask=kv_mask)
    tl.store(dV_ptr + kv_ptrs, dV_acc.to(tl.bfloat16), mask=kv_mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=5),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=5),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 16}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 16}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 32}, num_warps=8, num_stages=3),
    ],
    key=["S", "d"],
)
@triton.jit
def _dQ_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
    dQ_ptr,
    S, d, scale, num_n_blocks,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Computes dQ for one (batch*head, query_sequence_block)."""
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    base = pid_bh * S * d

    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = off_m < S

    off_d = tl.arange(0, 128)
    mask_d = off_d < d

    q_ptrs = base + off_m[:, None] * d + off_d[None, :]
    q_mask = mask_m[:, None] & mask_d[None, :]

    Q_m = tl.load(Q_ptr + q_ptrs, mask=q_mask, other=0.0)
    dO_m = tl.load(dO_ptr + q_ptrs, mask=q_mask, other=0.0)
    O_m = tl.load(O_ptr + q_ptrs, mask=q_mask, other=0.0)

    D_m = tl.sum(dO_m.to(tl.float32) * O_m.to(tl.float32), axis=1)
    L_m = tl.load(L_ptr + pid_bh * S + off_m, mask=mask_m, other=0.0).to(tl.float32)

    dQ_acc = tl.zeros((BLOCK_M, 128), dtype=tl.float32)

    for pid_n in range(num_n_blocks):
        off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = off_n < S

        kv_ptrs = base + off_n[:, None] * d + off_d[None, :]
        kv_mask = mask_n[:, None] & mask_d[None, :]

        K_n = tl.load(K_ptr + kv_ptrs, mask=kv_mask, other=0.0)
        V_n = tl.load(V_ptr + kv_ptrs, mask=kv_mask, other=0.0)

        S_mat = tl.dot(Q_m, K_n.T) * scale
        P = tl.exp(S_mat - L_m[:, None])
        dP = tl.dot(dO_m, V_n.T)
        dS = P * (dP - D_m[:, None]) * scale

        dQ_acc = tl.dot(dS.to(tl.bfloat16), K_n, acc=dQ_acc)

    tl.store(dQ_ptr + q_ptrs, dQ_acc.to(tl.bfloat16), mask=q_mask)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Multi-head attention backward pass.
    
    Inputs:
        Q  : [B, H, S, d] bf16
        K  : [B, H, S, d] bf16
        V  : [B, H, S, d] bf16
        O  : [B, H, S, d] bf16
        dO : [B, H, S, d] bf16
        L  : [B, H, S]   fp32
    
    Outputs (preallocated):
        dQ : [B, H, S, d] bf16
        dK : [B, H, S, d] bf16
        dV : [B, H, S, d] bf16
    """
    torch.cuda.set_device(Q.device)
    B, H, S, d_val = Q.shape
    scale = 1.0 / math.sqrt(d_val)

    num_bh = B * H

    num_m_blocks_hint = triton.cdiv(S, 64)
    num_n_blocks_hint = triton.cdiv(S, 64)

    # Zero outputs
    dK.zero_()
    dV.zero_()
    dQ.zero_()

    # Grid lambdas for autotune (need META dict access for selected block shapes)
    grid_dkv = lambda META: (num_bh, triton.cdiv(S, META["BLOCK_N"]))
    grid_dq = lambda META: (num_bh, triton.cdiv(S, META["BLOCK_M"]))

    # Launch dKV kernel
    _dKV_kernel[grid_dkv](
        Q, K, V, O, dO, L, dK, dV,
        S, d_val, scale, num_m_blocks_hint,
    )

    # Launch dQ kernel  
    _dQ_kernel[grid_dq](
        Q, K, V, O, dO, L, dQ,
        S, d_val, scale, num_n_blocks_hint,
    )