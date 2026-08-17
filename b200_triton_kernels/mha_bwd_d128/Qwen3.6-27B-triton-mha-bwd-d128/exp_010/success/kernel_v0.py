import math
import torch
import triton
import triton.language as tl


@triton.jit
def _dKV_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
    dK_ptr, dV_ptr,
    S, d, scale, num_m_blocks,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Computes dK and dV for one (batch*head, kv_sequence_block) assignment.
    Iterates over all query blocks to accumulate gradients.
    """
    pid_bh = tl.program_id(0)
    pid_n = tl.program_id(1)

    base = pid_bh * S * d

    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = off_n < S

    off_d = tl.arange(0, BLOCK_D)
    mask_d = off_d < d

    # Load K and V tiles: [BLOCK_N, BLOCK_D] — bf16
    kv_ptrs = base + off_n[:, None] * d + off_d[None, :]
    kv_mask = mask_n[:, None] & mask_d[None, :]
    K = tl.load(K_ptr + kv_ptrs, mask=kv_mask, other=0.0)   # bf16 [N,D]
    V = tl.load(V_ptr + kv_ptrs, mask=kv_mask, other=0.0)   # bf16 [N,D]

    # Initialize gradient accumulators in fp32
    dK_acc = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    dV_acc = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)

    L_base = pid_bh * S

    # Transpose copies for dot products (bf16 → stored separately)
    K_T = K.T     # [D, N]
    V_T = V.T     # [D, N]

    for pid_m in range(num_m_blocks):
        off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = off_m < S

        q_ptrs = base + off_m[:, None] * d + off_d[None, :]
        q_mask = mask_m[:, None] & mask_d[None, :]

        # Load query-side tiles — bf16 [M, D]
        Q_m = tl.load(Q_ptr + q_ptrs, mask=q_mask, other=0.0)
        dO_m = tl.load(dO_ptr + q_ptrs, mask=q_mask, other=0.0)
        O_m = tl.load(O_ptr + q_ptrs, mask=q_mask, other=0.0)

        # Trace correction: D_m[i] = sum_k(dO_m[i,k] * O_m[i,k]) — fp32 [M]
        D_m = tl.sum(dO_m.to(tl.float32) * O_m.to(tl.float32), axis=1)

        # Attention scores: [M, N] fp32  (both bf16 inputs → fp32 accumulator)
        S_mat = tl.dot(Q_m, K_T)
        S_mat = S_mat * scale

        # Load logsumexp: [M] fp32
        L_m = tl.load(L_ptr + L_base + off_m, mask=mask_m, other=0.0).to(tl.float32)

        # Attention weights: [M, N] fp32
        P = tl.exp(S_mat - L_m[:, None])

        # Gradient of loss wrt attention: dP[m,n] = sum_k(dO_m[m,k] * V[n,k])
        # Both bf16 → fp32 accumulator [M, N]
        dP = tl.dot(dO_m, V_T)

        # Softmax gradient + chain rule for scale: [M, N] fp32
        dS = P * (dP - D_m[:, None]) * scale

        # Accumulate dV: dV_n += P^T @ dO_m   [N, D]
        dV_acc = tl.dot(P.to(tl.bfloat16).T, dO_m, acc=dV_acc)

        # Accumulate dK: dK_n += dS^T @ Q_m   [N, D]
        dK_acc = tl.dot(dS.to(tl.bfloat16).T, Q_m, acc=dK_acc)

    # Write dK and dV back as bf16
    tl.store(dK_ptr + kv_ptrs, dK_acc.to(tl.bfloat16), mask=kv_mask)
    tl.store(dV_ptr + kv_ptrs, dV_acc.to(tl.bfloat16), mask=kv_mask)


@triton.jit
def _dQ_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
    dQ_ptr,
    S, d, scale, num_n_blocks,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Computes dQ for one (batch*head, query_sequence_block) assignment.
    Iterates over all KV blocks to accumulate gradient.
    """
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    base = pid_bh * S * d

    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = off_m < S

    off_d = tl.arange(0, BLOCK_D)
    mask_d = off_d < d

    q_ptrs = base + off_m[:, None] * d + off_d[None, :]
    q_mask = mask_m[:, None] & mask_d[None, :]

    # Load query-side tiles — bf16 [M, D]
    Q_m = tl.load(Q_ptr + q_ptrs, mask=q_mask, other=0.0)
    dO_m = tl.load(dO_ptr + q_ptrs, mask=q_mask, other=0.0)
    O_m = tl.load(O_ptr + q_ptrs, mask=q_mask, other=0.0)

    # Trace correction: [M] fp32
    D_m = tl.sum(dO_m.to(tl.float32) * O_m.to(tl.float32), axis=1)

    # Logsumexp: [M] fp32
    L_m = tl.load(L_ptr + pid_bh * S + off_m, mask=mask_m, other=0.0).to(tl.float32)

    # Initialize dQ accumulator in fp32
    dQ_acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for pid_n in range(num_n_blocks):
        off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = off_n < S

        # Load K, V tiles: [BLOCK_N, BLOCK_D] — bf16
        kv_ptrs = base + off_n[:, None] * d + off_d[None, :]
        kv_mask = mask_n[:, None] & mask_d[None, :]

        K_n = tl.load(K_ptr + kv_ptrs, mask=kv_mask, other=0.0)
        V_n = tl.load(V_ptr + kv_ptrs, mask=kv_mask, other=0.0)

        # Attention scores: [M, N] fp32 (both bf16 → fp32 accumulator)
        S_mat = tl.dot(Q_m, K_n.T)
        S_mat = S_mat * scale

        # Attention weights: [M, N] fp32
        P = tl.exp(S_mat - L_m[:, None])

        # Loss gradient wrt attention: [M, N] fp32 (both bf16 → fp32 accumulator)
        dP = tl.dot(dO_m, V_n.T)

        # Softmax gradient + chain rule: [M, N] fp32
        dS = P * (dP - D_m[:, None]) * scale

        # Accumulate dQ: dQ_m += dS @ K_n   [M, D]
        dQ_acc = tl.dot(dS.to(tl.bfloat16), K_n, acc=dQ_acc)

    # Write dQ back as bf16
    tl.store(dQ_ptr + q_ptrs, dQ_acc.to(tl.bfloat16), mask=q_mask)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Multi-head attention backward pass.
    
    Inputs (definition order):
        Q  : [B, H, S, d] bf16 — queries
        K  : [B, H, S, d] bf16 — keys
        V  : [B, H, S, d] bf16 — values
        O  : [B, H, S, d] bf16 — forward output
        dO : [B, H, S, d] bf16 — upstream gradient
        L  : [B, H, S]   fp32 — logsumexp of QK^T/sqrt(d)
    
    Outputs (preallocated, written in place):
        dQ : [B, H, S, d] bf16 — gradient w.r.t. Q
        dK : [B, H, S, d] bf16 — gradient w.r.t. K
        dV : [B, H, S, d] bf16 — gradient w.r.t. V
    """
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    scale = 1.0 / math.sqrt(d)

    # Tile configuration
    BLOCK_M = 64   # query sequence tile
    BLOCK_N = 64   # KV sequence tile
    BLOCK_D = 128  # head dimension tile (matches d=128)

    num_bh = B * H
    num_m_blocks = triton.cdiv(S, BLOCK_M)
    num_n_blocks = triton.cdiv(S, BLOCK_N)

    # --- dKV kernel: each CTA owns one (bh, kv_block), accumulates dK and dV ---
    grid_dkv = (num_bh, num_n_blocks)
    _dKV_kernel[grid_dkv](
        Q, K, V, O, dO, L,
        dK, dV,
        S, d, scale, num_m_blocks,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=8, num_stages=2,
    )

    # --- dQ kernel: each CTA owns one (bh, q_block), accumulates dQ ---
    grid_dq = (num_bh, num_m_blocks)
    _dQ_kernel[grid_dq](
        Q, K, V, O, dO, L,
        dQ,
        S, d, scale, num_n_blocks,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=8, num_stages=2,
    )