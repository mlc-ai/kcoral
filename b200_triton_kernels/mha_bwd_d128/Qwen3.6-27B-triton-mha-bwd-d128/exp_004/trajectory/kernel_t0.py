import torch
import triton
import triton.language as tl


@triton.jit
def _dkdv_kernel(
    Q, K, V, O, dO, L,
    dK, dV,
    B, H, S, d,
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute dK and dV.

    Grid: (B*H, num_kv_tiles)
    Each program owns one (B,H) and one KV-sequence tile, looping over all Q tiles.
    """
    pid_bh = tl.program_id(0)
    pid_n = tl.program_id(1)

    bh_base = pid_bh * S * d

    # Index arrays for this KV tile
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_D)

    K_mask = (offs_n[:, None] < S) & (offs_k[None, :] < d)
    ptr_nk = bh_base + offs_n[:, None] * d + offs_k[None, :]

    # Load K and V tiles once (reused across all Q-tile iterations)
    K_tile = tl.load(K + ptr_nk, mask=K_mask, other=0.0).to(tl.float32)
    V_tile = tl.load(V + ptr_nk, mask=K_mask, other=0.0).to(tl.float32)

    acc_dk = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    acc_dv = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)

    num_q_tiles = tl.cdiv(S, BLOCK_M)
    for pm in range(num_q_tiles):
        offs_m = pm * BLOCK_M + tl.arange(0, BLOCK_M)
        Q_mask = (offs_m[:, None] < S) & (offs_k[None, :] < d)
        ptr_mk = bh_base + offs_m[:, None] * d + offs_k[None, :]

        Q_tile = tl.load(Q + ptr_mk, mask=Q_mask, other=0.0).to(tl.float32)
        dO_tile = tl.load(dO + ptr_mk, mask=Q_mask, other=0.0).to(tl.float32)
        O_tile = tl.load(O + ptr_mk, mask=Q_mask, other=0.0).to(tl.float32)

        mask_m1 = offs_m < S
        L_block = tl.load(L + pid_bh * S + offs_m, mask=mask_m1, other=0.0)

        # D[i] = dot(dO[i,:], O[i,:])  per query row
        D = tl.sum(dO_tile * O_tile, axis=1)

        # Forward quantities (recomputed for attention backward)
        S_mat = tl.dot(Q_tile, K_tile.T) * sm_scale              # [BM, BN]
        P = tl.exp(S_mat - L_block[:, None])                     # [BM, BN]
        dP = tl.dot(dO_tile, V_tile.T)                           # [BM, BN]
        dS_raw = P * (dP - D[:, None]) * sm_scale                # [BM, BN]

        # Accumulate gradients
        acc_dk = tl.dot(dS_raw.T, Q_tile, acc=acc_dk)            # [BN, BD]
        acc_dv = tl.dot(P.T, dO_tile, acc=acc_dv)                # [BN, BD]

    tl.store(dK + ptr_nk, acc_dk.to(tl.bfloat16), mask=K_mask)
    tl.store(dV + ptr_nk, acc_dv.to(tl.bfloat16), mask=K_mask)


@triton.jit
def _dq_kernel(
    Q, K, V, O, dO, L,
    dQ,
    B, H, S, d,
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute dQ.

    Grid: (B*H, num_q_tiles)
    Each program owns one (B,H) and one Q-sequence tile, looping over all KV tiles.
    """
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    bh_base = pid_bh * S * d

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_D)

    Q_mask = (offs_m[:, None] < S) & (offs_k[None, :] < d)
    ptr_mk = bh_base + offs_m[:, None] * d + offs_k[None, :]

    # Load Q-side data once (reused across all KV-tile iterations)
    Q_tile = tl.load(Q + ptr_mk, mask=Q_mask, other=0.0).to(tl.float32)
    dO_tile = tl.load(dO + ptr_mk, mask=Q_mask, other=0.0).to(tl.float32)
    O_tile = tl.load(O + ptr_mk, mask=Q_mask, other=0.0).to(tl.float32)

    mask_m1 = offs_m < S
    L_block = tl.load(L + pid_bh * S + offs_m, mask=mask_m1, other=0.0)

    D = tl.sum(dO_tile * O_tile, axis=1)

    acc_dq = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    num_kv_tiles = tl.cdiv(S, BLOCK_N)
    for pn in range(num_kv_tiles):
        offs_n = pn * BLOCK_N + tl.arange(0, BLOCK_N)
        KV_mask = (offs_n[:, None] < S) & (offs_k[None, :] < d)
        ptr_nk = bh_base + offs_n[:, None] * d + offs_k[None, :]

        K_tile = tl.load(K + ptr_nk, mask=KV_mask, other=0.0).to(tl.float32)
        V_tile = tl.load(V + ptr_nk, mask=KV_mask, other=0.0).to(tl.float32)

        S_mat = tl.dot(Q_tile, K_tile.T) * sm_scale
        P = tl.exp(S_mat - L_block[:, None])
        dP = tl.dot(dO_tile, V_tile.T)
        dS_raw = P * (dP - D[:, None]) * sm_scale

        acc_dq = tl.dot(dS_raw, K_tile, acc=acc_dq)               # [BM, BD]

    tl.store(dQ + ptr_mk, acc_dq.to(tl.bfloat16), mask=Q_mask)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute multi-head attention backward: dQ, dK, dV."""
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    sm_scale = 1.0 / (d ** 0.5)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = d  # d == 128, fits in a single tile

    BH = B * H
    num_kv_tiles = triton.cdiv(S, BLOCK_N)
    num_q_tiles = triton.cdiv(S, BLOCK_M)

    # --- Pass 1: compute dK and dV ---
    grid_dkdv = (BH, num_kv_tiles)
    _dkdv_kernel[grid_dkdv](
        Q, K, V, O, dO, L,
        dK, dV,
        B, H, S, d, sm_scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=4, num_stages=3,
    )

    # --- Pass 2: compute dQ ---
    grid_dq = (BH, num_q_tiles)
    _dq_kernel[grid_dq](
        Q, K, V, O, dO, L,
        dQ,
        B, H, S, d, sm_scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=4, num_stages=3,
    )