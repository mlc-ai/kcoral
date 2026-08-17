import math
import torch
import triton
import triton.language as tl


@triton.jit
def _bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dQ_ptr,
    S_len,
    scale,
    stride_B, stride_H, stride_S, stride_D,
    stride_L_B, stride_L_H, stride_L_S,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D_DIM: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    m_tile = tl.program_id(2)

    kv_base = b * stride_B + h * stride_H
    l_base = b * stride_L_B + h * stride_L_H

    # Row offsets for query tile: [BLOCK_M]
    row_m = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    col_d = tl.arange(0, D_DIM)  # [D_DIM]

    # Mask for valid rows
    row_mask = (row_m < S_len).to(tl.int1)[:, None]  # [BLOCK_M, 1]

    # Load invariant query-side tiles into fp32
    qm_offs = row_m[:, None] * stride_S + col_d[None, :] * stride_D  # [BLOCK_M, D]
    Q_f = tl.load(Q_ptr + kv_base + qm_offs, mask=row_mask, other=0.0,
                  eviction_policy="evict_last").to(tl.float32)
    dO_f = tl.load(dO_ptr + kv_base + qm_offs, mask=row_mask, other=0.0,
                   eviction_policy="evict_last").to(tl.float32)
    O_f = tl.load(O_ptr + kv_base + qm_offs, mask=row_mask, other=0.0,
                  eviction_policy="evict_last").to(tl.float32)

    # Logsumexp and D = sum(dO*O, axis=1)
    L_vals = tl.load(L_ptr + l_base + row_m * stride_L_S, mask=(row_m < S_len), other=0.0,
                     eviction_policy="evict_last")  # [BLOCK_M]
    D_vals = tl.sum(dO_f * O_f, axis=1)  # [BLOCK_M]

    dQ_acc = tl.zeros((BLOCK_M, D_DIM), dtype=tl.float32)

    num_k_tiles = tl.cdiv(S_len, BLOCK_N)
    for k_idx in range(num_k_tiles):
        n_start = k_idx * BLOCK_N
        row_n = n_start + tl.arange(0, BLOCK_N)  # [BLOCK_N]
        n_mask = (row_n < S_len).to(tl.int1)[None, :]  # [1, BLOCK_N]

        kn_offs = row_n[:, None] * stride_S + col_d[None, :] * stride_D  # [BLOCK_N, D]
        K_f = tl.load(K_ptr + kv_base + kn_offs, mask=n_mask.T, other=0.0,
                      eviction_policy="evict_first").to(tl.float32)
        V_f = tl.load(V_ptr + kv_base + kn_offs, mask=n_mask.T, other=0.0,
                      eviction_policy="evict_first").to(tl.float32)

        # scores: [BLOCK_M, BLOCK_N]
        scores = tl.dot(Q_f, K_f.T) * scale

        # Causal mask + probability
        causal = (row_m[:, None] >= row_n[None, :]).to(tl.float32)
        P = tl.exp(tl.minimum(scores - L_vals[:, None], 0.0)) * causal

        # dP: [BLOCK_M, BLOCK_N], dS: same
        dP = tl.dot(dO_f, V_f.T)
        dS = P * (dP - D_vals[:, None]) * scale

        # Accumulate dQ += dS @ K
        dQ_acc = tl.dot(dS, K_f, dQ_acc)

    out_ptrs = dQ_ptr + kv_base + qm_offs
    tl.store(out_ptrs, dQ_acc.to(tl.bfloat16), mask=row_mask)


@triton.jit
def _bwd_dkv_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dK_ptr, dV_ptr,
    S_len,
    scale,
    stride_B, stride_H, stride_S, stride_D,
    stride_L_B, stride_L_H, stride_L_S,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D_DIM: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    n_tile = tl.program_id(2)

    kv_base = b * stride_B + h * stride_H
    l_base = b * stride_L_B + h * stride_L_H

    # Key-side row offsets: [BLOCK_N]
    row_n = n_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    col_d = tl.arange(0, D_DIM)

    n_mask = (row_n < S_len).to(tl.int1)[:, None]

    # Load invariant key-side tiles
    kn_offs = row_n[:, None] * stride_S + col_d[None, :] * stride_D
    K_f = tl.load(K_ptr + kv_base + kn_offs, mask=n_mask, other=0.0,
                  eviction_policy="evict_last").to(tl.float32)
    V_f = tl.load(V_ptr + kv_base + kn_offs, mask=n_mask, other=0.0,
                  eviction_policy="evict_last").to(tl.float32)

    dK_acc = tl.zeros((BLOCK_N, D_DIM), dtype=tl.float32)
    dV_acc = tl.zeros((BLOCK_N, D_DIM), dtype=tl.float32)

    num_q_tiles = tl.cdiv(S_len, BLOCK_M)
    for m_idx in range(num_q_tiles):
        m_start = m_idx * BLOCK_M
        row_m = m_start + tl.arange(0, BLOCK_M)
        m_mask = (row_m < S_len).to(tl.int1)[:, None]

        qm_offs = row_m[:, None] * stride_S + col_d[None, :] * stride_D
        Q_f = tl.load(Q_ptr + kv_base + qm_offs, mask=m_mask, other=0.0,
                      eviction_policy="evict_first").to(tl.float32)
        dO_f = tl.load(dO_ptr + kv_base + qm_offs, mask=m_mask, other=0.0,
                        eviction_policy="evict_first").to(tl.float32)
        O_f = tl.load(O_ptr + kv_base + qm_offs, mask=m_mask, other=0.0,
                       eviction_policy="evict_first").to(tl.float32)

        L_vals = tl.load(L_ptr + l_base + row_m * stride_L_S, mask=(row_m < S_len), other=0.0)
        D_vals = tl.sum(dO_f * O_f, axis=1)

        scores = tl.dot(Q_f, K_f.T) * scale
        causal = (row_m[:, None] >= row_n[None, :]).to(tl.float32)
        P = tl.exp(tl.minimum(scores - L_vals[:, None], 0.0)) * causal

        dP = tl.dot(dO_f, V_f.T)
        dS = P * (dP - D_vals[:, None]) * scale

        dK_acc = tl.dot(dS.T, Q_f, dK_acc)
        dV_acc = tl.dot(P.T, dO_f, dV_acc)

    tl.store(dK_ptr + kv_base + kn_offs, dK_acc.to(tl.bfloat16), mask=n_mask)
    tl.store(dV_ptr + kv_base + kn_offs, dV_acc.to(tl.bfloat16), mask=n_mask)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Causal multi-head attention backward pass."""
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape

    scale = 1.0 / math.sqrt(d)

    qs = Q.stride()
    ls = L.stride()

    # Use larger tiles for better tensor core utilization on Hopper
    BLOCK_M = 128
    BLOCK_N = 128
    NUM_WARPS = 8
    NUM_STAGES = 3

    # Launch both kernels
    grid_dq = (B, H, triton.cdiv(S, BLOCK_M))
    _bwd_dq_kernel[grid_dq](
        Q, K, V, O, dO, L, dQ,
        S, scale,
        qs[0], qs[1], qs[2], qs[3],
        ls[0], ls[1], ls[2],
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, D_DIM=d,
        num_warps=NUM_WARPS, num_stages=NUM_STAGES,
    )

    grid_dkv = (B, H, triton.cdiv(S, BLOCK_N))
    _bwd_dkv_kernel[grid_dkv](
        Q, K, V, O, dO, L, dK, dV,
        S, scale,
        qs[0], qs[1], qs[2], qs[3],
        ls[0], ls[1], ls[2],
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, D_DIM=d,
        num_warps=NUM_WARPS, num_stages=NUM_STAGES,
    )