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
    """Compute dQ for causal MHA backward."""
    b = tl.program_id(0)
    h = tl.program_id(1)
    m_tile = tl.program_id(2)

    # Base element offset for this (b, h) pair
    kv_offset = b * stride_B + h * stride_H
    l_offset = b * stride_L_B + h * stride_L_H

    # Query row offsets for this tile
    offs_m = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < S_len

    # Dimension template offsets
    offs_d = tl.arange(0, D_DIM)

    # Output accumulator: [BLOCK_M, D_DIM] in fp32
    dQ_acc = tl.zeros((BLOCK_M, D_DIM), dtype=tl.float32)

    # Matrix tile offsets for query-side [BLOCK_M, D_DIM]
    ptrs_m = offs_m[:, None] * stride_S + offs_d[None, :] * stride_D

    # Load invariant query-side tiles
    Q_tile = tl.load(Q_ptr + kv_offset + ptrs_m,
                      mask=mask_m[:, None], other=0.0)
    dO_tile = tl.load(dO_ptr + kv_offset + ptrs_m,
                       mask=mask_m[:, None], other=0.0)
    O_tile = tl.load(O_ptr + kv_offset + ptrs_m,
                      mask=mask_m[:, None], other=0.0)

    # Cast tiles to fp32 for stable computation
    Q_f = Q_tile.to(tl.float32)
    dO_f = dO_tile.to(tl.float32)
    O_f = O_tile.to(tl.float32)

    # Load logsumexp and compute D = row_dot(dO, O)
    L_ptrs = L_ptr + l_offset + offs_m * stride_L_S
    L_vals = tl.load(L_ptrs, mask=mask_m, other=0.0)
    D_vals = tl.sum(dO_f * O_f, axis=1)

    # Sweep over all key position tiles
    num_k_tiles = tl.cdiv(S_len, BLOCK_N)
    for k_idx in range(num_k_tiles):
        n_start = k_idx * BLOCK_N
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S_len

        # Matrix tile offsets for key-side [BLOCK_N, D_DIM]
        ptrs_n = offs_n[:, None] * stride_S + offs_d[None, :] * stride_D

        # Load K and V tiles
        K_tile = tl.load(K_ptr + kv_offset + ptrs_n,
                          mask=mask_n[:, None], other=0.0)
        V_tile = tl.load(V_ptr + kv_offset + ptrs_n,
                          mask=mask_n[:, None], other=0.0)

        K_f = K_tile.to(tl.float32)
        V_f = V_tile.to(tl.float32)

        # Recompute attention scores: [BLOCK_M, BLOCK_N]
        scores = tl.dot(Q_f, K_f.T) * scale

        # Causal mask: query_pos >= key_pos allows attention
        causal_mask = (offs_m[:, None] >= offs_n[None, :]).to(tl.float32)

        # Reconstruct probability matrix: P = exp(scaled_scores - L) * causal
        P = tl.exp(tl.minimum(scores - L_vals[:, None], 0.0)) * causal_mask

        # Gradient through softmax: dS = scale * P * (dO @ V^T - D)
        dP = tl.dot(dO_f, V_f.T)
        dS = P * (dP - D_vals[:, None]) * scale

        # Accumulate: dQ += dS @ K
        dQ_acc = tl.dot(dS, K_f, dQ_acc)

    # Store final dQ result back to bf16
    out_ptrs = dQ_ptr + kv_offset + ptrs_m
    tl.store(out_ptrs, dQ_acc.to(tl.bfloat16), mask=mask_m[:, None])


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
    """Compute dK and dV for causal MHA backward."""
    b = tl.program_id(0)
    h = tl.program_id(1)
    n_tile = tl.program_id(2)

    # Base element offset for this (b, h) pair
    kv_offset = b * stride_B + h * stride_H
    l_offset = b * stride_L_B + h * stride_L_H

    # Key row offsets (fixed for this program)
    offs_n = n_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < S_len

    # Dimension template offsets
    offs_d = tl.arange(0, D_DIM)

    # Output accumulators: [BLOCK_N, D_DIM] in fp32
    dK_acc = tl.zeros((BLOCK_N, D_DIM), dtype=tl.float32)
    dV_acc = tl.zeros((BLOCK_N, D_DIM), dtype=tl.float32)

    # Matrix tile offsets for key-side [BLOCK_N, D_DIM]
    ptrs_n = offs_n[:, None] * stride_S + offs_d[None, :] * stride_D

    # Load invariant key-side tiles
    K_tile = tl.load(K_ptr + kv_offset + ptrs_n,
                      mask=mask_n[:, None], other=0.0)
    V_tile = tl.load(V_ptr + kv_offset + ptrs_n,
                      mask=mask_n[:, None], other=0.0)

    K_f = K_tile.to(tl.float32)
    V_f = V_tile.to(tl.float32)

    # Sweep over all query position tiles
    num_q_tiles = tl.cdiv(S_len, BLOCK_M)
    for m_idx in range(num_q_tiles):
        m_start = m_idx * BLOCK_M
        offs_m = m_start + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S_len

        # Matrix tile offsets for query-side [BLOCK_M, D_DIM]
        ptrs_m = offs_m[:, None] * stride_S + offs_d[None, :] * stride_D

        # Load Q, dO, O tiles for this query tile
        Q_tile = tl.load(Q_ptr + kv_offset + ptrs_m,
                          mask=mask_m[:, None], other=0.0)
        dO_tile = tl.load(dO_ptr + kv_offset + ptrs_m,
                           mask=mask_m[:, None], other=0.0)
        O_tile = tl.load(O_ptr + kv_offset + ptrs_m,
                          mask=mask_m[:, None], other=0.0)

        Q_f = Q_tile.to(tl.float32)
        dO_f = dO_tile.to(tl.float32)
        O_f = O_tile.to(tl.float32)

        # Load logsumexp and compute D for softmax gradient
        L_ptrs = L_ptr + l_offset + offs_m * stride_L_S
        L_vals = tl.load(L_ptrs, mask=mask_m, other=0.0)
        D_vals = tl.sum(dO_f * O_f, axis=1)

        # Recompute attention scores: [BLOCK_M, BLOCK_N]
        scores = tl.dot(Q_f, K_f.T) * scale

        # Causal mask: query_pos >= key_pos allows attention
        causal_mask = (offs_m[:, None] >= offs_n[None, :]).to(tl.float32)

        # Reconstruct probability matrix
        P = tl.exp(tl.minimum(scores - L_vals[:, None], 0.0)) * causal_mask

        # Gradient through softmax
        dP = tl.dot(dO_f, V_f.T)
        dS = P * (dP - D_vals[:, None]) * scale

        # Accumulate: dK += dS^T @ Q, dV += P^T @ dO
        dK_acc = tl.dot(dS.T, Q_f, dK_acc)
        dV_acc = tl.dot(P.T, dO_f, dV_acc)

    # Store final results
    tl.store(dK_ptr + kv_offset + ptrs_n, dK_acc.to(tl.bfloat16), mask=mask_n[:, None])
    tl.store(dV_ptr + kv_offset + ptrs_n, dV_acc.to(tl.bfloat16), mask=mask_n[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Causal multi-head attention backward pass."""
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape

    # Attention scale factor: 1/sqrt(head_dim)
    scale = 1.0 / math.sqrt(d)

    # Extract strides from tensors (works for any layout)
    qs = Q.stride()
    ls = L.stride()

    # Tile sizes (must be powers of 2)
    BLOCK_M = 64
    BLOCK_N = 64

    # Kernel 1: compute dQ — grid is (B, H, num_query_tiles)
    grid_dq = (B, H, triton.cdiv(S, BLOCK_M))
    _bwd_dq_kernel[grid_dq](
        Q, K, V, O, dO, L, dQ,
        S, scale,
        qs[0], qs[1], qs[2], qs[3],
        ls[0], ls[1], ls[2],
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, D_DIM=d,
        num_warps=4, num_stages=2,
    )

    # Kernel 2: compute dK and dV — grid is (B, H, num_key_tiles)
    grid_dkv = (B, H, triton.cdiv(S, BLOCK_N))
    _bwd_dkv_kernel[grid_dkv](
        Q, K, V, O, dO, L, dK, dV,
        S, scale,
        qs[0], qs[1], qs[2], qs[3],
        ls[0], ls[1], ls[2],
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, D_DIM=d,
        num_warps=4, num_stages=2,
    )