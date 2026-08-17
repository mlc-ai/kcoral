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
    """
    Compute dQ for causal MHA backward.
    Each program handles one (b, h) pair and one query tile of size BLOCK_M.
    Sweeps over all key tiles, accumulating dQ.
    """
    b = tl.program_id(0)
    h = tl.program_id(1)
    m_tile = tl.program_id(2)

    # Base element offsets for this (b, h) pair
    kv_offset = b * stride_B + h * stride_H
    L_offset = b * stride_L_B + h * stride_L_H

    # Query row offsets for this tile
    offs_m = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < S_len

    # Dimension and key template offsets
    offs_d = tl.arange(0, D_DIM)
    offs_n = tl.arange(0, BLOCK_N)

    # Output accumulator: [BLOCK_M, D_DIM]
    dQ_acc = tl.zeros((BLOCK_M, D_DIM), dtype=tl.float32)

    # Pointer offsets for query-side tiles [BLOCK_M, D_DIM]
    ptrs_m = offs_m[:, None] * stride_S + offs_d[None, :] * stride_D

    # Load invariant query-side tiles (used across all K iterations)
    Q_tile = tl.load(Q_ptr + kv_offset + ptrs_m, mask=mask_m[:, None], other=0.0)
    dO_tile = tl.load(dO_ptr + kv_offset + ptrs_m, mask=mask_m[:, None], other=0.0)
    O_tile = tl.load(O_ptr + kv_offset + ptrs_m, mask=mask_m[:, None], other=0.0)

    # Load logsumexp and compute D = row_dot(dO, O) for the softmax gradient
    L_vals = tl.load(L_ptr + L_offset + offs_m * stride_L_S, mask=mask_m, other=0.0)
    D_vals = tl.sum(dO_tile * O_tile, axis=1)

    # Sweep over all key position tiles
    num_k_tiles = tl.cdiv(S_len, BLOCK_N)
    for k_idx in range(num_k_tiles):
        n_start = k_idx * BLOCK_N
        offs_n_cur = n_start + offs_n
        mask_n = offs_n_cur < S_len

        # Pointer offsets for key-side tiles [BLOCK_N, D_DIM]
        ptrs_n = offs_n_cur[:, None] * stride_S + offs_d[None, :] * stride_D

        # Load K and V tiles
        K_tile = tl.load(K_ptr + kv_offset + ptrs_n, mask=mask_n[:, None], other=0.0)
        V_tile = tl.load(V_ptr + kv_offset + ptrs_n, mask=mask_n[:, None], other=0.0)

        # Recompute attention scores: [BLOCK_M, BLOCK_N]
        scores = tl.dot(Q_tile, K_tile.T) * scale

        # Causal mask: query_pos >= key_pos allows attention
        causal = (offs_m[:, None] >= offs_n_cur[None, :]).to(tl.float32)

        # Reconstruct probability matrix: P = exp(scaled_scores - L) * causal
        P = tl.exp(tl.minimum(scores - L_vals[:, None], 0.0)) * causal

        # Gradient through softmax: dS = scale * P * (dO @ V^T - D)
        dP = tl.dot(dO_tile, V_tile.T)
        dS = P * (dP - D_vals[:, None]) * scale

        # Accumulate: dQ += dS @ K
        dQ_acc += tl.dot(dS, K_tile)

    # Store final dQ result
    tl.store(dQ_ptr + kv_offset + ptrs_m, dQ_acc.to(tl.bfloat16), mask=mask_m[:, None])


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
    """
    Compute dK and dV for causal MHA backward.
    Each program handles one (b, h) pair and one key tile of size BLOCK_N.
    Sweeps over all query tiles, accumulating dK and dV.
    """
    b = tl.program_id(0)
    h = tl.program_id(1)
    n_tile = tl.program_id(2)

    # Base element offsets for this (b, h) pair
    kv_offset = b * stride_B + h * stride_H
    L_offset = b * stride_L_B + h * stride_L_H

    # Key row offsets (fixed for this program)
    offs_n = n_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < S_len

    # Dimension and query template offsets
    offs_d = tl.arange(0, D_DIM)
    offs_m = tl.arange(0, BLOCK_M)

    # Output accumulators: [BLOCK_N, D_DIM]
    dK_acc = tl.zeros((BLOCK_N, D_DIM), dtype=tl.float32)
    dV_acc = tl.zeros((BLOCK_N, D_DIM), dtype=tl.float32)

    # Pointer offsets for key-side tiles [BLOCK_N, D_DIM]
    ptrs_n = offs_n[:, None] * stride_S + offs_d[None, :] * stride_D

    # Load invariant key-side tiles (used across all Q iterations)
    K_tile = tl.load(K_ptr + kv_offset + ptrs_n, mask=mask_n[:, None], other=0.0)
    V_tile = tl.load(V_ptr + kv_offset + ptrs_n, mask=mask_n[:, None], other=0.0)

    # Sweep over all query position tiles
    num_q_tiles = tl.cdiv(S_len, BLOCK_M)
    for m_idx in range(num_q_tiles):
        m_start = m_idx * BLOCK_M
        offs_m_cur = m_start + offs_m
        mask_m = offs_m_cur < S_len

        # Pointer offsets for query-side tiles [BLOCK_M, D_DIM]
        ptrs_m = offs_m_cur[:, None] * stride_S + offs_d[None, :] * stride_D

        # Load Q, dO, O tiles for this query tile
        Q_tile = tl.load(Q_ptr + kv_offset + ptrs_m, mask=mask_m[:, None], other=0.0)
        dO_tile = tl.load(dO_ptr + kv_offset + ptrs_m, mask=mask_m[:, None], other=0.0)
        O_tile = tl.load(O_ptr + kv_offset + ptrs_m, mask=mask_m[:, None], other=0.0)

        # Load logsumexp and compute D for softmax gradient
        L_vals = tl.load(L_ptr + L_offset + offs_m_cur * stride_L_S, mask=mask_m, other=0.0)
        D_vals = tl.sum(dO_tile * O_tile, axis=1)

        # Recompute attention scores: [BLOCK_M, BLOCK_N]
        scores = tl.dot(Q_tile, K_tile.T) * scale

        # Causal mask: query_pos >= key_pos allows attention
        causal = (offs_m_cur[:, None] >= offs_n[None, :]).to(tl.float32)

        # Reconstruct probability matrix
        P = tl.exp(tl.minimum(scores - L_vals[:, None], 0.0)) * causal

        # Gradient through softmax
        dP = tl.dot(dO_tile, V_tile.T)
        dS = P * (dP - D_vals[:, None]) * scale

        # Accumulate: dK += dS^T @ Q, dV += P^T @ dO
        dK_acc += tl.dot(dS.T, Q_tile)
        dV_acc += tl.dot(P.T, dO_tile)

    # Store final results
    tl.store(dK_ptr + kv_offset + ptrs_n, dK_acc.to(tl.bfloat16), mask=mask_n[:, None])
    tl.store(dV_ptr + kv_offset + ptrs_n, dV_acc.to(tl.bfloat16), mask=mask_n[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Causal multi-head attention backward pass.
    Computes dQ, dK, dV given Q, K, V, O (forward output), dO (upstream grad),
    and L (logsumexp of scaled attention scores from forward pass).

    Args:
        Q: [B, H, S, d] bf16 query tensor
        K: [B, H, S, d] bf16 key tensor
        V: [B, H, S, d] bf16 value tensor
        O: [B, H, S, d] bf16 forward attention output
        dO: [B, H, S, d] bf16 upstream gradient
        L: [B, H, S] float32 logsumexp statistics
        dQ: [B, H, S, d] bf16 preallocated output for dQ
        dK: [B, H, S, d] bf16 preallocated output for dK
        dV: [B, H, S, d] bf16 preallocated output for dV
    """
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape

    # Attention scale factor: 1/sqrt(head_dim)
    scale = 1.0 / math.sqrt(d)

    # Extract strides from tensors (works for any layout)
    qs = Q.stride()  # (stride_B, stride_H, stride_S, stride_D)
    ls = L.stride()  # (stride_B, stride_H, stride_S) for [B, H, S]

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