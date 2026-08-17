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

    # Base element offset for this (b, h) pair
    kv_base = b * stride_B + h * stride_H
    l_base = b * stride_L_B + h * stride_L_H

    # Query row offsets for this tile
    offs_m = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < S_len

    # Dimension template offsets
    offs_d = tl.arange(0, D_DIM)

    # Output accumulator: [BLOCK_M, D_DIM] in fp32
    dQ_acc = tl.zeros((BLOCK_M, D_DIM), dtype=tl.float32)

    # Matrix tile offsets for query-side [BLOCK_M, D_DIM]
    qm_offsets = offs_m[:, None] * stride_S + offs_d[None, :] * stride_D

    # Load invariant query-side tiles
    Q_tile = tl.load(Q_ptr + kv_base + qm_offsets,
                      mask=mask_m[:, None], other=0.0, eviction_policy="evict_last")
    dO_tile = tl.load(dO_ptr + kv_base + qm_offsets,
                       mask=mask_m[:, None], other=0.0, eviction_policy="evict_last")
    O_tile = tl.load(O_ptr + kv_base + qm_offsets,
                      mask=mask_m[:, None], other=0.0, eviction_policy="evict_last")

    # Cast tiles to fp32 for stable computation
    Q_f = Q_tile.to(tl.float32)
    dO_f = dO_tile.to(tl.float32)
    O_f = O_tile.to(tl.float32)

    # Load logsumexp and compute D = row_dot(dO, O)
    L_ptrs = L_ptr + l_base + offs_m * stride_L_S
    L_vals = tl.load(L_ptrs, mask=mask_m, other=0.0)
    D_vals = tl.sum(dO_f * O_f, axis=1)

    num_k_tiles = tl.cdiv(S_len, BLOCK_N)
    for k_idx in range(num_k_tiles):
        n_start = k_idx * BLOCK_N
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S_len

        # Matrix tile offsets for key-side [BLOCK_N, D_DIM]
        kn_offsets = offs_n[:, None] * stride_S + offs_d[None, :] * stride_D

        # Load K and V tiles
        K_tile = tl.load(K_ptr + kv_base + kn_offsets,
                          mask=mask_n[:, None], other=0.0, eviction_policy="evict_first")
        V_tile = tl.load(V_ptr + kv_base + kn_offsets,
                          mask=mask_n[:, None], other=0.0, eviction_policy="evict_first")

        K_f = K_tile.to(tl.float32)
        V_f = V_tile.to(tl.float32)

        # Recompute attention scores: [BLOCK_M, BLOCK_N]
        scores = tl.dot(Q_f, K_f.T) * scale

        # Causal mask: query_pos >= key_pos allows attention
        causal_mask = (offs_m[:, None] >= offs_n[None, :]).to(tl.float32)

        # Clamp before exp for numerical stability; L is logsumexp of valid region
        P = tl.exp(tl.minimum(scores - L_vals[:, None], 0.0)) * causal_mask

        # Gradient through attn+softmax: dS = scale * P * (dO @ V^T - D)
        dP = tl.dot(dO_f, V_f.T)
        dS = P * (dP - D_vals[:, None]) * scale

        # Accumulate: dQ += dS @ K
        dQ_acc = tl.dot(dS, K_f, dQ_acc)

    # Store final dQ result back to bf16
    out_ptrs = dQ_ptr + kv_base + qm_offsets
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
    """
    Compute dK and dV for causal MHA backward.
    Each program handles one (b, h) pair and one key tile of size BLOCK_N.
    Sweeps over all query tiles, accumulating dK and dV.
    """
    b = tl.program_id(0)
    h = tl.program_id(1)
    n_tile = tl.program_id(2)

    # Base element offset for this (b, h) pair
    kv_base = b * stride_B + h * stride_H
    l_base = b * stride_L_B + h * stride_L_H

    # Key row offsets (fixed for this program)
    offs_n = n_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < S_len

    # Dimension template offsets
    offs_d = tl.arange(0, D_DIM)

    # Output accumulators: [BLOCK_N, D_DIM] in fp32
    dK_acc = tl.zeros((BLOCK_N, D_DIM), dtype=tl.float32)
    dV_acc = tl.zeros((BLOCK_N, D_DIM), dtype=tl.float32)

    # Matrix tile offsets for key-side [BLOCK_N, D_DIM]
    kn_offsets = offs_n[:, None] * stride_S + offs_d[None, :] * stride_D

    # Load invariant key-side tiles
    K_tile = tl.load(K_ptr + kv_base + kn_offsets,
                      mask=mask_n[:, None], other=0.0, eviction_policy="evict_last")
    V_tile = tl.load(V_ptr + kv_base + kn_offsets,
                      mask=mask_n[:, None], other=0.0, eviction_policy="evict_last")

    K_f = K_tile.to(tl.float32)
    V_f = V_tile.to(tl.float32)

    num_q_tiles = tl.cdiv(S_len, BLOCK_M)
    for m_idx in range(num_q_tiles):
        m_start = m_idx * BLOCK_M
        offs_m = m_start + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S_len

        # Matrix tile offsets for query-side [BLOCK_M, D_DIM]
        qm_offsets = offs_m[:, None] * stride_S + offs_d[None, :] * stride_D

        # Load Q, dO, O tiles
        Q_tile = tl.load(Q_ptr + kv_base + qm_offsets,
                          mask=mask_m[:, None], other=0.0, eviction_policy="evict_first")
        dO_tile = tl.load(dO_ptr + kv_base + qm_offsets,
                           mask=mask_m[:, None], other=0.0, eviction_policy="evict_first")
        O_tile = tl.load(O_ptr + kv_base + qm_offsets,
                          mask=mask_m[:, None], other=0.0, eviction_policy="evict_first")

        Q_f = Q_tile.to(tl.float32)
        dO_f = dO_tile.to(tl.float32)
        O_f = O_tile.to(tl.float32)

        # Load logsumexp and compute D
        L_ptrs = L_ptr + l_base + offs_m * stride_L_S
        L_vals = tl.load(L_ptrs, mask=mask_m, other=0.0)
        D_vals = tl.sum(dO_f * O_f, axis=1)

        # Recompute attention scores: [BLOCK_M, BLOCK_N]
        scores = tl.dot(Q_f, K_f.T) * scale

        # Causal mask: query_pos >= key_pos
        causal_mask = (offs_m[:, None] >= offs_n[None, :]).to(tl.float32)

        # Reconstruct probability matrix
        P = tl.exp(tl.minimum(scores - L_vals[:, None], 0.0)) * causal_mask

        # Gradient through softmax
        dP = tl.dot(dO_f, V_f.T)
        dS = P * (dP - D_vals[:, None]) * scale

        # Accumulate: dK += dS^T @ Q, dV += P^T @ dO
        dK_acc = tl.dot(dS.T, Q_f, dK_acc)
        dV_acc = tl.dot(P.T, dO_f, dV_acc)

    # Store results
    dk_ptrs = dK_ptr + kv_base + kn_offsets
    dv_ptrs = dV_ptr + kv_base + kn_offsets
    tl.store(dk_ptrs, dK_acc.to(tl.bfloat16), mask=mask_n[:, None])
    tl.store(dv_ptrs, dV_acc.to(tl.bfloat16), mask=mask_n[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Causal multi-head attention backward pass.
    
    Args (destination-passing order):
        Q, K, V, O, dO, L  - inputs in definition order
        dQ, dK, dV          - preallocated outputs in definition order
    """
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape

    scale = 1.0 / math.sqrt(d)

    qs = Q.stride()
    ls = L.stride()

    BLOCK_M = 64
    BLOCK_N = 64

    grid_dq = (B, H, triton.cdiv(S, BLOCK_M))
    _bwd_dq_kernel[grid_dq](
        Q, K, V, O, dO, L, dQ,
        S, scale,
        qs[0], qs[1], qs[2], qs[3],
        ls[0], ls[1], ls[2],
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, D_DIM=d,
        num_warps=4, num_stages=2,
    )

    grid_dkv = (B, H, triton.cdiv(S, BLOCK_N))
    _bwd_dkv_kernel[grid_dkv](
        Q, K, V, O, dO, L, dK, dV,
        S, scale,
        qs[0], qs[1], qs[2], qs[3],
        ls[0], ls[1], ls[2],
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, D_DIM=d,
        num_warps=4, num_stages=2,
    )