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
    """Compute dQ for causal MHA backward with optimized pipeline."""
    b = tl.program_id(0)
    h = tl.program_id(1)
    m_tile = tl.program_id(2)

    kv_base = b * stride_B + h * stride_H
    l_base = b * stride_L_B + h * stride_L_H

    offs_m = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m[:, None] < S_len
    offs_d = tl.arange(0, D_DIM)

    # Load invariant query-side tiles once
    qm_offsets = offs_m * stride_S + offs_d[None, :] * stride_D
    Q_f = tl.load(Q_ptr + kv_base + qm_offsets,
                  mask=mask_m, other=0.0, eviction_policy="evict_last").to(tl.float32)
    dO_f = tl.load(dO_ptr + kv_base + qm_offsets,
                   mask=mask_m, other=0.0, eviction_policy="evict_last").to(tl.float32)
    O_f = tl.load(O_ptr + kv_base + qm_offsets,
                  mask=mask_m, other=0.0, eviction_policy="evict_last").to(tl.float32)

    L_vals = tl.load(L_ptr + l_base + offs_m * stride_L_S,
                     mask=offs_m < S_len, other=0.0, eviction_policy="evict_last")
    D_vals = tl.sum(dO_f * O_f, axis=1)

    dQ_acc = tl.zeros((BLOCK_M, D_DIM), dtype=tl.float32)

    num_k_tiles = tl.cdiv(S_len, BLOCK_N)
    for k_idx in range(num_k_tiles):
        n_start = k_idx * BLOCK_N
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n[None, :] < S_len

        kn_offsets = offs_n[:, None] * stride_S + offs_d[None, :] * stride_D
        K_f = tl.load(K_ptr + kv_base + kn_offsets,
                      mask=mask_n.T, other=0.0, eviction_policy="evict_first").to(tl.float32)
        V_f = tl.load(V_ptr + kv_base + kn_offsets,
                      mask=mask_n.T, other=0.0, eviction_policy="evict_first").to(tl.float32)

        scores = tl.dot(Q_f, K_f.T) * scale

        causal_mask = (offs_m[:, None] >= offs_n[None, :]).to(tl.float32)
        P = tl.exp(tl.minimum(scores - L_vals[:, None], 0.0)) * causal_mask

        dP = tl.dot(dO_f, V_f.T)
        dS = P * (dP - D_vals[:, None]) * scale
        dQ_acc = tl.dot(dS, K_f, dQ_acc)

    out_ptrs = dQ_ptr + kv_base + qm_offsets
    tl.store(out_ptrs, dQ_acc.to(tl.bfloat16), mask=mask_m)


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
    """Compute dK and dV for causal MHA backward with optimized pipeline."""
    b = tl.program_id(0)
    h = tl.program_id(1)
    n_tile = tl.program_id(2)

    kv_base = b * stride_B + h * stride_H
    l_base = b * stride_L_B + h * stride_L_H

    offs_n = n_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n[:, None] < S_len
    offs_d = tl.arange(0, D_DIM)

    # Load invariant key-side tiles once
    kn_offsets = offs_n * stride_S + offs_d[None, :] * stride_D
    K_f = tl.load(K_ptr + kv_base + kn_offsets,
                  mask=mask_n, other=0.0, eviction_policy="evict_last").to(tl.float32)
    V_f = tl.load(V_ptr + kv_base + kn_offsets,
                  mask=mask_n, other=0.0, eviction_policy="evict_last").to(tl.float32)

    dK_acc = tl.zeros((BLOCK_N, D_DIM), dtype=tl.float32)
    dV_acc = tl.zeros((BLOCK_N, D_DIM), dtype=tl.float32)

    num_q_tiles = tl.cdiv(S_len, BLOCK_M)
    for m_idx in range(num_q_tiles):
        m_start = m_idx * BLOCK_M
        offs_m = m_start + tl.arange(0, BLOCK_M)
        mask_m = offs_m[:, None] < S_len

        qm_offsets = offs_m[:, None] * stride_S + offs_d[None, :] * stride_D
        Q_f = tl.load(Q_ptr + kv_base + qm_offsets,
                      mask=mask_m, other=0.0, eviction_policy="evict_first").to(tl.float32)
        dO_f = tl.load(dO_ptr + kv_base + qm_offsets,
                        mask=mask_m, other=0.0, eviction_policy="evict_first").to(tl.float32)
        O_f = tl.load(O_ptr + kv_base + qm_offsets,
                       mask=mask_m, other=0.0, eviction_policy="evict_first").to(tl.float32)

        L_vals = tl.load(L_ptr + l_base + offs_m * stride_L_S,
                         mask=offs_m < S_len, other=0.0)
        D_vals = tl.sum(dO_f * O_f, axis=1)

        scores = tl.dot(Q_f, K_f.T) * scale
        causal_mask = (offs_m[:, None] >= offs_n[None, :]).to(tl.float32)
        P = tl.exp(tl.minimum(scores - L_vals[:, None], 0.0)) * causal_mask

        dP = tl.dot(dO_f, V_f.T)
        dS = P * (dP - D_vals[:, None]) * scale

        dK_acc = tl.dot(dS.T, Q_f, dK_acc)
        dV_acc = tl.dot(P.T, dO_f, dV_acc)

    tl.store(dK_ptr + kv_base + kn_offsets, dK_acc.to(tl.bfloat16), mask=mask_n)
    tl.store(dV_ptr + kv_base + kn_offsets, dV_acc.to(tl.bfloat16), mask=mask_n)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
    ],
    key=["S_len", "D_DIM"],
)
@triton.heuristics(values={"BLOCK_M": lambda kwargs: kwargs.get("BLOCK_M", 64)})
@triton.jit
def _bwd_dq_autotuned(
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

    offs_m = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m[:, None] < S_len
    offs_d = tl.arange(0, D_DIM)

    qm_offsets = offs_m * stride_S + offs_d[None, :] * stride_D
    Q_f = tl.load(Q_ptr + kv_base + qm_offsets,
                  mask=mask_m, other=0.0, eviction_policy="evict_last").to(tl.float32)
    dO_f = tl.load(dO_ptr + kv_base + qm_offsets,
                   mask=mask_m, other=0.0, eviction_policy="evict_last").to(tl.float32)
    O_f = tl.load(O_ptr + kv_base + qm_offsets,
                  mask=mask_m, other=0.0, eviction_policy="evict_last").to(tl.float32)

    L_vals = tl.load(L_ptr + l_base + offs_m * stride_L_S,
                     mask=offs_m < S_len, other=0.0, eviction_policy="evict_last")
    D_vals = tl.sum(dO_f * O_f, axis=1)

    dQ_acc = tl.zeros((BLOCK_M, D_DIM), dtype=tl.float32)

    num_k_tiles = tl.cdiv(S_len, BLOCK_N)
    for k_idx in range(num_k_tiles):
        n_start = k_idx * BLOCK_N
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n[None, :] < S_len

        kn_offsets = offs_n[:, None] * stride_S + offs_d[None, :] * stride_D
        K_f = tl.load(K_ptr + kv_base + kn_offsets,
                      mask=mask_n.T, other=0.0, eviction_policy="evict_first").to(tl.float32)
        V_f = tl.load(V_ptr + kv_base + kn_offsets,
                      mask=mask_n.T, other=0.0, eviction_policy="evict_first").to(tl.float32)

        scores = tl.dot(Q_f, K_f.T) * scale
        causal_mask = (offs_m[:, None] >= offs_n[None, :]).to(tl.float32)
        P = tl.exp(tl.minimum(scores - L_vals[:, None], 0.0)) * causal_mask

        dP = tl.dot(dO_f, V_f.T)
        dS = P * (dP - D_vals[:, None]) * scale
        dQ_acc = tl.dot(dS, K_f, dQ_acc)

    out_ptrs = dQ_ptr + kv_base + qm_offsets
    tl.store(out_ptrs, dQ_acc.to(tl.bfloat16), mask=mask_m)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
    ],
    key=["S_len", "D_DIM"],
)
@triton.jit
def _bwd_dkv_autotuned(
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

    offs_n = n_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n[:, None] < S_len
    offs_d = tl.arange(0, D_DIM)

    kn_offsets = offs_n * stride_S + offs_d[None, :] * stride_D
    K_f = tl.load(K_ptr + kv_base + kn_offsets,
                  mask=mask_n, other=0.0, eviction_policy="evict_last").to(tl.float32)
    V_f = tl.load(V_ptr + kv_base + kn_offsets,
                  mask=mask_n, other=0.0, eviction_policy="evict_last").to(tl.float32)

    dK_acc = tl.zeros((BLOCK_N, D_DIM), dtype=tl.float32)
    dV_acc = tl.zeros((BLOCK_N, D_DIM), dtype=tl.float32)

    num_q_tiles = tl.cdiv(S_len, BLOCK_M)
    for m_idx in range(num_q_tiles):
        m_start = m_idx * BLOCK_M
        offs_m = m_start + tl.arange(0, BLOCK_M)
        mask_m = offs_m[:, None] < S_len

        qm_offsets = offs_m[:, None] * stride_S + offs_d[None, :] * stride_D
        Q_f = tl.load(Q_ptr + kv_base + qm_offsets,
                      mask=mask_m, other=0.0, eviction_policy="evict_first").to(tl.float32)
        dO_f = tl.load(dO_ptr + kv_base + qm_offsets,
                        mask=mask_m, other=0.0, eviction_policy="evict_first").to(tl.float32)
        O_f = tl.load(O_ptr + kv_base + qm_offsets,
                       mask=mask_m, other=0.0, eviction_policy="evict_first").to(tl.float32)

        L_vals = tl.load(L_ptr + l_base + offs_m * stride_L_S,
                         mask=offs_m < S_len, other=0.0)
        D_vals = tl.sum(dO_f * O_f, axis=1)

        scores = tl.dot(Q_f, K_f.T) * scale
        causal_mask = (offs_m[:, None] >= offs_n[None, :]).to(tl.float32)
        P = tl.exp(tl.minimum(scores - L_vals[:, None], 0.0)) * causal_mask

        dP = tl.dot(dO_f, V_f.T)
        dS = P * (dP - D_vals[:, None]) * scale

        dK_acc = tl.dot(dS.T, Q_f, dK_acc)
        dV_acc = tl.dot(P.T, dO_f, dV_acc)

    tl.store(dK_ptr + kv_base + kn_offsets, dK_acc.to(tl.bfloat16), mask=mask_n)
    tl.store(dV_ptr + kv_base + kn_offsets, dV_acc.to(tl.bfloat16), mask=mask_n)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Causal multi-head attention backward pass."""
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape

    scale = 1.0 / math.sqrt(d)

    qs = Q.stride()
    ls = L.stride()

    grid_dq = (B, H, triton.cdiv(S, 64))
    _bwd_dq_autotuned[grid_dq](
        Q, K, V, O, dO, L, dQ,
        S, scale,
        qs[0], qs[1], qs[2], qs[3],
        ls[0], ls[1], ls[2],
        D_DIM=d,
    )

    grid_dkv = (B, H, triton.cdiv(S, 64))
    _bwd_dkv_autotuned[grid_dkv](
        Q, K, V, O, dO, L, dK, dV,
        S, scale,
        qs[0], qs[1], qs[2], qs[3],
        ls[0], ls[1], ls[2],
        D_DIM=d,
    )