import torch
import triton
import triton.language as tl


@triton.jit
def _dq_kernel(
    Q, K, V, O, dO, L, dQ_out,
    stride_h, stride_s, stride_lh, stride_ls,
    S, D, scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Compute dQ for one logical head and one query sequence block.
    Iterates over all KV sequence blocks.
    """
    pid_h = tl.program_id(0)
    pid_m = tl.program_id(1)

    base = pid_h * stride_h
    base_l = pid_h * stride_lh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < S
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    # Build 2D address tile [BLOCK_M, BLOCK_D] for this Q block
    ptr_md = base + offs_m[:, None] * stride_s + offs_d[None, :]

    q_tile = tl.load(Q + ptr_md, mask=mask_m[:, None] & mask_d[None, :], other=0.0)
    do_tile = tl.load(dO + ptr_md, mask=mask_m[:, None] & mask_d[None, :], other=0.0)
    o_tile = tl.load(O + ptr_md, mask=mask_m[:, None] & mask_d[None, :], other=0.0)

    # D = row-sum(dO * O) -> shape [BLOCK_M]
    d_vec = tl.sum(do_tile * o_tile, axis=1)

    # Load logsumexp
    l_vec = tl.load(L + base_l + offs_m * stride_ls, mask=mask_m, other=0.0)

    # Accumulator for dQ
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    # Iterate over all KV blocks
    for off_n in range(tl.cdiv(S, BLOCK_N)):
        offs_n = off_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S

        ptr_nd = base + offs_n[:, None] * stride_s + offs_d[None, :]
        k_tile = tl.load(K + ptr_nd, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        v_tile = tl.load(V + ptr_nd, mask=mask_n[:, None] & mask_d[None, :], other=0.0)

        # Scaled attention scores: Q @ K^T * scale
        s = tl.dot(q_tile, k_tile.T) * scale

        # Attention probabilities: exp(S - L)
        p = tl.exp(s - l_vec[:, None])

        # Gradient of attention scores w.r.t. O: dO @ V^T
        dp = tl.dot(do_tile, v_tile.T)

        # dS = P * (dP - D) * scale
        ds = p * (dp - d_vec[:, None]) * scale

        # Accumulate dQ += dS @ K
        acc = tl.dot(ds, k_tile, acc)

    # Write result
    tl.store(
        dQ_out + ptr_md,
        acc.to(dtype=tl.bfloat16),
        mask=mask_m[:, None] & mask_d[None, :],
    )


@triton.jit
def _dkv_kernel(
    Q, K, V, O, dO, L, dK_out, dV_out,
    stride_h, stride_s, stride_lh, stride_ls,
    S, D, scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Compute dK and dV for one logical head and one KV sequence block.
    Iterates over all query sequence blocks.
    """
    pid_h = tl.program_id(0)
    pid_n = tl.program_id(1)

    base = pid_h * stride_h
    base_l = pid_h * stride_lh

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < S
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    # Load K and V tiles (constant throughout the loop)
    ptr_nd = base + offs_n[:, None] * stride_s + offs_d[None, :]
    k_tile = tl.load(K + ptr_nd, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
    v_tile = tl.load(V + ptr_nd, mask=mask_n[:, None] & mask_d[None, :], other=0.0)

    # Accumulators for dK and dV
    dk_acc = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    dv_acc = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)

    # Iterate over all Q blocks
    for off_m in range(tl.cdiv(S, BLOCK_M)):
        offs_m = off_m * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S

        ptr_md = base + offs_m[:, None] * stride_s + offs_d[None, :]
        q_tile = tl.load(Q + ptr_md, mask=mask_m[:, None] & mask_d[None, :], other=0.0)
        do_tile = tl.load(dO + ptr_md, mask=mask_m[:, None] & mask_d[None, :], other=0.0)
        o_tile = tl.load(O + ptr_md, mask=mask_m[:, None] & mask_d[None, :], other=0.0)

        # D = row-sum(dO * O) -> shape [BLOCK_M]
        d_vec = tl.sum(do_tile * o_tile, axis=1)

        # Load logsumexp for this Q block
        l_vec = tl.load(L + base_l + offs_m * stride_ls, mask=mask_m, other=0.0)

        # Scaled attention scores
        s = tl.dot(q_tile, k_tile.T) * scale

        # Attention probabilities
        p = tl.exp(s - l_vec[:, None])

        # Gradient of attention scores w.r.t. O: dO @ V^T
        dp = tl.dot(do_tile, v_tile.T)

        # dS = P * (dP - D) * scale
        ds = p * (dp - d_vec[:, None]) * scale

        # dK += dS^T @ Q
        dk_acc = tl.dot(ds.T, q_tile, dk_acc)

        # dV += P^T @ dO
        dv_acc = tl.dot(p.T, do_tile, dv_acc)

    # Write results
    tl.store(
        dK_out + ptr_nd,
        dk_acc.to(dtype=tl.bfloat16),
        mask=mask_n[:, None] & mask_d[None, :],
    )
    tl.store(
        dV_out + ptr_nd,
        dv_acc.to(dtype=tl.bfloat16),
        mask=mask_n[:, None] & mask_d[None, :],
    )


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Multi-head attention backward pass.
    
    Computes dQ, dK, dV given Q, K, V, forward output O,
    upstream gradient dO, and log-sum-exp statistics L.
    """
    torch.cuda.set_device(Q.device)

    B, H, S, D = Q.shape
    LOG_B = B * H  # Total logical heads

    # Extract strides from the 4D tensors (assumes consistent layout)
    stride_h = Q.stride(1)      # Elements between consecutive logical heads
    stride_s = Q.stride(2)      # Elements between consecutive sequence positions
    stride_lh = L.stride(1)     # Stride between logical heads in L
    stride_ls = L.stride(2)     # Stride between sequence positions in L

    scale = 1.0 / (D ** 0.5)

    # Tile sizes
    BLOCK_M = 64   # Query sequence block size
    BLOCK_N = 64   # KV sequence block size
    BLOCK_D = D    # Head dimension (must be power of 2; D=128 here)

    num_m_tiles = triton.cdiv(S, BLOCK_M)
    num_n_tiles = triton.cdiv(S, BLOCK_N)

    # Kernel 1: compute dQ
    # Each program handles one (logical_head, query_block) -> writes one dQ tile
    grid_dq = (LOG_B, num_m_tiles)
    _dq_kernel[grid_dq](
        Q, K, V, O, dO, L, dQ,
        stride_h, stride_s, stride_lh, stride_ls,
        S, D, scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=4, num_stages=2,
    )

    # Kernel 2: compute dK and dV
    # Each program handles one (logical_head, kv_block) -> writes one dK and dV tile
    grid_dkv = (LOG_B, num_n_tiles)
    _dkv_kernel[grid_dkv](
        Q, K, V, O, dO, L, dK, dV,
        stride_h, stride_s, stride_lh, stride_ls,
        S, D, scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=4, num_stages=2,
    )