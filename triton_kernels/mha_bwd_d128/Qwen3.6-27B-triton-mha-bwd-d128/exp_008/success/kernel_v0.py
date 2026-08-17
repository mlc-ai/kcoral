import torch
import triton
import triton.language as tl


@triton.jit
def _dq_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    dO_ptr,
    L_ptr,
    dQ_ptr,
    stride_qh,
    stride_qs,
    stride_qd,
    stride_kh,
    stride_ks,
    stride_kd,
    stride_vh,
    stride_vs,
    stride_vd,
    stride_oh,
    stride_os,
    stride_od,
    stride_doh,
    stride_dos,
    stride_dod,
    stride_lh,
    stride_ls,
    S,
    D,
    scale,
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

    # Offsets within this Q block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < S
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    mask_md = mask_m[:, None] & mask_d[None, :]

    # Base pointers for this logical head
    q_base = Q_ptr + pid_h * stride_qh
    k_base = K_ptr + pid_h * stride_kh
    v_base = V_ptr + pid_h * stride_vh
    o_base = O_ptr + pid_h * stride_oh
    do_base = dO_ptr + pid_h * stride_doh
    l_base = L_ptr + pid_h * stride_lh

    # Load Q tile -> FP32
    q_ptrs = q_base + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    q_tile = tl.load(q_ptrs, mask=mask_md, other=0.0).to(tl.float32)

    # Load dO tile -> FP32
    do_ptrs = do_base + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod
    do_tile = tl.load(do_ptrs, mask=mask_md, other=0.0).to(tl.float32)

    # Load O tile -> FP32
    o_ptrs = o_base + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    o_tile = tl.load(o_ptrs, mask=mask_md, other=0.0).to(tl.float32)

    # D = row_sum(dO * O)  -> shape [BLOCK_M]
    d_vec = tl.sum(do_tile * o_tile, axis=1)

    # Load logsumexp L vector
    l_ptrs = l_base + offs_m * stride_ls
    l_vec = tl.load(l_ptrs, mask=mask_m, other=float('-inf'))

    # Accumulator for dQ [BLOCK_M, BLOCK_D]
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    num_n_tiles = tl.cdiv(S, BLOCK_N)

    for off_n in range(num_n_tiles):
        offs_n = off_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S

        # Load K tile [BLOCK_N, BLOCK_D] -> FP32
        k_ptrs = k_base + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        mask_nd = mask_n[:, None] & mask_d[None, :]
        k_tile = tl.load(k_ptrs, mask=mask_nd, other=0.0).to(tl.float32)

        # Load V tile [BLOCK_N, BLOCK_D] -> FP32
        v_ptrs = v_base + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
        v_tile = tl.load(v_ptrs, mask=mask_nd, other=0.0).to(tl.float32)

        # Q @ K^T * scale -> [BLOCK_M, BLOCK_N]
        scores = tl.dot(q_tile, k_tile.T) * scale

        # P = exp(scores - L) -> [BLOCK_M, BLOCK_N]
        attn = tl.exp(scores - l_vec[:, None])

        # dO @ V^T -> [BLOCK_M, BLOCK_N]
        dp_attn = tl.dot(do_tile, v_tile.T)

        # dS = P * (dP - D) * scale
        dscores = attn * (dp_attn - d_vec[:, None]) * scale

        # dQ += dS @ K
        acc = tl.dot(dscores, k_tile, acc)

    # Write result to dQ
    dq_ptrs = dQ_ptr + pid_h * stride_qh + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    tl.store(dq_ptrs, acc.to(dtype=tl.bfloat16), mask=mask_md)


@triton.jit
def _dkv_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    dO_ptr,
    L_ptr,
    dK_ptr,
    dV_ptr,
    stride_qh,
    stride_qs,
    stride_qd,
    stride_kh,
    stride_ks,
    stride_kd,
    stride_vh,
    stride_vs,
    stride_vd,
    stride_oh,
    stride_os,
    stride_od,
    stride_doh,
    stride_dos,
    stride_dod,
    stride_lh,
    stride_ls,
    S,
    D,
    scale,
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

    # Offsets within this KV block
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < S
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    mask_nd = mask_n[:, None] & mask_d[None, :]

    # Base pointers for this logical head
    q_base = Q_ptr + pid_h * stride_qh
    k_base = K_ptr + pid_h * stride_kh
    v_base = V_ptr + pid_h * stride_vh
    o_base = O_ptr + pid_h * stride_oh
    do_base = dO_ptr + pid_h * stride_doh
    l_base = L_ptr + pid_h * stride_lh

    # Load constant K and V tiles for this kernel instance -> FP32
    k_ptrs = k_base + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    k_tile = tl.load(k_ptrs, mask=mask_nd, other=0.0).to(tl.float32)
    v_ptrs = v_base + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
    v_tile = tl.load(v_ptrs, mask=mask_nd, other=0.0).to(tl.float32)

    # Accumulators [BLOCK_N, BLOCK_D]
    dk_acc = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    dv_acc = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)

    num_m_tiles = tl.cdiv(S, BLOCK_M)

    for off_m in range(num_m_tiles):
        offs_m = off_m * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S
        mask_md = mask_m[:, None] & mask_d[None, :]

        # Load Q tile -> FP32
        q_ptrs = q_base + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
        q_tile = tl.load(q_ptrs, mask=mask_md, other=0.0).to(tl.float32)

        # Load dO tile -> FP32
        do_ptrs = do_base + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod
        do_tile = tl.load(do_ptrs, mask=mask_md, other=0.0).to(tl.float32)

        # Load O tile -> FP32
        o_ptrs = o_base + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
        o_tile = tl.load(o_ptrs, mask=mask_md, other=0.0).to(tl.float32)

        # D = row_sum(dO * O) -> [BLOCK_M]
        d_vec = tl.sum(do_tile * o_tile, axis=1)

        # Load L vector
        l_ptrs = l_base + offs_m * stride_ls
        l_vec = tl.load(l_ptrs, mask=mask_m, other=float('-inf'))

        # Q @ K^T * scale -> [BLOCK_M, BLOCK_N]
        scores = tl.dot(q_tile, k_tile.T) * scale

        # P = exp(scores - L) -> [BLOCK_M, BLOCK_N]
        attn = tl.exp(scores - l_vec[:, None])

        # dO @ V^T -> [BLOCK_M, BLOCK_N]
        dp_attn = tl.dot(do_tile, v_tile.T)

        # dS = P * (dP - D) * scale
        dscores = attn * (dp_attn - d_vec[:, None]) * scale

        # dK += dS^T @ Q
        dk_acc = tl.dot(dscores.T, q_tile, dk_acc)

        # dV += P^T @ dO
        dv_acc = tl.dot(attn.T, do_tile, dv_acc)

    # Write dK result
    dk_ptrs = dK_ptr + pid_h * stride_kh + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    tl.store(dk_ptrs, dk_acc.to(dtype=tl.bfloat16), mask=mask_nd)

    # Write dV result
    dv_ptrs = dV_ptr + pid_h * stride_vh + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
    tl.store(dv_ptrs, dv_acc.to(dtype=tl.bfloat16), mask=mask_nd)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Multi-head attention backward pass.
    
    Computes dQ, dK, dV given Q, K, V, forward output O,
    upstream gradient dO, and log-sum-exp statistics L.
    """
    torch.cuda.set_device(Q.device)

    B, H, S, D = Q.shape
    LOG_B = B * H  # Total logical heads

    # Tile sizes
    BLOCK_M = 64   # Query sequence block size
    BLOCK_N = 64   # KV sequence block size
    BLOCK_D = D    # Head dimension (128, power of 2)

    num_m_tiles = triton.cdiv(S, BLOCK_M)
    num_n_tiles = triton.cdiv(S, BLOCK_N)

    scale = 1.0 / (D ** 0.5)

    # Extract individual strides for clarity
    stride_qh = Q.stride(1)
    stride_qs = Q.stride(2)
    stride_qd = Q.stride(3)
    stride_kh = K.stride(1)
    stride_ks = K.stride(2)
    stride_kd = K.stride(3)
    stride_vh = V.stride(1)
    stride_vs = V.stride(2)
    stride_vd = V.stride(3)
    stride_oh = O.stride(1)
    stride_os = O.stride(2)
    stride_od = O.stride(3)
    stride_doh = dO.stride(1)
    stride_dos = dO.stride(2)
    stride_dod = dO.stride(3)
    stride_lh = L.stride(1)
    stride_ls = L.stride(2)

    # Launch dQ kernel
    grid_dq = (LOG_B, num_m_tiles)
    _dq_kernel[grid_dq](
        Q, K, V, O, dO, L, dQ,
        stride_qh, stride_qs, stride_qd,
        stride_kh, stride_ks, stride_kd,
        stride_vh, stride_vs, stride_vd,
        stride_oh, stride_os, stride_od,
        stride_doh, stride_dos, stride_dod,
        stride_lh, stride_ls,
        S, D, scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=4, num_stages=2,
    )

    # Launch dKV kernel
    grid_dkv = (LOG_B, num_n_tiles)
    _dkv_kernel[grid_dkv](
        Q, K, V, O, dO, L, dK, dV,
        stride_qh, stride_qs, stride_qd,
        stride_kh, stride_ks, stride_kd,
        stride_vh, stride_vs, stride_vd,
        stride_oh, stride_os, stride_od,
        stride_doh, stride_dos, stride_dod,
        stride_lh, stride_ls,
        S, D, scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=4, num_stages=2,
    )