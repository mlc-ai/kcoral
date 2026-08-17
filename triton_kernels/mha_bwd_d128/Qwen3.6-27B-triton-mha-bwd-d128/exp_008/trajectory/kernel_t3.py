import torch
import triton
import triton.language as tl


@triton.jit
def _cast_to_fp32(x):
    return x.to(tl.float32)


@triton.jit
def _dq_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dQ_ptr,
    stride_qh, stride_qs, stride_qd,
    stride_kh, stride_ks, stride_kd,
    stride_vh, stride_vs, stride_vd,
    stride_oh, stride_os, stride_od,
    stride_doh, stride_dos, stride_dod,
    stride_lh, stride_ls,
    S, D, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """Compute dQ for one logical head and one query sequence block."""
    pid_h = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < S
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    mask_md = mask_m[:, None] & mask_d[None, :]

    # Build base offsets for this head
    h_off_q = pid_h * stride_qh
    h_off_k = pid_h * stride_kh
    h_off_v = pid_h * stride_vh
    h_off_o = pid_h * stride_oh
    h_off_do = pid_h * stride_doh
    h_off_l = pid_h * stride_lh

    # Load Q
    q_ptrs = Q_ptr + h_off_q + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    q_raw = tl.load(q_ptrs, mask=mask_md, other=0.0)
    q_tile = _cast_to_fp32(q_raw)

    # Load dO
    do_ptrs = dO_ptr + h_off_do + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod
    do_raw = tl.load(do_ptrs, mask=mask_md, other=0.0)
    do_tile = _cast_to_fp32(do_raw)

    # Load O
    o_ptrs = O_ptr + h_off_o + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    o_raw = tl.load(o_ptrs, mask=mask_md, other=0.0)
    o_tile = _cast_to_fp32(o_raw)

    # D = row_sum(dO * O), shape [BLOCK_M]
    d_vec = tl.sum(do_tile * o_tile, axis=1)

    # Load L (logsumexp) vector, shape [BLOCK_M]
    l_ptrs = L_ptr + h_off_l + offs_m * stride_ls
    l_raw = tl.load(l_ptrs, mask=mask_m, other=0.0)
    l_vec = _cast_to_fp32(l_raw)

    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    num_n_blocks = tl.cdiv(S, BLOCK_N)

    for i in range(num_n_blocks):
        offs_n = i * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S
        mask_nd = mask_n[:, None] & mask_d[None, :]

        # Load K tile [BLOCK_N, BLOCK_D]
        k_ptrs = K_ptr + h_off_k + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        k_raw = tl.load(k_ptrs, mask=mask_nd, other=0.0)
        k_tile = _cast_to_fp32(k_raw)

        # Load V tile [BLOCK_N, BLOCK_D]
        v_ptrs = V_ptr + h_off_v + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
        v_raw = tl.load(v_ptrs, mask=mask_nd, other=0.0)
        v_tile = _cast_to_fp32(v_raw)

        # Scores = Q @ K^T * scale, shape [BLOCK_M, BLOCK_N]
        scores = tl.dot(q_tile, k_tile.T) * scale

        # Attention weights P = exp(scores - L)
        attn = tl.exp(scores - l_vec[:, None])

        # dP = dO @ V^T, shape [BLOCK_M, BLOCK_N]
        dp_attn = tl.dot(do_tile, v_tile.T)

        # dS = P * (dP - D) * scale
        dscores = attn * (dp_attn - d_vec[:, None]) * scale

        # Accumulate dQ += dS @ K
        acc = tl.dot(dscores, k_tile, acc)

    # Store dQ result
    dq_ptrs = dQ_ptr + h_off_q + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    tl.store(dq_ptrs, acc.to(dtype=tl.bfloat16), mask=mask_md)


@triton.jit
def _dkv_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dK_ptr, dV_ptr,
    stride_qh, stride_qs, stride_qd,
    stride_kh, stride_ks, stride_kd,
    stride_vh, stride_vs, stride_vd,
    stride_oh, stride_os, stride_od,
    stride_doh, stride_dos, stride_dod,
    stride_lh, stride_ls,
    S, D, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """Compute dK and dV for one logical head and one KV sequence block."""
    pid_h = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < S
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    mask_nd = mask_n[:, None] & mask_d[None, :]

    # Head offsets
    h_off_q = pid_h * stride_qh
    h_off_k = pid_h * stride_kh
    h_off_v = pid_h * stride_vh
    h_off_o = pid_h * stride_oh
    h_off_do = pid_h * stride_doh
    h_off_l = pid_h * stride_lh

    # Load constant K tile [BLOCK_N, BLOCK_D] for this kernel instance
    k_ptrs = K_ptr + h_off_k + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    k_raw = tl.load(k_ptrs, mask=mask_nd, other=0.0)
    k_tile = _cast_to_fp32(k_raw)

    # Load constant V tile [BLOCK_N, BLOCK_D] for this kernel instance
    v_ptrs = V_ptr + h_off_v + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
    v_raw = tl.load(v_ptrs, mask=mask_nd, other=0.0)
    v_tile = _cast_to_fp32(v_raw)

    dk_acc = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    dv_acc = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)

    num_m_blocks = tl.cdiv(S, BLOCK_M)

    for i in range(num_m_blocks):
        offs_m = i * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S
        mask_md = mask_m[:, None] & mask_d[None, :]

        # Load Q tile [BLOCK_M, BLOCK_D]
        q_ptrs_i = Q_ptr + h_off_q + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
        q_raw = tl.load(q_ptrs_i, mask=mask_md, other=0.0)
        q_tile = _cast_to_fp32(q_raw)

        # Load dO tile [BLOCK_M, BLOCK_D]
        do_ptrs_i = dO_ptr + h_off_do + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod
        do_raw = tl.load(do_ptrs_i, mask=mask_md, other=0.0)
        do_tile = _cast_to_fp32(do_raw)

        # Load O tile [BLOCK_M, BLOCK_D]
        o_ptrs_i = O_ptr + h_off_o + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
        o_raw = tl.load(o_ptrs_i, mask=mask_md, other=0.0)
        o_tile = _cast_to_fp32(o_raw)

        # D = row_sum(dO * O), shape [BLOCK_M]
        d_vec = tl.sum(do_tile * o_tile, axis=1)

        # Load L vector
        l_ptrs_i = L_ptr + h_off_l + offs_m * stride_ls
        l_raw = tl.load(l_ptrs_i, mask=mask_m, other=0.0)
        l_vec = _cast_to_fp32(l_raw)

        # Scores = Q @ K^T * scale, shape [BLOCK_M, BLOCK_N]
        scores = tl.dot(q_tile, k_tile.T) * scale

        # Attention weights P = exp(scores - L)
        attn = tl.exp(scores - l_vec[:, None])

        # dP = dO @ V^T, shape [BLOCK_M, BLOCK_N]
        dp_attn = tl.dot(do_tile, v_tile.T)

        # dS = P * (dP - D) * scale
        dscores = attn * (dp_attn - d_vec[:, None]) * scale

        # dK += dS^T @ Q
        dk_acc = tl.dot(dscores.T, q_tile, dk_acc)

        # dV += P^T @ dO
        dv_acc = tl.dot(attn.T, do_tile, dv_acc)

    # Write results
    tl.store(dK_ptr + k_ptrs, dk_acc.to(dtype=tl.bfloat16), mask=mask_nd)
    tl.store(dV_ptr + v_ptrs, dv_acc.to(dtype=tl.bfloat16), mask=mask_nd)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Multi-head attention backward pass."""
    torch.cuda.set_device(Q.device)

    B, H, S, D = Q.shape
    LOG_B = B * H

    # Tile sizes tuned for BF16 on Hopper with d=128
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = D

    num_m_tiles = triton.cdiv(S, BLOCK_M)
    num_n_tiles = triton.cdiv(S, BLOCK_N)

    scale = 1.0 / (D ** 0.5)

    # Get strides
    sqh, sqs, sqd = Q.stride(1), Q.stride(2), Q.stride(3)
    skh, sks, skd = K.stride(1), K.stride(2), K.stride(3)
    svh, svs, svd = V.stride(1), V.stride(2), V.stride(3)
    soh, sos, sod = O.stride(1), O.stride(2), O.stride(3)
    sdoh, sdos, sdod = dO.stride(1), dO.stride(2), dO.stride(3)
    slh, sls = L.stride(1), L.stride(2)

    # Launch dQ kernel: grid over (logical_heads, query_sequence_blocks)
    grid_dq = (LOG_B, num_m_tiles)
    _dq_kernel[grid_dq](
        Q, K, V, O, dO, L, dQ,
        sqh, sqs, sqd,
        skh, sks, skd,
        svh, svs, svd,
        soh, sos, sod,
        sdoh, sdos, sdod,
        slh, sls,
        S, D, scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=8, num_stages=3,
    )

    # Launch dKV kernel: grid over (logical_heads, kv_sequence_blocks)
    grid_dkv = (LOG_B, num_n_tiles)
    _dkv_kernel[grid_dkv](
        Q, K, V, O, dO, L, dK, dV,
        sqh, sqs, sqd,
        skh, sks, skd,
        svh, svs, svd,
        soh, sos, sod,
        sdoh, sdos, sdod,
        slh, sls,
        S, D, scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=8, num_stages=3,
    )