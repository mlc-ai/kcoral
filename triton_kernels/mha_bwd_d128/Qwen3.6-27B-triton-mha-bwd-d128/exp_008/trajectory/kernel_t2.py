import torch
import triton
import triton.language as tl


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

    q_base = Q_ptr + pid_h * stride_qh
    k_base = K_ptr + pid_h * stride_kh
    v_base = V_ptr + pid_h * stride_vh
    o_base = O_ptr + pid_h * stride_oh
    do_base = dO_ptr + pid_h * stride_doh
    l_base = L_ptr + pid_h * stride_lh

    q_ptrs = q_base + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    q_tile = tl.load(q_ptrs, mask=mask_md, other=0.0).to(tl.float32)

    do_ptrs = do_base + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod
    do_tile = tl.load(do_ptrs, mask=mask_md, other=0.0).to(tl.float32)

    o_ptrs = o_base + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    o_tile = tl.load(o_ptrs, mask=mask_md, other=0.0).to(tl.float32)

    d_vec = tl.sum(do_tile * o_tile, axis=1)

    l_ptrs = l_base + offs_m * stride_ls
    l_vec = tl.load(l_ptrs, mask=mask_m, other=0.0)

    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    num_n_blocks = tl.cdiv(S, BLOCK_N)

    for i in range(num_n_blocks):
        offs_n = i * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S
        mask_nd = mask_n[:, None] & mask_d[None, :]

        k_ptrs = k_base + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        k_tile = tl.load(k_ptrs, mask=mask_nd, other=0.0).to(tl.float32)

        v_ptrs = v_base + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
        v_tile = tl.load(v_ptrs, mask=mask_nd, other=0.0).to(tl.float32)

        scores = tl.dot(q_tile, k_tile.T) * scale
        attn = tl.exp(scores - l_vec[:, None])
        dp_attn = tl.dot(do_tile, v_tile.T)
        dscores = attn * (dp_attn - d_vec[:, None]) * scale
        acc = tl.dot(dscores, k_tile, acc)

    tl.store(dQ_ptr + q_ptrs, acc.to(dtype=tl.bfloat16), mask=mask_md)


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

    k_base = K_ptr + pid_h * stride_kh
    v_base = V_ptr + pid_h * stride_vh
    q_base = Q_ptr + pid_h * stride_qh
    o_base = O_ptr + pid_h * stride_oh
    do_base = dO_ptr + pid_h * stride_doh
    l_base = L_ptr + pid_h * stride_lh

    k_ptrs = k_base + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    k_tile = tl.load(k_ptrs, mask=mask_nd, other=0.0).to(tl.float32)

    v_ptrs = v_base + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
    v_tile = tl.load(v_ptrs, mask=mask_nd, other=0.0).to(tl.float32)

    dk_acc = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    dv_acc = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)

    num_m_blocks = tl.cdiv(S, BLOCK_M)

    for i in range(num_m_blocks):
        offs_m = i * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S
        mask_md = mask_m[:, None] & mask_d[None, :]

        q_ptrs_i = q_base + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
        q_tile = tl.load(q_ptrs_i, mask=mask_md, other=0.0).to(tl.float32)

        do_ptrs_i = do_base + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod
        do_tile = tl.load(do_ptrs_i, mask=mask_md, other=0.0).to(tl.float32)

        o_ptrs_i = o_base + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
        o_tile = tl.load(o_ptrs_i, mask=mask_md, other=0.0).to(tl.float32)

        d_vec = tl.sum(do_tile * o_tile, axis=1)

        l_ptrs_i = l_base + offs_m * stride_ls
        l_vec = tl.load(l_ptrs_i, mask=mask_m, other=0.0)

        scores = tl.dot(q_tile, k_tile.T) * scale
        attn = tl.exp(scores - l_vec[:, None])
        dp_attn = tl.dot(do_tile, v_tile.T)
        dscores = attn * (dp_attn - d_vec[:, None]) * scale

        dk_acc = tl.dot(dscores.T, q_tile, dk_acc)
        dv_acc = tl.dot(attn.T, do_tile, dv_acc)

    tl.store(dK_ptr + k_ptrs, dk_acc.to(dtype=tl.bfloat16), mask=mask_nd)
    tl.store(dV_ptr + v_ptrs, dv_acc.to(dtype=tl.bfloat16), mask=mask_nd)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Multi-head attention backward pass."""
    torch.cuda.set_device(Q.device)

    B, H, S, D = Q.shape
    LOG_B = B * H

    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_D = D

    num_m = triton.cdiv(S, BLOCK_M)
    num_n = triton.cdiv(S, BLOCK_N)

    scale = 1.0 / (D ** 0.5)

    # Handle L shape (may be [B,H,S] or [B,H,S,1])
    if L.dim() == 4:
        L = L.squeeze(-1)

    stride_qh, stride_qs, stride_qd = Q.stride(1), Q.stride(2), Q.stride(3)
    stride_kh, stride_ks, stride_kd = K.stride(1), K.stride(2), K.stride(3)
    stride_vh, stride_vs, stride_vd = V.stride(1), V.stride(2), V.stride(3)
    stride_oh, stride_os, stride_od = O.stride(1), O.stride(2), O.stride(3)
    stride_doh, stride_dos, stride_dod = dO.stride(1), dO.stride(2), dO.stride(3)
    stride_lh, stride_ls = L.stride(1), L.stride(2)

    # Launch dQ kernel
    grid_dq = (LOG_B, num_m)
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
        num_warps=8, num_stages=3,
    )

    # Launch dKV kernel
    grid_dkv = (LOG_B, num_n)
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
        num_warps=8, num_stages=3,
    )