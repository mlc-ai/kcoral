import math
import torch
import triton
import triton.language as tl


@triton.jit
def _precompute_d_kernel(
    dO_ptr,
    O_ptr,
    D_ptr,
    BH,
    S,
    D_dim,
    stride_O_B, stride_O_H, stride_O_S, stride_O_D,
    stride_D_B, stride_D_H, stride_D_S,
    BLOCK_HYBRID: tl.constexpr,
):
    pid = tl.program_id(0)
    num_total = BH * S
    idx = pid * BLOCK_HYBRID + tl.arange(0, BLOCK_HYBRID)
    mask_base = idx < num_total

    bh_idx = idx // S
    s_idx = idx % S

    bh_b = bh_idx // 48
    bh_h = bh_idx % 48

    acc = tl.zeros((BLOCK_HYBRID,), dtype=tl.float32)
    for d_off in range(0, D_dim, 64):
        d_idx = d_off + tl.arange(0, 64)
        mask = mask_base[:, None] & (d_idx[None, :] < D_dim)

        dO_ptrs = dO_ptr + bh_b[:, None] * stride_O_B + bh_h[:, None] * stride_O_H + s_idx[:, None] * stride_O_S + d_idx[None, :] * stride_O_D
        O_ptrs = O_ptr + bh_b[:, None] * stride_O_B + bh_h[:, None] * stride_O_H + s_idx[:, None] * stride_O_S + d_idx[None, :] * stride_O_D

        dO_vals = tl.load(dO_ptrs, mask=mask, other=0.0).to(tl.float32)
        O_vals = tl.load(O_ptrs, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(dO_vals * O_vals, axis=1)

    D_ptrs = D_ptr + bh_b * stride_D_B + bh_h * stride_D_H + s_idx * stride_D_S
    tl.store(D_ptrs, acc, mask=(idx < num_total))


@triton.jit
def _dKV_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, D_ptr, L_ptr, dK_ptr, dV_ptr,
    S, D_dim, scale,
    stride_B, stride_H, stride_S, stride_D,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_q = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_kv = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    mask_q = offs_q < S
    mask_kv = offs_kv < S
    mask_qkv = (offs_q[:, None] >= offs_kv[None, :]) & mask_q[:, None] & mask_kv[None, :]

    dK_acc = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    dV_acc = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)

    for bm in range(tl.cdiv(S, BLOCK_M)):
        offs_q_blk = bm * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_q_blk = offs_q_blk < S
        mask_qkv_blk = (offs_q_blk[:, None] >= offs_kv[None, :]) & mask_q_blk[:, None] & mask_kv[None, :]

        q_ptr = Q_ptr + offs_q_blk[:, None] * stride_S + offs_d[None, :] * stride_D
        q = tl.load(q_ptr, mask=mask_q_blk[:, None] & (offs_d[None, :] < D_dim), other=0.0).to(tl.float32)

        k_ptr = K_ptr + offs_kv[:, None] * stride_S + offs_d[None, :] * stride_D
        k = tl.load(k_ptr, mask=mask_kv[:, None] & (offs_d[None, :] < D_dim), other=0.0).to(tl.float32)

        v_ptr = V_ptr + offs_kv[:, None] * stride_S + offs_d[None, :] * stride_D
        v = tl.load(v_ptr, mask=mask_kv[:, None] & (offs_d[None, :] < D_dim), other=0.0).to(tl.float32)

        do_ptr = dO_ptr + offs_q_blk[:, None] * stride_S + offs_d[None, :] * stride_D
        dO = tl.load(do_ptr, mask=mask_q_blk[:, None] & (offs_d[None, :] < D_dim), other=0.0).to(tl.float32)

        d_ptr = D_ptr + offs_q_blk * stride_S
        D_row = tl.load(d_ptr, mask=mask_q_blk, other=0.0).to(tl.float32)

        S_tile = tl.dot(q, k.T)

        S_safe = tl.where(mask_qkv_blk, S_tile, -1e5)
        m_row = tl.max(S_safe, axis=1, keep_dims=True)

        S_scale = S_safe * scale - m_row
        P_exp = tl.exp(S_scale)

        l_ptr = L_ptr + offs_q_blk * stride_S
        l_val = tl.load(l_ptr, mask=mask_q_blk, other=0.0).to(tl.float32)
        P_exp = P_exp * tl.exp(m_row - l_val[:, None])
        P = tl.where(mask_qkv_blk, P_exp, 0.0)

        dP_V = tl.dot(dO, v.T)
        dS_pre = dP_V - D_row[:, None]
        dS = tl.where(mask_qkv_blk, dS_pre * scale, 0.0)

        dK_acc += tl.dot(dS.T, q)
        dV_acc += tl.dot(P.T, dO)

    dK_out = dK_acc.to(K_ptr.dtype.element_ty)
    dK_ptr_out = dK_ptr + offs_kv[:, None] * stride_S + offs_d[None, :] * stride_D
    tl.store(dK_ptr_out, dK_out, mask=(mask_kv[:, None] & (offs_d[None, :] < D_dim)))

    dV_out = dV_acc.to(V_ptr.dtype.element_ty)
    dV_ptr_out = dV_ptr + offs_kv[:, None] * stride_S + offs_d[None, :] * stride_D
    tl.store(dV_ptr_out, dV_out, mask=(mask_kv[:, None] & (offs_d[None, :] < D_dim)))


@triton.jit
def _dQ_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, D_ptr, L_ptr, dQ_ptr,
    S, D_dim, scale,
    stride_B, stride_H, stride_S, stride_D,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_q = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    mask_q = offs_q < S
    mask_k = offs_k < S
    mask_qk = (offs_q[:, None] >= offs_k[None, :]) & mask_q[:, None] & mask_k[None, :]

    acc_dq = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    offs_d_range = offs_d < D_dim

    k_base = offs_k[:, None] * stride_S + offs_d[None, :] * stride_D
    d_base = offs_d[None, :] * stride_D

    for bn in range(tl.cdiv(S, BLOCK_N)):
        offs_k_blk = bn * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_k_blk = offs_k_blk < S
        mask_qk_blk = (offs_q[:, None] >= offs_k_blk[None, :]) & mask_q[:, None] & mask_k_blk[None, :]

        q_ptr = Q_ptr + offs_q[:, None] * stride_S + offs_d[None, :] * stride_D
        q = tl.load(q_ptr, mask=mask_q[:, None] & offs_d_range[None, :], other=0.0).to(tl.float32)

        k_ptr = K_ptr + offs_k_blk[:, None] * stride_S + offs_d[None, :] * stride_D
        k = tl.load(k_ptr, mask=mask_k_blk[:, None] & offs_d_range[None, :], other=0.0).to(tl.float32)

        v_ptr = V_ptr + offs_k_blk[:, None] * stride_S + offs_d[None, :] * stride_D
        v = tl.load(v_ptr, mask=mask_k_blk[:, None] & offs_d_range[None, :], other=0.0).to(tl.float32)

        do_ptr = dO_ptr + offs_q[:, None] * stride_S + offs_d[None, :] * stride_D
        dO = tl.load(do_ptr, mask=mask_q[:, None] & offs_d_range[None, :], other=0.0).to(tl.float32)

        d_ptr = D_ptr + offs_q * stride_S
        D_row = tl.load(d_ptr, mask=mask_q, other=0.0).to(tl.float32)

        S_tile = tl.dot(q, k.T)

        S_safe = tl.where(mask_qk_blk, S_tile, -1e5)
        m_row = tl.max(S_safe, axis=1, keep_dims=True)

        S_scal = S_safe * scale - m_row
        P_exp = tl.exp(S_scal)

        l_ptr = L_ptr + offs_q * stride_S
        l_val = tl.load(l_ptr, mask=mask_q, other=0.0).to(tl.float32)
        P_exp = P_exp * tl.exp(m_row - l_val[:, None])
        P = tl.where(mask_qk_blk, P_exp, 0.0)

        dP_V = tl.dot(dO, v.T)
        dS_pre = dP_V - D_row[:, None]
        dS = tl.where(mask_qk_blk, dS_pre * scale, 0.0)

        acc_dq += tl.dot(dS, k)

    dQ_out = acc_dq.to(Q_ptr.dtype.element_ty)
    dQ_ptr_out = dQ_ptr + offs_q[:, None] * stride_S + offs_d[None, :] * stride_D
    tl.store(dQ_ptr_out, dQ_out, mask=(mask_q[:, None] & offs_d_range[None, :]))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Causal multi-head attention backward pass.

    Inputs: Q, K, V, O, dO, L
    Outputs: dQ, dK, dV (written into preallocated tensors)

    Strategy: Two-kernel approach following FlashAttention:
    1. Precompute D = rowsum(dO * O)
    2. Kernel for dK/dV: each program owns one (batch_head, kv_block), loops over q_blocks
    3. Kernel for dQ: each program owns one (batch_head, q_block), loops over kv_blocks
    """
    torch.cuda.set_device(Q.device)

    B, H, S, D = Q.shape
    assert K.shape == Q.shape and V.shape == Q.shape
    assert O.shape == Q.shape and dO.shape == Q.shape
    assert L.shape == (B, H, S)
    assert dQ.shape == Q.shape and dK.shape == Q.shape and dV.shape == Q.shape

    BH = B * H
    scale = 1.0 / math.sqrt(D)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = 128
    NUM_WARPS = 4
    NUM_STAGES = 2

    if L.dim() == 3:
        L_work = L
    else:
        L_work = L.squeeze(-1) if L.shape[-1] == 1 else L

    stride_B = Q.stride(0)
    stride_H = Q.stride(1)
    stride_S = Q.stride(2)
    stride_D = Q.stride(3)

    # Step 1: Precompute D = rowsum(dO * O) on fp32
    D_tensor = torch.empty((B, H, S), dtype=torch.float32, device=Q.device)
    BLOCK_HYBRID = 256
    grid_D = (triton.cdiv(BH * S, BLOCK_HYBRID),)
    _precompute_d_kernel[grid_D](
        dO, O, D_tensor,
        BH, S, D,
        stride_B, stride_H, stride_S, stride_D,
        D_tensor.stride(0), D_tensor.stride(1), D_tensor.stride(2),
        BLOCK_HYBRID=BLOCK_HYBRID,
        num_warps=2, num_stages=2,
    )

    # Initialize dK and dV to zeros
    dK.zero_()
    dV.zero_()

    # Step 2: dK and dV kernel
    grid_dKV = (BH, triton.cdiv(S, BLOCK_N))
    _dKV_kernel[grid_dKV](
        Q, K, V, dO, D_tensor, L_work, dK, dV,
        S, D, scale,
        stride_B, stride_H, stride_S, stride_D,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=NUM_WARPS, num_stages=NUM_STAGES,
    )

    # Step 3: dQ kernel (no zero-init needed, each position written once)
    grid_dQ = (BH, triton.cdiv(S, BLOCK_M))
    _dQ_kernel[grid_dQ](
        Q, K, V, dO, D_tensor, L_work, dQ,
        S, D, scale,
        stride_B, stride_H, stride_S, stride_D,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=NUM_WARPS, num_stages=NUM_STAGES,
    )