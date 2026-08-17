import torch
import triton
import triton.language as tl


@triton.jit
def _precompute_d_kernel(
    do_ptr,
    o_ptr,
    d_ptr,
    S,
    D,
    stride_bs,
    stride_bd,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """For each (b,h) pair, compute D[s] = sum_d(dO[s,d] * O[s,d])."""
    pid = tl.program_id(0)
    s_offs = pid * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < S
    acc = tl.zeros((BLOCK_S,), dtype=tl.float32)
    for db in range(0, D, BLOCK_D):
        d_offs = db + tl.arange(0, BLOCK_D)
        d_mask = d_offs < D
        do_ptrs = do_ptr + s_offs[:, None] * stride_bs + d_offs[None, :] * stride_bd
        o_ptrs = o_ptr + s_offs[:, None] * stride_bs + d_offs[None, :] * stride_bd
        do_tile = tl.load(do_ptrs, mask=s_mask[:, None] & d_mask[None, :], other=0.0)
        o_tile = tl.load(o_ptrs, mask=s_mask[:, None] & d_mask[None, :], other=0.0)
        acc += tl.sum(do_tile * o_tile, axis=1)
    tl.store(d_ptr + s_offs, acc, mask=s_mask)


@triton.jit
def _dq_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, D_ptr, dQ_ptr,
    S, D_MODEL,
    stride_bhs_q, stride_s_q, stride_d_q,
    stride_bhs_k, stride_s_k, stride_d_k,
    stride_bhs_v, stride_s_v, stride_d_v,
    stride_bhs_do, stride_s_do, stride_d_do,
    stride_bhs_l, stride_s_l,
    stride_bhs_d, stride_s_d,
    stride_bhs_dq, stride_s_dq, stride_d_dq,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute dQ: each (b,h,q_block) accumulates over all KV blocks."""
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    q_base = Q_ptr + pid_bh * stride_bhs_q
    k_base = K_ptr + pid_bh * stride_bhs_k
    v_base = V_ptr + pid_bh * stride_bhs_v
    do_base = dO_ptr + pid_bh * stride_bhs_do
    l_base = L_ptr + pid_bh * stride_bhs_l
    d_base = D_ptr + pid_bh * stride_bhs_d
    dq_base = dQ_ptr + pid_bh * stride_bhs_dq

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    d_offs = tl.arange(0, BLOCK_D)

    q_ptrs = q_base + m_offs[:, None] * stride_s_q + d_offs[None, :] * stride_d_q
    q = tl.load(q_ptrs, mask=(m_offs[:, None] < S) & (d_offs[None, :] < D_MODEL), other=0.0)

    l_ptrs = l_base + m_offs * stride_s_l
    l_expanded = tl.load(l_ptrs, mask=m_offs < S, other=0.0).to(tl.float32)[:, None]

    d_ptrs = d_base + m_offs * stride_s_d
    d_expanded = tl.load(d_ptrs, mask=m_offs < S, other=0.0).to(tl.float32)[:, None]

    do_ptrs = do_base + m_offs[:, None] * stride_s_do + d_offs[None, :] * stride_d_do
    do = tl.load(do_ptrs, mask=(m_offs[:, None] < S) & (d_offs[None, :] < D_MODEL), other=0.0)

    scale = 1.0 / tl.sqrt(float(D_MODEL))
    num_n_blocks = tl.cdiv(S, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for nb in range(num_n_blocks):
        n_offs = nb * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = n_offs < S

        k_ptrs = k_base + n_offs[:, None] * stride_s_k + d_offs[None, :] * stride_d_k
        k = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)

        v_ptrs = v_base + n_offs[:, None] * stride_s_v + d_offs[None, :] * stride_d_v
        v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)

        qk = tl.dot(q, k.T) * scale
        probs = tl.exp(qk - l_expanded)
        dp = tl.dot(do, v.T)
        di = dp - d_expanded
        da = probs * di * scale
        acc = tl.dot(da, k, acc)

    dq_val = acc.to(Q_ptr.dtype.element_ty)
    dq_ptrs = dq_base + m_offs[:, None] * stride_s_dq + d_offs[None, :] * stride_d_dq
    tl.store(dq_ptrs, dq_val, mask=(m_offs[:, None] < S) & (d_offs[None, :] < D_MODEL))


@triton.jit
def _dkv_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, D_ptr, dK_ptr, dV_ptr,
    S, D_MODEL,
    stride_bhs_q, stride_s_q, stride_d_q,
    stride_bhs_k, stride_s_k, stride_d_k,
    stride_bhs_v, stride_s_v, stride_d_v,
    stride_bhs_do, stride_s_do, stride_d_do,
    stride_bhs_l, stride_s_l,
    stride_bhs_d, stride_s_d,
    stride_bhs_dk, stride_s_dk, stride_d_dk,
    stride_bhs_dv, stride_s_dv, stride_d_dv,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute dK, dV: each (b,h,kv_block) accumulates over all Q blocks."""
    pid_bh = tl.program_id(0)
    pid_n = tl.program_id(1)

    q_base = Q_ptr + pid_bh * stride_bhs_q
    k_base = K_ptr + pid_bh * stride_bhs_k
    v_base = V_ptr + pid_bh * stride_bhs_v
    do_base = dO_ptr + pid_bh * stride_bhs_do
    l_base = L_ptr + pid_bh * stride_bhs_l
    d_base = D_ptr + pid_bh * stride_bhs_d
    dk_base = dK_ptr + pid_bh * stride_bhs_dk
    dv_base = dV_ptr + pid_bh * stride_bhs_dv

    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    d_offs = tl.arange(0, BLOCK_D)

    k_ptrs = k_base + n_offs[:, None] * stride_s_k + d_offs[None, :] * stride_d_k
    k = tl.load(k_ptrs, mask=(n_offs[:, None] < S) & (d_offs[None, :] < D_MODEL), other=0.0)

    v_ptrs = v_base + n_offs[:, None] * stride_s_v + d_offs[None, :] * stride_d_v
    v = tl.load(v_ptrs, mask=(n_offs[:, None] < S) & (d_offs[None, :] < D_MODEL), other=0.0)

    scale = 1.0 / tl.sqrt(float(D_MODEL))
    num_m_blocks = tl.cdiv(S, BLOCK_M)
    acc_dk = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    acc_dv = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)

    for mb in range(num_m_blocks):
        m_offs = mb * BLOCK_M + tl.arange(0, BLOCK_M)
        m_mask = m_offs < S

        q_ptrs = q_base + m_offs[:, None] * stride_s_q + d_offs[None, :] * stride_d_q
        q = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0)

        l_ptrs = l_base + m_offs * stride_s_l
        l_expanded = tl.load(l_ptrs, mask=m_mask, other=0.0).to(tl.float32)[:, None]

        d_ptrs = d_base + m_offs * stride_s_d
        d_expanded = tl.load(d_ptrs, mask=m_mask, other=0.0).to(tl.float32)[:, None]

        do_ptrs = do_base + m_offs[:, None] * stride_s_do + d_offs[None, :] * stride_d_do
        do = tl.load(do_ptrs, mask=m_mask[:, None], other=0.0)

        kq = tl.dot(q, k.T) * scale
        probs = tl.exp(kq - l_expanded)
        dp = tl.dot(do, v.T)
        di = dp - d_expanded
        da = probs * di * scale
        acc_dk = tl.dot(da.T, q, acc_dk)
        acc_dv = tl.dot(probs.T, do, acc_dv)

    dk_val = acc_dk.to(K_ptr.dtype.element_ty)
    dk_ptrs = dk_base + n_offs[:, None] * stride_s_dk + d_offs[None, :] * stride_d_dk
    tl.store(dk_ptrs, dk_val, mask=(n_offs[:, None] < S) & (d_offs[None, :] < D_MODEL))

    dv_val = acc_dv.to(V_ptr.dtype.element_ty)
    dv_ptrs = dv_base + n_offs[:, None] * stride_s_dv + d_offs[None, :] * stride_d_dv
    tl.store(dv_ptrs, dv_val, mask=(n_offs[:, None] < S) & (d_offs[None, :] < D_MODEL))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Multi-head attention backward: compute dQ, dK, dV from Q,K,V,O,dO,L."""
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    assert K.shape == (B, H, S, D)
    assert V.shape == (B, H, S, D)
    assert O.shape == (B, H, S, D)
    assert dO.shape == (B, H, S, D)
    assert dQ.shape == (B, H, S, D)
    assert dK.shape == (B, H, S, D)
    assert dV.shape == (B, H, S, D)

    BH = B * H

    # Precompute D = rowsum(dO * O) with shape [B*H, S]
    D_tensor = torch.empty((BH, S), device=Q.device, dtype=torch.float32)
    BLOCK_PRE_S = 256
    BLOCK_PRE_D = 64
    grid_pre = (triton.cdiv(BH * S, BLOCK_PRE_S),)
    _precompute_d_kernel[grid_pre](
        dO, O, D_tensor,
        S, D,
        dO.stride(-2), dO.stride(-1),
        BLOCK_S=BLOCK_PRE_S,
        BLOCK_D=BLOCK_PRE_D,
        num_warps=4,
        num_stages=2,
    )

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = D  # 128
    num_warps = 8
    num_stages = 3

    num_m_blocks = triton.cdiv(S, BLOCK_M)
    num_n_blocks = triton.cdiv(S, BLOCK_N)

    # Launch dQ kernel: grid = (B*H, num_q_blocks)
    grid_dq = (BH, num_m_blocks)
    _dq_kernel[grid_dq](
        Q, K, V, dO, L, D_tensor, dQ,
        S, BLOCK_D,
        Q.stride(0) * H + Q.stride(1) * 0, Q.stride(2), Q.stride(3),
        K.stride(0) * H + K.stride(1) * 0, K.stride(2), K.stride(3),
        V.stride(0) * H + V.stride(1) * 0, V.stride(2), V.stride(3),
        dO.stride(0) * H + dO.stride(1) * 0, dO.stride(2), dO.stride(3),
        L.stride(0) * H + L.stride(1) * 0, L.stride(2),
        D_tensor.stride(0), D_tensor.stride(1),
        dQ.stride(0) * H + dQ.stride(1) * 0, dQ.stride(2), dQ.stride(3),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    # Launch dKV kernel: grid = (B*H, num_kv_blocks)
    grid_dkv = (BH, num_n_blocks)
    _dkv_kernel[grid_dkv](
        Q, K, V, dO, L, D_tensor, dK, dV,
        S, BLOCK_D,
        Q.stride(0) * H + Q.stride(1) * 0, Q.stride(2), Q.stride(3),
        K.stride(0) * H + K.stride(1) * 0, K.stride(2), K.stride(3),
        V.stride(0) * H + V.stride(1) * 0, V.stride(2), V.stride(3),
        dO.stride(0) * H + dO.stride(1) * 0, dO.stride(2), dO.stride(3),
        L.stride(0) * H + L.stride(1) * 0, L.stride(2),
        D_tensor.stride(0), D_tensor.stride(1),
        dK.stride(0) * H + dK.stride(1) * 0, dK.stride(2), dK.stride(3),
        dV.stride(0) * H + dV.stride(1) * 0, dV.stride(2), dV.stride(3),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        num_warps=num_warps,
        num_stages=num_stages,
    )