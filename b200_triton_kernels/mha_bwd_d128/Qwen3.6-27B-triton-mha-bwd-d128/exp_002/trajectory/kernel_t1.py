import torch
import triton
import triton.language as tl


@triton.jit
def _mha_bwd_dq_kernel(
    Q, K, V, dO, L, dQ,
    B, H, S,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute dQ: one program per (B, H, Q_tile)."""
    pid = tl.program_id(0)
    bh, qi = pid // H, pid % H
    b = bh // H
    h = bh % H

    inv_sqrt_d = 1.0 / tl.sqrt(tl.float32(BLOCK_D))

    Q_base = Q + b * stride_qb + h * stride_qh
    K_base = K + b * stride_kb + h * stride_kh
    V_base = V + b * stride_vb + h * stride_vh
    dO_base = dO + b * stride_dob + h * stride_doh
    L_base = L + b * stride_lb + h * stride_lh
    dQ_base = dQ + b * stride_dqb + h * stride_dqh

    offs_d = tl.arange(0, BLOCK_D)
    qs = qi * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
    mq = qs < S

    Q_t = tl.load(Q_base + qs[:, None] * stride_qs + offs_d[None, :] * stride_qd,
                  mask=mq[:, None], other=0.0)
    dO_t = tl.load(dO_base + qs[:, None] * stride_dos + offs_d[None, :] * stride_dod,
                   mask=mq[:, None], other=0.0)
    L_t = tl.load(L_base + qs * stride_ls, mask=mq, other=0.0)

    num_kv_tiles = tl.cdiv(S, BLOCK_SEQ)
    lse_acc = tl.zeros((BLOCK_SEQ,), dtype=tl.float32)
    acc_WK = tl.zeros((BLOCK_SEQ, BLOCK_D), dtype=tl.float32)
    acc_PK = tl.zeros((BLOCK_SEQ, BLOCK_D), dtype=tl.float32)

    for ki in range(num_kv_tiles):
        ks = ki * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
        mk = ks < S

        K_t = tl.load(K_base + ks[:, None] * stride_ks + offs_d[None, :] * stride_kd,
                      mask=mk[:, None], other=0.0)
        V_t = tl.load(V_base + ks[:, None] * stride_vs + offs_d[None, :] * stride_vd,
                      mask=mk[:, None], other=0.0)

        scores = tl.dot(Q_t, K_t.T) * inv_sqrt_d
        attn = tl.exp(scores - L_t[:, None])
        vmask = mq[:, None] & mk[None, :]
        attn = tl.where(vmask, attn, 0.0)

        dattn = tl.dot(dO_t, V_t.T)
        W = attn * dattn

        lse_acc = lse_acc + tl.sum(W, axis=1)
        acc_WK = acc_WK + tl.dot(W, K_t)
        acc_PK = acc_PK + tl.dot(attn, K_t)

    tl.store(dQ_base + qs[:, None] * stride_dqs + offs_d[None, :] * stride_dqd,
             ((acc_WK - lse_acc[:, None] * acc_PK) * inv_sqrt_d).to(dQ.dtype.element_ty),
             mask=mq[:, None])


@triton.jit
def _mha_bwd_dkv_kernel(
    Q, K, V, dO, L, dK, dV,
    B, H, S,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute dK and dV: one program per (B, H, KV_tile)."""
    pid = tl.program_id(0)
    bh, ki = pid // H, pid % H
    b = bh // H
    h = bh % H

    inv_sqrt_d = 1.0 / tl.sqrt(tl.float32(BLOCK_D))

    Q_base = Q + b * stride_qb + h * stride_qh
    K_base = K + b * stride_kb + h * stride_kh
    V_base = V + b * stride_vb + h * stride_vh
    dO_base = dO + b * stride_dob + h * stride_doh
    L_base = L + b * stride_lb + h * stride_lh
    dK_base = dK + b * stride_dkb + h * stride_dkh
    dV_base = dV + b * stride_dvb + h * stride_dvh

    offs_d = tl.arange(0, BLOCK_D)
    ks = ki * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
    mk = ks < S

    K_t = tl.load(K_base + ks[:, None] * stride_ks + offs_d[None, :] * stride_kd,
                  mask=mk[:, None], other=0.0)
    V_t = tl.load(V_base + ks[:, None] * stride_vs + offs_d[None, :] * stride_vd,
                  mask=mk[:, None], other=0.0)

    num_q_tiles = tl.cdiv(S, BLOCK_SEQ)
    lse_acc = tl.zeros((BLOCK_SEQ,), dtype=tl.float32)

    for qi in range(num_q_tiles):
        qs = qi * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
        mq = qs < S

        Q_t = tl.load(Q_base + qs[:, None] * stride_qs + offs_d[None, :] * stride_qd,
                      mask=mq[:, None], other=0.0)
        dO_t = tl.load(dO_base + qs[:, None] * stride_dos + offs_d[None, :] * stride_dod,
                       mask=mq[:, None], other=0.0)
        L_t = tl.load(L_base + qs * stride_ls, mask=mq, other=0.0)

        scores = tl.dot(Q_t, K_t.T) * inv_sqrt_d
        attn = tl.exp(scores - L_t[:, None])
        vmask = mq[:, None] & mk[None, :]
        attn = tl.where(vmask, attn, 0.0)

        dattn = tl.dot(dO_t, V_t.T)
        W = attn * dattn
        lse_acc = lse_acc + tl.sum(W, axis=1)

        adjusted = attn * lse_acc[:, None]
        tl.atomic_add(
            dK_base + ks[:, None] * stride_dks + offs_d[None, :] * stride_dkd,
            (tl.dot((W - adjusted).to(dK.dtype.element_ty).T, Q_t) * inv_sqrt_d).to(dK.dtype.element_ty),
            mask=mk[:, None],
        )
        tl.atomic_add(
            dV_base + ks[:, None] * stride_dvs + offs_d[None, :] * stride_dvd,
            (tl.dot(attn.to(dV.dtype.element_ty).T, dO_t)).to(dV.dtype.element_ty),
            mask=mk[:, None],
        )


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Multi-head attention backward: compute dQ, dK, dV."""
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape

    if L.dim() == 4:
        L = L.squeeze(-1)

    BLOCK_SEQ = 64
    num_q_tiles = triton.cdiv(S, BLOCK_SEQ)
    num_kv_tiles = triton.cdiv(S, BLOCK_SEQ)

    strides_Q = [Q.stride(i) for i in range(4)]
    strides_K = [K.stride(i) for i in range(4)]
    strides_V = [V.stride(i) for i in range(4)]
    strides_dO = [dO.stride(i) for i in range(4)]
    strides_L = [L.stride(i) for i in range(3)]
    strides_dQ = [dQ.stride(i) for i in range(4)]
    strides_dK = [dK.stride(i) for i in range(4)]
    strides_dV = [dV.stride(i) for i in range(4)]

    stride_args = (
        *strides_Q, *strides_K, *strides_V,
        *strides_dO, *strides_L,
        *strides_dQ, *strides_dK, *strides_dV,
    )

    # First pass: compute dQ (direct store, no atomics)
    grid_dq = (B * H * num_q_tiles,)
    _mha_bwd_dq_kernel[grid_dq](
        Q, K, V, dO, L, dQ,
        B, H, S,
        *stride_args,
        BLOCK_SEQ=BLOCK_SEQ,
        BLOCK_D=D,
        num_warps=4,
        num_stages=2,
    )

    # Zero dK and dV before atomic accumulation
    dK.zero_()
    dV.zero_()

    # Second pass: compute dK and dV (atomic add over Q dimension)
    grid_dkv = (B * H * num_kv_tiles,)
    _mha_bwd_dkv_kernel[grid_dkv](
        Q, K, V, dO, L, dK, dV,
        B, H, S,
        *stride_args,
        BLOCK_SEQ=BLOCK_SEQ,
        BLOCK_D=D,
        num_warps=4,
        num_stages=2,
    )