import torch
import triton
import triton.language as tl


@triton.jit
def _mha_bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, dQ_ptr,
    B, H, S, D,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H

    inv_sqrt_d = 1.0 / tl.sqrt(tl.float32(BLOCK_D))

    Q_base = Q_ptr + b * stride_qb + h * stride_qh
    K_base = K_ptr + b * stride_kb + h * stride_kh
    V_base = V_ptr + b * stride_vb + h * stride_vh
    dO_base = dO_ptr + b * stride_dob + h * stride_doh
    L_base = L_ptr + b * stride_lb + h * stride_lh
    dQ_base = dQ_ptr + b * stride_dqb + h * stride_dqh

    offs_d = tl.arange(0, BLOCK_D)
    num_kv_tiles = tl.cdiv(S, BLOCK_S)

    for qi in range(tl.cdiv(S, BLOCK_S)):
        qs = qi * BLOCK_S + tl.arange(0, BLOCK_S)
        mq = qs < S

        Q_ptrs = Q_base + qs[:, None] * stride_qs + offs_d[None, :] * stride_qd
        dO_ptrs = dO_base + qs[:, None] * stride_dos + offs_d[None, :] * stride_dod
        dQ_ptrs = dQ_base + qs[:, None] * stride_dqs + offs_d[None, :] * stride_dqd

        Q_t = tl.load(Q_ptrs, mask=mq[:, None], other=0.0)
        dO_t = tl.load(dO_ptrs, mask=mq[:, None], other=0.0)
        L_t = tl.load(L_base + qs * stride_ls, mask=mq, other=0.0)

        lse_acc = tl.zeros((BLOCK_S,), dtype=tl.float32)
        acc_WK = tl.zeros((BLOCK_S, BLOCK_D), dtype=tl.float32)
        acc_PK = tl.zeros((BLOCK_S, BLOCK_D), dtype=tl.float32)

        for ki in range(num_kv_tiles):
            ks = ki * BLOCK_S + tl.arange(0, BLOCK_S)
            mk = ks < S

            K_ptrs = K_base + ks[:, None] * stride_ks + offs_d[None, :] * stride_kd
            V_ptrs = V_base + ks[:, None] * stride_vs + offs_d[None, :] * stride_vd

            K_t = tl.load(K_ptrs, mask=mk[:, None], other=0.0)
            V_t = tl.load(V_ptrs, mask=mk[:, None], other=0.0)

            scores = tl.dot(Q_t, K_t.T) * inv_sqrt_d
            attn = tl.exp(scores - L_t[:, None])
            vmask = mq[:, None] & mk[None, :]
            attn = tl.where(vmask, attn, 0.0)

            dattn = tl.dot(dO_t, V_t.T)
            W = attn * dattn

            lse_acc = lse_acc + tl.sum(W, axis=1)
            acc_WK = acc_WK + tl.dot(W.to(tl.bfloat16), K_t)
            acc_PK = acc_PK + tl.dot(attn.to(tl.bfloat16), K_t)

        dQ_val = ((acc_WK - lse_acc[:, None] * acc_PK) * inv_sqrt_d).to(tl.bfloat16)
        tl.store(dQ_ptrs, dQ_val, mask=mq[:, None])


@triton.jit
def _mha_bwd_dkv_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, dK_ptr, dV_ptr,
    B, H, S, D,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H

    inv_sqrt_d = 1.0 / tl.sqrt(tl.float32(BLOCK_D))

    Q_base = Q_ptr + b * stride_qb + h * stride_qh
    K_base = K_ptr + b * stride_kb + h * stride_kh
    V_base = V_ptr + b * stride_vb + h * stride_vh
    dO_base = dO_ptr + b * stride_dob + h * stride_doh
    L_base = L_ptr + b * stride_lb + h * stride_lh
    dK_base = dK_ptr + b * stride_dkb + h * stride_dkh
    dV_base = dV_ptr + b * stride_dvb + h * stride_dvh

    offs_d = tl.arange(0, BLOCK_D)
    num_q_tiles = tl.cdiv(S, BLOCK_S)

    # Accumulate dK and dV directly in registers, then store once per kv_tile
    for ki in range(tl.cdiv(S, BLOCK_S)):
        ks = ki * BLOCK_S + tl.arange(0, BLOCK_S)
        mk = ks < S

        K_ptrs = K_base + ks[:, None] * stride_ks + offs_d[None, :] * stride_kd
        V_ptrs = V_base + ks[:, None] * stride_vs + offs_d[None, :] * stride_vd
        dK_ptrs = dK_base + ks[:, None] * stride_dks + offs_d[None, :] * stride_dkd
        dV_ptrs = dV_base + ks[:, None] * stride_dvs + offs_d[None, :] * stride_dvd

        K_t = tl.load(K_ptrs, mask=mk[:, None], other=0.0)
        V_t = tl.load(V_ptrs, mask=mk[:, None], other=0.0)

        acc_dK = tl.zeros((BLOCK_S, BLOCK_D), dtype=tl.float32)
        acc_dV = tl.zeros((BLOCK_S, BLOCK_D), dtype=tl.float32)

        for qi in range(num_q_tiles):
            qs = qi * BLOCK_S + tl.arange(0, BLOCK_S)
            mq = qs < S

            Q_ptrs = Q_base + qs[:, None] * stride_qs + offs_d[None, :] * stride_qd
            dO_ptrs = dO_base + qs[:, None] * stride_dos + offs_d[None, :] * stride_dod

            Q_t = tl.load(Q_ptrs, mask=mq[:, None], other=0.0)
            dO_t = tl.load(dO_ptrs, mask=mq[:, None], other=0.0)
            L_t = tl.load(L_base + qs * stride_ls, mask=mq, other=0.0)

            scores = tl.dot(Q_t, K_t.T) * inv_sqrt_d
            attn = tl.exp(scores - L_t[:, None])
            vmask = mq[:, None] & mk[None, :]
            attn = tl.where(vmask, attn, 0.0)

            dattn = tl.dot(dO_t, V_t.T)
            W = attn * dattn
            PK_q = tl.sum(W, axis=1)

            # dE[q,kv] = W - attn * sum_W_over_kv  ... but we need running PK correction
            corrected_E = W - attn * PK_q[:, None]
            corr = corrected_E.to(tl.bfloat16)

            acc_dK = acc_dK + tl.dot(corr.T, Q_t) * inv_sqrt_d
            acc_dV = acc_dV + tl.dot(attn.to(tl.bfloat16).T, dO_t)

        tl.store(dK_ptrs, acc_dK.to(tl.bfloat16), mask=mk[:, None])
        tl.store(dV_ptrs, acc_dV.to(tl.bfloat16), mask=mk[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Multi-head attention backward: compute dQ, dK, dV."""
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape

    if L.dim() == 4:
        L = L.squeeze(-1)

    BLOCK_S = 64

    sq = Q.stride(0); sh = Q.stride(1); ss = Q.stride(2); sd = Q.stride(3)
    strides_Q = (sq, sh, ss, sd)
    strides_K = (K.stride(0), K.stride(1), K.stride(2), K.stride(3))
    strides_V = (V.stride(0), V.stride(1), V.stride(2), V.stride(3))
    strides_dO = (dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3))
    strides_L = (L.stride(0), L.stride(1), L.stride(2))
    strides_dQ = (dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3))
    strides_dK = (dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3))
    strides_dV = (dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3))

    all_strides = (*strides_Q, *strides_K, *strides_V,
                   *strides_dO, *strides_L,
                   *strides_dQ, *strides_dK, *strides_dV)

    grid = (B * H,)

    # Launch dQ kernel
    _mha_bwd_dq_kernel[grid](
        Q, K, V, dO, L, dQ,
        B, H, S, D,
        *all_strides[:19],  # Q,K,V,dO,L strides (16+3)
        *all_strides[19:23],  # dQ strides
        BLOCK_S=BLOCK_S,
        BLOCK_D=D,
        num_warps=4,
        num_stages=2,
    )

    # Zero outputs before storing from dkv kernel
    dK.zero_()
    dV.zero_()

    # Launch dK/dV kernel
    _mha_bwd_dkv_kernel[grid](
        Q, K, V, dO, L, dK, dV,
        B, H, S, D,
        *all_strides,
        BLOCK_S=BLOCK_S,
        BLOCK_D=D,
        num_warps=4,
        num_stages=2,
    )