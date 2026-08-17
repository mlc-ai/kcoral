import torch
import triton
import triton.language as tl


@triton.jit
def _mha_bwd_kernel(
    Q, K, V, dO, L,
    dQ, dK, dV,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    B, H, S,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Multi-head attention backward kernel.

    Each program instance handles one (batch, head) combination.

    Algorithm:
      For each Q tile:
        For each KV tile (first pass):
          Compute scores, attn, dattn, W=attn*dattn
          Accumulate lse correction term and WK/AK terms for dQ
          Atomic-add contributions to dV and dK term1
        Store dQ directly (no atomics needed)
        For each KV tile (second pass):
          Recompute attn, apply LSE correction, atomic-subtract dK term2
    """
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H

    inv_scale = 1.0 / tl.sqrt(tl.float32(BLOCK_D))

    # Base pointers offset for this (b, h)
    Q_base = Q + b * stride_qb + h * stride_qh
    K_base = K + b * stride_kb + h * stride_kh
    V_base = V + b * stride_vb + h * stride_vh
    dO_base = dO + b * stride_dob + h * stride_doh
    L_base = L + b * stride_lb + h * stride_lh
    dQ_base = dQ + b * stride_dqb + h * stride_dqh
    dK_base = dK + b * stride_dkb + h * stride_dkh
    dV_base = dV + b * stride_dvb + h * stride_dvh

    offs_d = tl.arange(0, BLOCK_D)
    num_tiles = tl.cdiv(S, BLOCK_SEQ)

    for qi in range(num_tiles):
        qs = qi * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
        mq = qs < S

        # Load Q[BS, BD], dO[BS, BD], L[BS] for this Q tile
        Q_t = tl.load(
            Q_base + qs[:, None] * stride_qs + offs_d[None, :] * stride_qd,
            mask=mq[:, None], other=0.0,
        )
        dO_t = tl.load(
            dO_base + qs[:, None] * stride_dos + offs_d[None, :] * stride_dod,
            mask=mq[:, None], other=0.0,
        )
        L_t = tl.load(L_base + qs * stride_ls, mask=mq, other=0.0)

        # Accumulators for dQ
        lse_acc = tl.zeros((BLOCK_SEQ,), dtype=tl.float32)
        acc_WK = tl.zeros((BLOCK_SEQ, BLOCK_D), dtype=tl.float32)
        acc_AK = tl.zeros((BLOCK_SEQ, BLOCK_D), dtype=tl.float32)

        # --- First pass over KV tiles ---
        for ki in range(num_tiles):
            ks = ki * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
            mk = ks < S

            K_t = tl.load(
                K_base + ks[:, None] * stride_ks + offs_d[None, :] * stride_kd,
                mask=mk[:, None], other=0.0,
            )
            V_t = tl.load(
                V_base + ks[:, None] * stride_vs + offs_d[None, :] * stride_vd,
                mask=mk[:, None], other=0.0,
            )

            # scores[BS, BS] = Q @ K^T / sqrt(d)
            scores = tl.dot(Q_t, K_t.T) * inv_scale

            # attn = exp(scores - L); mask invalid positions
            attn = tl.exp(scores - L_t[:, None])
            vmask = mq[:, None] & mk[None, :]
            attn = tl.where(vmask, attn, 0.0)

            # dattn = dO @ V^T; W = attn * dattn
            dattn = tl.dot(dO_t, V_t.T)
            W = attn * dattn

            # Accumulate for dQ
            lse_acc = lse_acc + tl.sum(W, axis=1)
            acc_WK = acc_WK + tl.dot(W.to(tl.bfloat16), K_t)
            acc_AK = acc_AK + tl.dot(attn.to(tl.bfloat16), K_t)

            # dV[kv, d] += sum_q attn[q,kv]*dO[q,d]  (no scaling factor)
            tl.atomic_add(
                dV_base + ks[:, None] * stride_dvs + offs_d[None, :] * stride_dvd,
                (tl.dot(attn.to(tl.bfloat16).T, dO_t)).to(tl.bfloat16),
                mask=mk[:, None],
            )

            # dK term1: sum_q W[q,kv]*Q[q,d]/sqrt(d)
            tl.atomic_add(
                dK_base + ks[:, None] * stride_dks + offs_d[None, :] * stride_dkd,
                (tl.dot(W.to(tl.bfloat16).T, Q_t) * inv_scale).to(tl.bfloat16),
                mask=mk[:, None],
            )

        # Store dQ directly: (acc_WK - lse * acc_AK) / sqrt(d)
        tl.store(
            dQ_base + qs[:, None] * stride_dqs + offs_d[None, :] * stride_dqd,
            ((acc_WK - lse_acc[:, None] * acc_AK) * inv_scale).to(tl.bfloat16),
            mask=mq[:, None],
        )

        # --- Second pass: dK correction term ---
        for ki in range(num_tiles):
            ks = ki * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
            mk = ks < S

            K_t = tl.load(
                K_base + ks[:, None] * stride_ks + offs_d[None, :] * stride_kd,
                mask=mk[:, None], other=0.0,
            )

            scores = tl.dot(Q_t, K_t.T) * inv_scale
            attn = tl.exp(scores - L_t[:, None])
            vmask = mq[:, None] & mk[None, :]
            attn = tl.where(vmask, attn, 0.0)

            # Correction: -sum_q attn[q,kv]*lse[q]*Q[q,d]/sqrt(d)
            adjusted = attn * lse_acc[:, None]
            tl.atomic_add(
                dK_base + ks[:, None] * stride_dks + offs_d[None, :] * stride_dkd,
                (-tl.dot(adjusted.to(tl.bfloat16).T, Q_t) * inv_scale).to(tl.bfloat16),
                mask=mk[:, None],
            )


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute multi-head attention backward: dQ, dK, dV.

    Args (in definition order):
        Inputs: Q, K, V, O, dO, L
        Outputs: dQ, dK, dV (preallocated, written in-place)
    """
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape

    # Handle L being [B, H, S] vs [B, H, S, 1]
    if L.dim() == 4:
        L = L.squeeze(-1)

    # Zero outputs so atomic accumulation starts from zero
    dQ.zero_()
    dK.zero_()
    dV.zero_()

    BLOCK_SEQ = 64
    grid = (B * H,)

    _mha_bwd_kernel[grid](
        Q, K, V, dO, L,
        dQ, dK, dV,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        B, H, S,
        BLOCK_SEQ=BLOCK_SEQ,
        BLOCK_D=D,
        num_warps=4,
        num_stages=2,
    )