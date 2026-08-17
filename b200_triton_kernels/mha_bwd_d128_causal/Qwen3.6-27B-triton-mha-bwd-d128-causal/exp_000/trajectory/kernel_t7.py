import math

import torch
import triton
import triton.language as tl


@triton.jit
def _dq_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, O_ptr, L_ptr, dQ_ptr,
    stride_b, stride_h, stride_s, stride_d,
    stride_l_b, stride_l_h, stride_l_s,
    S, D, scale,
    TILE_S: tl.constexpr,
    TILE_D: tl.constexpr,
    H: tl.constexpr,
):
    """dQ[s,:] = scale * (W@K - bias[:,None]*(P@K))
    Bias: B[s] = sum_d(dO[s,d]*O[s,d]) — proven identity from logsumexp backward.
    Grid: (B*H, cdiv(S,TILE_S)). Fix s-tile, stream t-tiles."""
    pid_bh = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H
    bh_off = pid_b * stride_b + pid_h * stride_h
    l_off = pid_b * stride_l_b + pid_h * stride_l_h

    q_base = Q_ptr + bh_off
    k_base = K_ptr + bh_off
    v_base = V_ptr + bh_off
    do_base = dO_ptr + bh_off
    o_base = O_ptr + bh_off
    dq_base = dQ_ptr + bh_off
    l_base = L_ptr + l_off

    qs = pid_s * TILE_S + tl.arange(0, TILE_S)
    ds = tl.arange(0, TILE_D)
    mask_qs = qs < S
    mask_ds = ds < D
    mask_qsd = mask_qs[:, None] & mask_ds[None, :]

    Q_s = tl.load(q_base + qs[:, None] * stride_s + ds[None, :] * stride_d,
                  mask=mask_qsd, other=0.0).to(tl.float32)
    dO_s = tl.load(do_base + qs[:, None] * stride_s + ds[None, :] * stride_d,
                   mask=mask_qsd, other=0.0).to(tl.float32)
    O_s = tl.load(o_base + qs[:, None] * stride_s + ds[None, :] * stride_d,
                  mask=mask_qsd, other=0.0).to(tl.float32)
    L_s = tl.load(l_base + qs * stride_l_s, mask=mask_qs, other=0.0)

    # Bias: B[s] = sum_d dO[s,d] * O[s,d]
    bias_s = tl.sum(dO_s * O_s, axis=1)

    acc_wK = tl.zeros((TILE_S, TILE_D), dtype=tl.float32)
    acc_pK = tl.zeros((TILE_S, TILE_D), dtype=tl.float32)

    for t_i in range(tl.cdiv(S, TILE_S)):
        ts = t_i * TILE_S + tl.arange(0, TILE_S)
        mask_ts = ts < S
        mask_tsd = mask_ts[:, None] & mask_ds[None, :]

        K_t = tl.load(k_base + ts[:, None] * stride_s + ds[None, :] * stride_d,
                       mask=mask_tsd, other=0.0).to(tl.float32)
        V_t = tl.load(v_base + ts[:, None] * stride_s + ds[None, :] * stride_d,
                       mask=mask_tsd, other=0.0).to(tl.float32)

        # scores = Q_s @ K_t^T  [S,T]
        scores = tl.dot(Q_s, K_t) * scale
        P = tl.exp(scores - L_s[:, None])

        causal = qs[:, None] >= ts[None, :]
        valid = mask_qs[:, None] & mask_ts[None, :]
        P = tl.where(valid & causal, P, 0.0)

        # dP = dO_s @ V_t^T  [S,T]
        dP = tl.dot(dO_s, V_t)
        W = P * dP

        # W @ K_t^T -> tl.dot(W[S,T], K_t[T,D]) = [S,D]
        acc_wK = tl.dot(W, K_t, acc=acc_wK)
        acc_pK = tl.dot(P, K_t, acc=acc_pK)

    dQ_val = scale * (acc_wK - bias_s[:, None] * acc_pK)
    tl.store(dq_base + qs[:, None] * stride_s + ds[None, :] * stride_d,
             dQ_val.to(dtype=tl.bfloat16), mask=mask_qsd)


@triton.jit
def _dk_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, O_ptr, L_ptr, dK_ptr,
    stride_b, stride_h, stride_s, stride_d,
    stride_l_b, stride_l_h, stride_l_s,
    S, D, scale,
    TILE_S: tl.constexpr,
    TILE_D: tl.constexpr,
    H: tl.constexpr,
):
    """dK[t,:] = scale * (W^T@Q - (P*B)^T@Q)
    Load K,V in [T,D] format; load Q,dO,O,L in [S,D] format per s-tile.
    Use x.T trick for reversed-multiplication: trans(Q_s)[D,S] @ W[S,T] = [D,T].
    Grid: (B*H, cdiv(S,TILE_S)). Fix t-tile, stream s-tiles."""
    pid_bh = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H
    bh_off = pid_b * stride_b + pid_h * stride_h
    l_off = pid_b * stride_l_b + pid_h * stride_l_h

    q_base = Q_ptr + bh_off
    k_base = K_ptr + bh_off
    v_base = V_ptr + bh_off
    do_base = dO_ptr + bh_off
    o_base = O_ptr + bh_off
    dk_base = dK_ptr + bh_off
    l_base = L_ptr + l_off

    ts = pid_t * TILE_S + tl.arange(0, TILE_S)
    ds = tl.arange(0, TILE_D)
    mask_ts = ts < S
    mask_ds = ds < D
    mask_tsd = mask_ts[:, None] & mask_ds[None, :]

    # Load K,V once for this t-tile
    K_t = tl.load(k_base + ts[:, None] * stride_s + ds[None, :] * stride_d,
                   mask=mask_tsd, other=0.0).to(tl.float32)
    V_t = tl.load(v_base + ts[:, None] * stride_s + ds[None, :] * stride_d,
                   mask=mask_tsd, other=0.0).to(tl.float32)

    acc_wQ = tl.zeros((TILE_S, TILE_D), dtype=tl.float32)
    acc_pbQ = tl.zeros((TILE_S, TILE_D), dtype=tl.float32)

    for s_i in range(tl.cdiv(S, TILE_S)):
        ss = s_i * TILE_S + tl.arange(0, TILE_S)
        mask_ss = ss < S
        mask_ssd = mask_ss[:, None] & mask_ds[None, :]

        Q_s = tl.load(q_base + ss[:, None] * stride_s + ds[None, :] * stride_d,
                       mask=mask_ssd, other=0.0).to(tl.float32)
        dO_s = tl.load(do_base + ss[:, None] * stride_s + ds[None, :] * stride_d,
                        mask=mask_ssd, other=0.0).to(tl.float32)
        O_s = tl.load(o_base + ss[:, None] * stride_s + ds[None, :] * stride_d,
                       mask=mask_ssd, other=0.0).to(tl.float32)
        L_s = tl.load(l_base + ss * stride_l_s, mask=mask_ss, other=0.0)

        # scores = Q_s @ K_t^T -> [S,T]
        scores = tl.dot(Q_s, K_t) * scale
        P = tl.exp(scores - L_s[:, None])

        causal = ss[:, None] >= ts[None, :]
        valid = mask_ss[:, None] & mask_ts[None, :]
        P = tl.where(valid & causal, P, 0.0)

        dP = tl.dot(dO_s, V_t)
        W = P * dP

        # Bias: B[s] = sum_d(dO[s,d]*O[s,d])
        bias_s = tl.sum(dO_s * O_s, axis=1)
        P_b = P * bias_s[:, None]

        # W^T@Q: use trans(Q_s)[D,S] @ W[S,T] -> [D,T]
        # Then transpose result to [T,D]
        acc_wQ = tl.dot(Q_s.T, W, acc=acc_wQ)
        acc_pbQ = tl.dot(Q_s.T, P_b, acc=acc_pbQ)

    dK_val = scale * (acc_wQ - acc_pbQ)
    # acc is [TILE_S x TILE_D] = [T,D] since we accumulated [D,T]-shaped products
    # Wait — Q_s.T is [D,S], W is [S,T] -> dot is [D,T] -> need .T for [T,D]
    tl.store(dk_base + ts[:, None] * stride_s + ds[None, :] * stride_d,
             dK_val.T.to(dtype=tl.bfloat16), mask=mask_tsd)


@triton.jit
def _dv_kernel(
    Q_ptr, K_ptr, dO_ptr, L_ptr, dV_ptr,
    stride_b, stride_h, stride_s, stride_d,
    stride_l_b, stride_l_h, stride_l_s,
    S, D, scale,
    TILE_S: tl.constexpr,
    TILE_D: tl.constexpr,
    H: tl.constexpr,
):
    """dV[t,:] = P^T @ dO.  Fix t-tile, stream s-tiles."""
    pid_bh = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H
    bh_off = pid_b * stride_b + pid_h * stride_h
    l_off = pid_b * stride_l_b + pid_h * stride_l_h

    q_base = Q_ptr + bh_off
    k_base = K_ptr + bh_off
    do_base = dO_ptr + bh_off
    dv_base = dV_ptr + bh_off
    l_base = L_ptr + l_off

    ts = pid_t * TILE_S + tl.arange(0, TILE_S)
    ds = tl.arange(0, TILE_D)
    mask_ts = ts < S
    mask_ds = ds < D
    mask_tsd = mask_ts[:, None] & mask_ds[None, :]

    K_t = tl.load(k_base + ts[:, None] * stride_s + ds[None, :] * stride_d,
                   mask=mask_tsd, other=0.0).to(tl.float32)

    acc_dv = tl.zeros((TILE_S, TILE_D), dtype=tl.float32)

    for s_i in range(tl.cdiv(S, TILE_S)):
        ss = s_i * TILE_S + tl.arange(0, TILE_S)
        mask_ss = ss < S
        mask_ssd = mask_ss[:, None] & mask_ds[None, :]

        Q_s = tl.load(q_base + ss[:, None] * stride_s + ds[None, :] * stride_d,
                       mask=mask_ssd, other=0.0).to(tl.float32)
        dO_s = tl.load(do_base + ss[:, None] * stride_s + ds[None, :] * stride_d,
                        mask=mask_ssd, other=0.0).to(tl.float32)
        L_s = tl.load(l_base + ss * stride_l_s, mask=mask_ss, other=0.0)

        scores = tl.dot(Q_s, K_t) * scale
        P = tl.exp(scores - L_s[:, None])

        causal = ss[:, None] >= ts[None, :]
        valid = mask_ss[:, None] & mask_ts[None, :]
        P = tl.where(valid & causal, P, 0.0)

        # P^T @ dO = trans(P)[T,S] @ trans(dO_s)[S,D] = [T,D]
        # Using .T trick: tl.dot(dO_s.T[S,D], P.T[S,T]) hmm no...
        # trans(P) has shape [T,S], trans(dO_s) has shape [S,D]... but in layout
        # P is [S,T], P.T is [T,S]; dO_s is [S,D], dO_s.T is [D,S]
        # We want [T,S] @ [S,D] = [T,D]
        # tl.dot(P.T[D??], dO_s.T[??]) — wait, P.T is [T,S] and dO_s.T is [D,S]
        # tl.dot needs same last dim: S==S ✓
        # tl.dot(P.T, dO_s.T) = [T,S] @ [D,S]^T? NO.
        # tl.dot(A,B) = A @ B^T. So tl.dot(P.T, dO_s.T) = P.T @ dO_s.T^T = P.T @ dO_s
        #   P.T is [T,S], dO_s is [S,D] -> [T,D] ✓

        acc_dv = tl.dot(P.T, dO_s, acc=acc_dv)

    tl.store(dv_base + ts[:, None] * stride_s + ds[None, :] * stride_d,
             acc_dv.to(dtype=tl.bfloat16), mask=mask_tsd)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Causal multi-head attention backward."""
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape

    for t in [Q, K, V, O, dO, L]:
        if not t.is_contiguous():
            locals()[t.data_ptr().__hash__() % 6] = t.contiguous()
    Q_c = Q if Q.is_contiguous() else Q.contiguous()
    K_c = K if K.is_contiguous() else K.contiguous()
    V_c = V if V.is_contiguous() else V.contiguous()
    O_c = O if O.is_contiguous() else O.contiguous()
    dO_c = dO if dO.is_contiguous() else dO.contiguous()
    L_c = L if L.is_contiguous() else L.contiguous()

    if L_c.dim() == 4:
        L_c = L_c.squeeze(-1)

    stride_b = Q_c.stride(0)
    stride_h = Q_c.stride(1)
    stride_s = Q_c.stride(2)
    stride_d = Q_c.stride(3)
    stride_l_b = L_c.stride(0)
    stride_l_h = L_c.stride(1)
    stride_l_s = L_c.stride(2)

    scale = 1.0 / math.sqrt(float(D))

    TILE_S = 64
    TILE_D = D  # 128
    num_bh = B * H
    num_st = triton.cdiv(S, TILE_S)
    grid = (num_bh, num_st)

    _dq_kernel[grid](
        Q_c, K_c, V_c, dO_c, O_c, L_c, dQ,
        stride_b, stride_h, stride_s, stride_d,
        stride_l_b, stride_l_h, stride_l_s,
        S, D, scale,
        TILE_S=TILE_S, TILE_D=TILE_D, H=H,
        num_warps=4, num_stages=2,
    )

    _dk_kernel[grid](
        Q_c, K_c, V_c, dO_c, O_c, L_c, dK,
        stride_b, stride_h, stride_s, stride_d,
        stride_l_b, stride_l_h, stride_l_s,
        S, D, scale,
        TILE_S=TILE_S, TILE_D=TILE_D, H=H,
        num_warps=4, num_stages=2,
    )

    _dv_kernel[grid](
        Q_c, K_c, dO_c, L_c, dV,
        stride_b, stride_h, stride_s, stride_d,
        stride_l_b, stride_l_h, stride_l_s,
        S, D, scale,
        TILE_S=TILE_S, TILE_D=TILE_D, H=H,
        num_warps=4, num_stages=2,
    )