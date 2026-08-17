import torch
import triton
import triton.language as tl


@triton.jit
def _mha_bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, dQ_ptr,
    S, D,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    BS: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H
    inv_sd = 1.0 / tl.sqrt(tl.float32(D))

    off_qb = Q_ptr + b * stride_qb + h * stride_qh
    off_kb = K_ptr + b * stride_kb + h * stride_kh
    off_vb = V_ptr + b * stride_vb + h * stride_vh
    off_ob = dO_ptr + b * stride_dob + h * stride_doh
    off_lb = L_ptr + b * stride_lb + h * stride_lh
    off_qo = dQ_ptr + b * stride_dqb + h * stride_dqh

    od = tl.arange(0, D)
    nk = tl.cdiv(S, BS)

    for qi in range(nk):
        qs = qi * BS + tl.arange(0, BS)
        mq = qs < S
        pQ = off_qb + qs[:, None] * stride_qs + od[None, :] * stride_qd
        pO = off_ob + qs[:, None] * stride_dos + od[None, :] * stride_dod
        q = tl.load(pQ, mask=mq[:, None], other=0.0)
        do = tl.load(pO, mask=mq[:, None], other=0.0)
        lt = tl.load(off_lb + qs * stride_ls, mask=mq, other=0.0)

        lw = tl.zeros((BS,), dtype=tl.float32)
        wk = tl.zeros((BS, D), dtype=tl.float32)
        pk = tl.zeros((BS, D), dtype=tl.float32)

        for ki in range(nk):
            ks = ki * BS + tl.arange(0, BS)
            mk = ks < S
            k = tl.load(off_kb + ks[:, None] * stride_ks + od[None, :] * stride_kd,
                        mask=mk[:, None], other=0.0)
            v = tl.load(off_vb + ks[:, None] * stride_vs + od[None, :] * stride_vd,
                        mask=mk[:, None], other=0.0)
            sc = tl.dot(q, k.T) * inv_sd
            at = tl.exp(sc - lt[:, None])
            at = tl.where(mq[:, None] & mk[None, :], at, 0.0)
            da = tl.dot(do, v.T)
            w = at * da
            lw = lw + tl.sum(w, axis=1)
            wk = wk + tl.dot(w.to(tl.bfloat16), k)
            pk = pk + tl.dot(at.to(tl.bfloat16), k)

        dq = ((wk - lw[:, None] * pk) * inv_sd).to(tl.bfloat16)
        tl.store(off_qo + qs[:, None] * stride_dqs + od[None, :] * stride_dqd,
                 dq, mask=mq[:, None])


@triton.jit
def _mha_bwd_dkv_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, dK_ptr, dV_ptr,
    S, D,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    BS: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H
    inv_sd = 1.0 / tl.sqrt(tl.float32(D))

    off_qb = Q_ptr + b * stride_qb + h * stride_qh
    off_kb = K_ptr + b * stride_kb + h * stride_kh
    off_vb = V_ptr + b * stride_vb + h * stride_vh
    off_ob = dO_ptr + b * stride_dob + h * stride_doh
    off_lb = L_ptr + b * stride_lb + h * stride_lh
    off_kk = dK_ptr + b * stride_dkb + h * stride_dkh
    off_vk = dV_ptr + b * stride_dvb + h * stride_dvh

    od = tl.arange(0, D)
    nq = tl.cdiv(S, BS)

    for ki in range(tl.cdiv(S, BS)):
        ks = ki * BS + tl.arange(0, BS)
        mk = ks < S
        k = tl.load(off_kb + ks[:, None] * stride_ks + od[None, :] * stride_kd,
                    mask=mk[:, None], other=0.0)
        v = tl.load(off_vb + ks[:, None] * stride_vs + od[None, :] * stride_vd,
                    mask=mk[:, None], other=0.0)

        adk = tl.zeros((BS, D), dtype=tl.float32)
        adv = tl.zeros((BS, D), dtype=tl.float32)
        lpk = tl.zeros((BS,), dtype=tl.float32)

        for qi in range(nq):
            qs = qi * BS + tl.arange(0, BS)
            mq = qs < S
            q = tl.load(off_qb + qs[:, None] * stride_qs + od[None, :] * stride_qd,
                        mask=mq[:, None], other=0.0)
            do = tl.load(off_ob + qs[:, None] * stride_dos + od[None, :] * stride_dod,
                         mask=mq[:, None], other=0.0)
            lt = tl.load(off_lb + qs * stride_ls, mask=mq, other=0.0)
            sc = tl.dot(q, k.T) * inv_sd
            at = tl.exp(sc - lt[:, None])
            at = tl.where(mq[:, None] & mk[None, :], at, 0.0)
            da = tl.dot(do, v.T)
            w = at * da
            lpk = lpk + tl.sum(w, axis=1)
            ce = (w - at * lpk[:, None]).to(tl.bfloat16)
            adk = adk + tl.dot(ce.T, q) * inv_sd
            adv = adv + tl.dot(at.to(tl.bfloat16).T, do)

        tl.store(off_kk + ks[:, None] * stride_dks + od[None, :] * stride_dkd,
                 adk.to(tl.bfloat16), mask=mk[:, None])
        tl.store(off_vk + ks[:, None] * stride_dvs + od[None, :] * stride_dvd,
                 adv.to(tl.bfloat16), mask=mk[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Multi-head attention backward: compute dQ, dK, dV."""
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape

    if L.dim() == 4:
        L = L.squeeze(-1)

    BS = 64
    grid = (B * H,)

    sq0, sq1, sq2, sq3 = Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3)
    sk0, sk1, sk2, sk3 = K.stride(0), K.stride(1), K.stride(2), K.stride(3)
    sv0, sv1, sv2, sv3 = V.stride(0), V.stride(1), V.stride(2), V.stride(3)
    so0, so1, so2, so3 = dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3)
    sl0, sl1, sl2 = L.stride(0), L.stride(1), L.stride(2)
    sdq0, sdq1, sdq2, sdq3 = dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3)
    sdk0, sdk1, sdk2, sdk3 = dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3)
    sdu0, sdu1, sdu2, sdu3 = dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3)

    # Launch dQ kernel
    _mha_bwd_dq_kernel[grid](
        Q, K, V, dO, L, dQ,
        S, D,
        sq0, sq1, sq2, sq3,
        sk0, sk1, sk2, sk3,
        sv0, sv1, sv2, sv3,
        so0, so1, so2, so3,
        sl0, sl1, sl2,
        sdq0, sdq1, sdq2, sdq3,
        BS=BS,
        H=H,
        num_warps=4,
        num_stages=2,
    )

    # Launch dK/dV kernel
    _mha_bwd_dkv_kernel[grid](
        Q, K, V, dO, L, dK, dV,
        S, D,
        sq0, sq1, sq2, sq3,
        sk0, sk1, sk2, sk3,
        sv0, sv1, sv2, sv3,
        so0, so1, so2, so3,
        sl0, sl1, sl2,
        sdk0, sdk1, sdk2, sdk3,
        sdu0, sdu1, sdu2, sdu3,
        BS=BS,
        H=H,
        num_warps=4,
        num_stages=2,
    )