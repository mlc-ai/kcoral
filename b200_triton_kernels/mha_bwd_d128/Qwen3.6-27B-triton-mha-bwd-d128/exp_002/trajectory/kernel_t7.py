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
    BS: tl.constexpr,
    BD: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H
    inv_sd = 1.0 / tl.sqrt(tl.float32(BD))

    off_qb = Q_ptr + b * stride_qb + h * stride_qh
    off_kb = K_ptr + b * stride_kb + h * stride_kh
    off_vb = V_ptr + b * stride_vb + h * stride_vh
    off_ob = dO_ptr + b * stride_dob + h * stride_doh
    off_lb = L_ptr + b * stride_lb + h * stride_lh
    off_qo = dQ_ptr + b * stride_dqb + h * stride_dqh

    od = tl.arange(0, BD)
    mq_mask = tl.full((BS,), True, dtype=tl.int1)
    mk_mask = tl.full((BS,), True, dtype=tl.int1)
    if S % BS != 0:
        mq_mask = tl.where(tl.arange(0, BS) < (S % BS), True, False)

    nk = tl.cdiv(S, BS)

    for qi in range(nk):
        qs = qi * BS + tl.arange(0, BS)
        last_tile = (qi == nk - 1)
        if last_tile:
            mq = mq_mask
        else:
            mq = tl.full((BS,), True, dtype=tl.int1)

        pQ = off_qb + qs[:, None] * stride_qs + od[None, :] * stride_qd
        pO = off_ob + qs[:, None] * stride_dos + od[None, :] * stride_dod
        q = tl.load(pQ, mask=mq[:, None], other=0.0)
        do = tl.load(pO, mask=mq[:, None], other=0.0)
        lt = tl.load(off_lb + qs * stride_ls, mask=mq, other=0.0)

        lw = tl.zeros((BS,), dtype=tl.float32)
        wk = tl.zeros((BS, BD), dtype=tl.float32)
        pk = tl.zeros((BS, BD), dtype=tl.float32)

        for ki in range(nk):
            ks = ki * BS + tl.arange(0, BS)
            last_k = (ki == nk - 1)
            if last_k:
                mk = mq_mask
            else:
                mk = tl.full((BS,), True, dtype=tl.int1)

            k = tl.load(off_kb + ks[:, None] * stride_ks + od[None, :] * stride_kd,
                        mask=mk[:, None], other=0.0)
            v = tl.load(off_vb + ks[:, None] * stride_vs + od[None, :] * stride_vd,
                        mask=mk[:, None], other=0.0)
            sc = tl.dot(q, k.T) * inv_sd
            at = tl.exp(sc - lt[:, None])
            vmask = mq[:, None] & mk[None, :]
            at = tl.where(vmask, at, 0.0)
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
    B, H, S, D,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    BS: tl.constexpr,
    BD: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H
    inv_sd = 1.0 / tl.sqrt(tl.float32(BD))

    off_qb = Q_ptr + b * stride_qb + h * stride_qh
    off_kb = K_ptr + b * stride_kb + h * stride_kh
    off_vb = V_ptr + b * stride_vb + h * stride_vh
    off_ob = dO_ptr + b * stride_dob + h * stride_doh
    off_lb = L_ptr + b * stride_lb + h * stride_lh
    off_kk = dK_ptr + b * stride_dkb + h * stride_dkh
    off_vk = dV_ptr + b * stride_dvb + h * stride_dvh

    od = tl.arange(0, BD)
    nq = tl.cdiv(S, BS)
    nm = tl.cdiv(S, BS)

    for ki in range(nm):
        ks = ki * BS + tl.arange(0, BS)
        last_k = (ki == nm - 1)
        mk = tl.full((BS,), True, dtype=tl.int1)
        if last_k and (S % BS != 0):
            mk = tl.where(tl.arange(0, BS) < (S % BS), True, False)

        k = tl.load(off_kb + ks[:, None] * stride_ks + od[None, :] * stride_kd,
                    mask=mk[:, None], other=0.0)
        v = tl.load(off_vb + ks[:, None] * stride_vs + od[None, :] * stride_vd,
                    mask=mk[:, None], other=0.0)

        adk = tl.zeros((BS, BD), dtype=tl.float32)
        adv = tl.zeros((BS, BD), dtype=tl.float32)
        lpk = tl.zeros((BS,), dtype=tl.float32)

        for qi in range(nq):
            qs = qi * BS + tl.arange(0, BS)
            last_q = (qi == nq - 1)
            mq = tl.full((BS,), True, dtype=tl.int1)
            if last_q and (S % BS != 0):
                mq = tl.where(tl.arange(0, BS) < (S % BS), True, False)

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
    BD = D  # Must be power of two and match head dimension
    grid = (B * H,)

    strides = []
    for t in [Q, K, V, dO]:
        strides.extend([t.stride(i) for i in range(4)])
    strides.extend([L.stride(i) for i in range(3)])
    strides.extend([dQ.stride(i) for i in range(4)])
    strides.extend([dK.stride(i) for i in range(4)])
    strides.extend([dV.stride(i) for i in range(4)])

    _mha_bwd_dq_kernel[grid](
        Q, K, V, dO, L, dQ,
        B, H, S, D,
        *strides[:19],
        *strides[19:23],
        BS=BS,
        BD=BD,
        num_warps=4,
        num_stages=2,
    )

    _mha_bwd_dkv_kernel[grid](
        Q, K, V, dO, L, dK, dV,
        B, H, S, D,
        *strides[:19],
        *strides[23:31],
        BS=BS,
        BD=BD,
        num_warps=4,
        num_stages=2,
    )