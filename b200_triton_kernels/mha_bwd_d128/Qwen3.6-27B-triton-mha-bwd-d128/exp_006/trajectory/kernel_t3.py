import torch
import triton
import triton.language as tl


@triton.jit
def _dq_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dQ_ptr,
    S, D_MODEL,
    BH_stride_q, S_stride_q, D_stride_q,
    BH_stride_k, S_stride_k, D_stride_k,
    BH_stride_v, S_stride_v, D_stride_v,
    BH_stride_o, S_stride_o, D_stride_o,
    BH_stride_do, S_stride_do, D_stride_do,
    BH_stride_l, S_stride_l,
    BH_stride_dq, S_stride_dq, D_stride_dq,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Compute dQ for multi-head attention backward."""
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    q_base = Q_ptr + pid_bh * BH_stride_q
    k_base = K_ptr + pid_bh * BH_stride_k
    v_base = V_ptr + pid_bh * BH_stride_v
    o_base = O_ptr + pid_bh * BH_stride_o
    do_base = dO_ptr + pid_bh * BH_stride_do
    l_base = L_ptr + pid_bh * BH_stride_l
    dq_base = dQ_ptr + pid_bh * BH_stride_dq

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offs < S
    d_offs = tl.arange(0, D_MODEL)
    d_mask = d_offs < D_MODEL

    q_ptrs = q_base + m_offs[:, None] * S_stride_q + d_offs[None, :] * D_stride_q
    q = tl.load(q_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)

    do_ptrs = do_base + m_offs[:, None] * S_stride_do + d_offs[None, :] * D_stride_do
    do_tile = tl.load(do_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)

    l_ptrs = l_base + m_offs * S_stride_l
    l_val = tl.load(l_ptrs, mask=m_mask, other=0.0).to(tl.float32)[:, None]

    scale = 1.0 / tl.sqrt(float(D_MODEL))
    num_n_blocks = tl.cdiv(S, BLOCK_N)
    acc = tl.zeros((BLOCK_M, D_MODEL), dtype=tl.float32)

    # Load O tile once per M block
    o_ptrs = o_base + m_offs[:, None] * S_stride_o + d_offs[None, :] * D_stride_o
    o_tile = tl.load(o_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)
    d_row = tl.sum((do_tile * o_tile).to(tl.float32), axis=1)[:, None]

    for nb in range(num_n_blocks):
        n_offs = nb * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = n_offs < S

        k_ptrs = k_base + n_offs[:, None] * S_stride_k + d_offs[None, :] * D_stride_k
        k = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)

        v_ptrs = v_base + n_offs[:, None] * S_stride_v + d_offs[None, :] * D_stride_v
        v = tl.load(v_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)

        qk = tl.dot(q.to(tl.float32), k.T.to(tl.float32)) * scale
        probs = tl.exp(qk - l_val)

        dp = tl.dot(do_tile.to(tl.float32), v.T.to(tl.float32))
        da = probs * (dp - d_row) * scale

        acc = tl.dot(da, k.to(tl.float32), acc)

    dq_out = acc.to(Q_ptr.dtype.element_ty)
    dq_ptrs = dq_base + m_offs[:, None] * S_stride_dq + d_offs[None, :] * D_stride_dq
    tl.store(dq_ptrs, dq_out, mask=m_mask[:, None] & d_mask[None, :])


@triton.jit
def _dkv_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dK_ptr, dV_ptr,
    S, D_MODEL,
    BH_stride_q, S_stride_q, D_stride_q,
    BH_stride_k, S_stride_k, D_stride_k,
    BH_stride_v, S_stride_v, D_stride_v,
    BH_stride_o, S_stride_o, D_stride_o,
    BH_stride_do, S_stride_do, D_stride_do,
    BH_stride_l, S_stride_l,
    BH_stride_dk, S_stride_dk, D_stride_dk,
    BH_stride_dv, S_stride_dv, D_stride_dv,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Compute dK and dV for multi-head attention backward."""
    pid_bh = tl.program_id(0)
    pid_n = tl.program_id(1)

    q_base = Q_ptr + pid_bh * BH_stride_q
    k_base = K_ptr + pid_bh * BH_stride_k
    v_base = V_ptr + pid_bh * BH_stride_v
    o_base = O_ptr + pid_bh * BH_stride_o
    do_base = dO_ptr + pid_bh * BH_stride_do
    l_base = L_ptr + pid_bh * BH_stride_l
    dk_base = dK_ptr + pid_bh * BH_stride_dk
    dv_base = dV_ptr + pid_bh * BH_stride_dv

    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offs < S
    d_offs = tl.arange(0, D_MODEL)
    d_mask = d_offs < D_MODEL

    k_ptrs = k_base + n_offs[:, None] * S_stride_k + d_offs[None, :] * D_stride_k
    k = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)

    v_ptrs = v_base + n_offs[:, None] * S_stride_v + d_offs[None, :] * D_stride_v
    v = tl.load(v_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)

    scale = 1.0 / tl.sqrt(float(D_MODEL))
    num_m_blocks = tl.cdiv(S, BLOCK_M)
    acc_dk = tl.zeros((BLOCK_N, D_MODEL), dtype=tl.float32)
    acc_dv = tl.zeros((BLOCK_N, D_MODEL), dtype=tl.float32)

    for mb in range(num_m_blocks):
        m_offs = mb * BLOCK_M + tl.arange(0, BLOCK_M)
        m_mask = m_offs < S

        q_ptrs = q_base + m_offs[:, None] * S_stride_q + d_offs[None, :] * D_stride_q
        q = tl.load(q_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)

        l_ptrs = l_base + m_offs * S_stride_l
        l_row = tl.load(l_ptrs, mask=m_mask, other=0.0).to(tl.float32)[:, None]

        do_ptrs = do_base + m_offs[:, None] * S_stride_do + d_offs[None, :] * D_stride_do
        do_tile = tl.load(do_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)

        o_ptrs = o_base + m_offs[:, None] * S_stride_o + d_offs[None, :] * D_stride_o
        o_tile = tl.load(o_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)
        d_row = tl.sum((do_tile * o_tile).to(tl.float32), axis=1)[:, None]

        qkt = tl.dot(q.to(tl.float32), k.T.to(tl.float32)) * scale
        probs = tl.exp(qkt - l_row)

        dp = tl.dot(do_tile.to(tl.float32), v.T.to(tl.float32))
        da = probs * (dp - d_row) * scale

        acc_dk = tl.dot(da.T, q.to(tl.float32), acc_dk)
        acc_dv = tl.dot(probs.T, do_tile.to(tl.float32), acc_dv)

    dk_out = acc_dk.to(K_ptr.dtype.element_ty)
    dk_ptrs = dk_base + n_offs[:, None] * S_stride_dk + d_offs[None, :] * D_stride_dk
    tl.store(dk_ptrs, dk_out, mask=n_mask[:, None] & d_mask[None, :])

    dv_out = acc_dv.to(V_ptr.dtype.element_ty)
    dv_ptrs = dv_base + n_offs[:, None] * S_stride_dv + d_offs[None, :] * D_stride_dv
    tl.store(dv_ptrs, dv_out, mask=n_mask[:, None] & d_mask[None, :])


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
    q_r = Q.reshape(BH, S, D)
    k_r = K.reshape(BH, S, D)
    v_r = V.reshape(BH, S, D)
    o_r = O.reshape(BH, S, D)
    do_r = dO.reshape(BH, S, D)
    l_r = L.reshape(BH, S)
    dq_r = dQ.reshape(BH, S, D)
    dk_r = dK.reshape(BH, S, D)
    dv_r = dV.reshape(BH, S, D)

    bh_q, s_q, d_q = q_r.stride()
    bh_k, s_k, d_k = k_r.stride()
    bh_v, s_v, d_v = v_r.stride()
    bh_o, s_o, d_o = o_r.stride()
    bh_do, s_do, d_do = do_r.stride()
    bh_l, s_l = l_r.stride()
    bh_dq, s_dq, d_dq = dq_r.stride()
    bh_dk, s_dk, d_dk = dk_r.stride()
    bh_dv, s_dv, d_dv = dv_r.stride()

    BLOCK_M = 64
    BLOCK_N = 64
    NUM_WARPS = 8
    NUM_STAGES = 3

    num_m_blocks = triton.cdiv(S, BLOCK_M)
    num_n_blocks = triton.cdiv(S, BLOCK_N)

    grid_dq = (BH, num_m_blocks)
    _dq_kernel[grid_dq](
        q_r, k_r, v_r, o_r, do_r, l_r, dq_r,
        S, D,
        bh_q, s_q, d_q,
        bh_k, s_k, d_k,
        bh_v, s_v, d_v,
        bh_o, s_o, d_o,
        bh_do, s_do, d_do,
        bh_l, s_l,
        bh_dq, s_dq, d_dq,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )

    grid_dkv = (BH, num_n_blocks)
    _dkv_kernel[grid_dkv](
        q_r, k_r, v_r, o_r, do_r, l_r, dk_r, dv_r,
        S, D,
        bh_q, s_q, d_q,
        bh_k, s_k, d_k,
        bh_v, s_v, d_v,
        bh_o, s_o, d_o,
        bh_do, s_do, d_do,
        bh_l, s_l,
        bh_dk, s_dk, d_dk,
        bh_dv, s_dv, d_dv,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )