import torch
import triton
import triton.language as tl


@triton.jit
def _precompute_d_kernel(
    o_ptr,
    do_ptr,
    d_ptr,
    n_bh,
    seq_len,
    dim_d,
    stride_o_s,
    stride_o_d,
    stride_do_s,
    stride_do_d,
    stride_d_s,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    bh_idx = pid
    if bh_idx >= n_bh:
        return

    row_offs = tl.arange(0, BLOCK_S)
    col_offs = tl.arange(0, BLOCK_D)

    for blk_s in range(0, tl.cdiv(seq_len, BLOCK_S)):
        s_base = blk_s * BLOCK_S
        s_offs = s_base + row_offs
        mask_s = s_offs < seq_len

        o_ptrs = o_ptr + s_offs[:, None] * stride_o_s + col_offs[None, :] * stride_o_d
        do_ptrs = do_ptr + s_offs[:, None] * stride_do_s + col_offs[None, :] * stride_do_d

        o_val = tl.load(o_ptrs, mask=mask_s[:, None], other=0.0)
        do_val = tl.load(do_ptrs, mask=mask_s[:, None], other=0.0)

        d_row = tl.sum(o_val * do_val, axis=1)
        d_ptrs = d_ptr + s_offs * stride_d_s
        tl.store(d_ptrs, d_row, mask=mask_s)


@triton.jit
def _dkv_kernel(
    q, k, v, o, do, l,
    dk, dv,
    seq_len, head_dim,
    stride_q_s, stride_q_d,
    stride_k_s, stride_k_d,
    stride_v_s, stride_v_d,
    stride_l_s,
    stride_dk_s, stride_dk_d,
    stride_dv_s, stride_dv_d,
    total_assignments,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    work_id = tl.program_id(0)
    inv_scale = 1.0 / tl.sqrt(tl.cast(head_dim, tl.float32))

    offs_q_inner = tl.arange(0, BLOCK_Q)
    offs_k_inner = tl.arange(0, BLOCK_K)
    offs_d_inner = tl.arange(0, BLOCK_D)

    qk_mask = offs_q_inner[:, None] < seq_len

    while work_id < total_assignments:
        idx = work_id
        work_id += tl.num_programs(0)

        kv_blk = idx // BLOCK_K
        bh_idx = idx % BLOCK_K
        kv_offset = kv_blk * BLOCK_K

        kh_base = q + bh_idx
        kk_base = k + bh_idx
        kv_base = v + bh_idx
        kl_base = l + bh_idx
        kd_base = do + bh_idx
        odk_base = dk + bh_idx
        odv_base = dv + bh_idx

        offs_k_abs = kv_offset + offs_k_inner
        mask_k = offs_k_abs < seq_len

        k_ptrs = kk_base + offs_k_abs[:, None] * stride_k_s + offs_d_inner[None, :] * stride_k_d
        k = tl.load(k_ptrs, mask=mask_k[:, None], other=0.0)

        v_ptrs = kv_base + offs_k_abs[:, None] * stride_v_s + offs_d_inner[None, :] * stride_v_d
        v = tl.load(v_ptrs, mask=mask_k[:, None], other=0.0)

        acc_dk = tl.zeros((BLOCK_K, BLOCK_D), dtype=tl.float32)
        acc_dv = tl.zeros((BLOCK_K, BLOCK_D), dtype=tl.float32)

        num_q_blks = tl.cdiv(seq_len, BLOCK_Q)
        for qb in range(num_q_blks):
            q_base = qb * BLOCK_Q
            offs_q_abs = q_base + offs_q_inner
            mask_q = offs_q_abs < seq_len

            q_ptrs = kh_base + offs_q_abs[:, None] * stride_q_s + offs_d_inner[None, :] * stride_q_d
            q_t = tl.load(q_ptrs, mask=mask_q[:, None], other=0.0)

            do_ptrs = kd_base + offs_q_abs[:, None] * stride_q_s + offs_d_inner[None, :] * stride_q_d
            do_t = tl.load(do_ptrs, mask=mask_q[:, None], other=0.0)

            l_ptrs = kl_base + offs_q_abs * stride_l_s
            l_t = tl.load(l_ptrs, mask=mask_q, other=float("-inf"))

            causal_mask = (offs_q_abs[:, None] >= offs_k_abs[None, :])

            q_f32 = q_t.to(tl.float32)
            do_f32 = do_t.to(tl.float32)

            s = tl.dot(q_f32, k.T)
            s *= inv_scale

            p = tl.exp(tl.minimum(60.0, tl.maximum(-60.0, s - l_t[:, None])))
            p *= causal_mask

            dp = tl.dot(do_f32, v.T)

            ds = p * (dp - l_t[:, None] * 0.0) * inv_scale
            ds_zero = tl.zeros((BLOCK_Q, BLOCK_K), dtype=tl.float32)

            acc_dk = tl.dot(ds.T, q_f32, acc_dk)
            acc_dv = tl.dot(p.T, do_f32, acc_dv)

        dk_ptrs = odk_base + offs_k_abs[:, None] * stride_dk_s + offs_d_inner[None, :] * stride_dk_d
        tl.store(dk_ptrs, acc_dk, mask=mask_k[:, None])

        dv_ptrs = odv_base + offs_k_abs[:, None] * stride_dv_s + offs_d_inner[None, :] * stride_dv_d
        tl.store(dv_ptrs, acc_dv, mask=mask_k[:, None])


@triton.jit
def _dq_kernel(
    q, k, v, o, do, l,
    dq,
    seq_len, head_dim,
    stride_q_s, stride_q_d,
    stride_k_s, stride_k_d,
    stride_v_s, stride_v_d,
    stride_l_s,
    stride_dq_s, stride_dq_d,
    total_assignments,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    work_id = tl.program_id(0)
    inv_scale = 1.0 / tl.sqrt(tl.cast(head_dim, tl.float32))

    offs_q_inner = tl.arange(0, BLOCK_Q)
    offs_k_inner = tl.arange(0, BLOCK_K)
    offs_d_inner = tl.arange(0, BLOCK_D)

    while work_id < total_assignments:
        idx = work_id
        work_id += tl.num_programs(0)

        q_blk = idx
        bh_idx = idx % BLOCK_Q
        q_offset = q_blk * BLOCK_Q

        hq_base = q + bh_idx
        hk_base = k + bh_idx
        hv_base = v + bh_idx
        hl_base = l + bh_idx
        hdo_base = do + bh_idx
        hdq_base = dq + bh_idx

        offs_q_abs = q_offset + offs_q_inner
        mask_q = offs_q_abs < seq_len

        q_ptrs = hq_base + offs_q_abs[:, None] * stride_q_s + offs_d_inner[None, :] * stride_q_d
        q = tl.load(q_ptrs, mask=mask_q[:, None], other=0.0)

        do_ptrs = hdo_base + offs_q_abs[:, None] * stride_q_s + offs_d_inner[None, :] * stride_q_d
        do_t = tl.load(do_ptrs, mask=mask_q[:, None], other=0.0)

        l_ptrs = hl_base + offs_q_abs * stride_l_s
        l_t = tl.load(l_ptrs, mask=mask_q, other=float("-inf"))

        q_f32 = q.to(tl.float32)
        do_f32 = do_t.to(tl.float32)

        acc_dq = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)

        num_k_blks = tl.cdiv(seq_len, BLOCK_K)
        for kb in range(num_k_blks):
            kv_offset = kb * BLOCK_K
            offs_k_abs = kv_offset + offs_k_inner
            mask_k = offs_k_abs < seq_len

            causal_mask = (offs_q_abs[:, None] >= offs_k_abs[None, :])

            k_ptrs = hk_base + offs_k_abs[:, None] * stride_k_s + offs_d_inner[None, :] * stride_k_d
            k_t = tl.load(k_ptrs, mask=mask_k[:, None], other=0.0)

            v_ptrs = hv_base + offs_k_abs[:, None] * stride_v_s + offs_d_inner[None, :] * stride_v_d
            v_t = tl.load(v_ptrs, mask=mask_k[:, None], other=0.0)

            s = tl.dot(q_f32, k_t.T)
            s *= inv_scale

            p = tl.exp(tl.minimum(60.0, tl.maximum(-60.0, s - l_t[:, None])))
            p *= causal_mask

            dp = tl.dot(do_f32, v_t.T)

            ds = p * (dp - l_t[:, None] * 0.0) * inv_scale

            acc_dq = tl.dot(ds, k_t, acc_dq)

        dq_ptrs = hdq_base + offs_q_abs[:, None] * stride_dq_s + offs_d_inner[None, :] * stride_dq_d
        tl.store(dq_ptrs, acc_dq, mask=mask_q[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute causal multi-head attention backward: dQ, dK, dV."""
    torch.cuda.set_device(Q.device)

    B, H, S, D = Q.shape
    assert K.shape == Q.shape
    assert V.shape == Q.shape
    assert O.shape == Q.shape
    assert dO.shape == Q.shape
    assert L.shape == (B, H, S)
    assert dQ.shape == Q.shape
    assert dK.shape == Q.shape
    assert dV.shape == Q.shape

    BH = B * H
    n_heads = H
    seq_len = S
    head_dim = D

    stride_q_s = Q.stride(2)
    stride_q_d = Q.stride(3)
    stride_k_s = K.stride(2)
    stride_k_d = K.stride(3)
    stride_v_s = V.stride(2)
    stride_v_d = V.stride(3)
    stride_l_s = L.stride(2)
    stride_do_s = dO.stride(2)
    stride_do_d = dO.stride(3)
    stride_dk_s = dK.stride(2)
    stride_dk_d = dK.stride(3)
    stride_dv_s = dV.stride(2)
    stride_dv_d = dV.stride(3)
    stride_dq_s = dQ.stride(2)
    stride_dq_d = dQ.stride(3)

    BLOCK_Q = 64
    BLOCK_K = 64
    BLOCK_D = 128
    BLOCK_S_PRE = 256

    # Precompute D = rowsum(dO * O) temporarily - borrow dQ storage
    # We'll use a temporary tensor allocated only for workspace
    D_temp = torch.empty((B, H, S), dtype=torch.float32, device=Q.device)
    d_pre_strides = D_temp.stride()
    grid_pre = (triton.cdiv(seq_len, BLOCK_S_PRE), BH)
    _precompute_d_kernel[(grid_pre[0], grid_pre[1])](
        O, dO, D_temp,
        seq_len, head_dim,
        stride_q_s, stride_q_d,
        stride_do_s, stride_do_d,
        d_pre_strides[2],
        BLOCK_S=BLOCK_S_PRE,
        BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=2,
    )

    # Get number of SMs
    try:
        num_sms = torch.cuda.get_device_properties(Q.device).multi_processor_count
    except Exception:
        num_sms = 132

    # Total assignments for dKV: BH * ceil(S / BLOCK_K)
    ns_k_blocks = triton.cdiv(seq_len, BLOCK_K)
    total_dkv_assignments = BH * ns_k_blocks
    grid_dkv = min(num_sms, total_dkv_assignments)
    if grid_dkv < 1:
        grid_dkv = 1

    _dkv_kernel[(grid_dkv,)](
        Q, K, V, O, dO, L,
        dK, dV,
        seq_len, head_dim,
        stride_q_s, stride_q_d,
        stride_k_s, stride_k_d,
        stride_v_s, stride_v_d,
        stride_l_s,
        stride_dk_s, stride_dk_d,
        stride_dv_s, stride_dv_d,
        total_dkv_assignments,
        BLOCK_Q=BLOCK_Q,
        BLOCK_K=BLOCK_K,
        BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=3,
    )

    # dQ: total assignments = BH * ceil(S / BLOCK_Q)
    ns_q_blocks = triton.cdiv(seq_len, BLOCK_Q)
    total_dq_assignments = BH * ns_q_blocks
    grid_dq = min(num_sms, total_dq_assignments)
    if grid_dq < 1:
        grid_dq = 1

    _dq_kernel[(grid_dq,)](
        Q, K, V, O, dO, L,
        dQ,
        seq_len, head_dim,
        stride_q_s, stride_q_d,
        stride_k_s, stride_k_d,
        stride_v_s, stride_v_d,
        stride_l_s,
        stride_dq_s, stride_dq_d,
        total_dq_assignments,
        BLOCK_Q=BLOCK_Q,
        BLOCK_K=BLOCK_K,
        BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=3,
    )