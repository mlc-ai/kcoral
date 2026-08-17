import torch
import triton
import triton.language as tl


@triton.jit
def _dkv_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, do_ptr, l_ptr,
    dk_ptr, dv_ptr,
    seq_len, head_dim,
    stride_q_s, stride_q_d,
    stride_k_s, stride_k_d,
    stride_v_s, stride_v_d,
    stride_o_s, stride_o_d,
    stride_do_s, stride_do_d,
    stride_l_s,
    stride_dk_s, stride_dk_d,
    stride_dv_s, stride_dv_d,
    total_assignments,
    ns_q_blocks,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    work_id = tl.program_id(0)
    inv_scale = 1.0 / tl.sqrt(tl.cast(head_dim, tl.float32))

    offs_k_inner = tl.arange(0, BLOCK_K)
    offs_d_inner = tl.arange(0, BLOCK_D)

    while work_id < total_assignments:
        idx = work_id
        work_id += tl.num_programs(0)

        # Decode assignment: iterate over (bh_idx, kv_blk)
        kv_blk = idx // BLOCK_K
        bh_idx = idx % BLOCK_K
        kv_offset = kv_blk * BLOCK_K

        kh_base = q_ptr + bh_idx
        kk_base = k_ptr + bh_idx
        kv_base = v_ptr + bh_idx
        ko_base = o_ptr + bh_idx
        kdo_base = do_ptr + bh_idx
        kl_base = l_ptr + bh_idx
        odk_base = dk_ptr + bh_idx
        odv_base = dv_ptr + bh_idx

        offs_k_abs = kv_offset + offs_k_inner
        mask_k = offs_k_abs < seq_len

        # Load K and V for this (bh_idx, kv_blk)
        k_ptrs = kk_base + offs_k_abs[:, None] * stride_k_s + offs_d_inner[None, :] * stride_k_d
        k_tile = tl.load(k_ptrs, mask=mask_k[:, None], other=0.0)

        v_ptrs = kv_base + offs_k_abs[:, None] * stride_v_s + offs_d_inner[None, :] * stride_v_d
        v_tile = tl.load(v_ptrs, mask=mask_k[:, None], other=0.0)

        acc_dk = tl.zeros((BLOCK_K, BLOCK_D), dtype=tl.float32)
        acc_dv = tl.zeros((BLOCK_K, BLOCK_D), dtype=tl.float32)

        # Loop over Q blocks (grid-stride over ns_q_blocks worth of tasks)
        qb_start = idx // BLOCK_K  # irrelevant, always starts at 0 since we decode above
        # Actually we iterate ALL Q blocks for this (bh_idx, kv_blk)
        for qb in range(ns_q_blocks):
            q_base_offset = qb * BLOCK_Q
            offs_q_abs = q_base_offset + tl.arange(0, BLOCK_Q)
            mask_q = offs_q_abs < seq_len

            q_ptrs = kh_base + offs_q_abs[:, None] * stride_q_s + offs_d_inner[None, :] * stride_q_d
            q_t = tl.load(q_ptrs, mask=mask_q[:, None], other=0.0)

            do_ptrs = kdo_base + offs_q_abs[:, None] * stride_do_s + offs_d_inner[None, :] * stride_do_d
            do_t = tl.load(do_ptrs, mask=mask_q[:, None], other=0.0)

            l_ptrs = kl_base + offs_q_abs * stride_l_s
            l_t = tl.load(l_ptrs, mask=mask_q, other=float("-inf"))

            causal_mask = (offs_q_abs[:, None] >= offs_k_abs[None, :])

            q_f32 = q_t.to(tl.float32)
            do_f32 = do_t.to(tl.float32)

            # S = Q @ K.T / sqrt(d)
            s = tl.dot(q_f32, k_tile.T)
            s *= inv_scale

            # P = exp(min(s - l, 60)) * causal
            p = tl.exp(tl.minimum(tl.maximum(-60.0, s - l_t[:, None]), 60.0))
            p = p * causal_mask

            # dp = dO @ V.T
            dp = tl.dot(do_f32, v_tile.T)

            # ds = P * dp * inv_scale (note: D term cancels out in dK/dV via dot identity)
            ds = p * dp * inv_scale

            # dK += ds^T @ Q
            acc_dk = tl.dot(ds.T, q_f32, acc_dk)

            # dV += P^T @ dO
            acc_dv = tl.dot(p.T, do_f32, acc_dv)

        # Write back dK and dV
        dk_ptrs = odk_base + offs_k_abs[:, None] * stride_dk_s + offs_d_inner[None, :] * stride_dk_d
        tl.store(dk_ptrs, acc_dk, mask=mask_k[:, None])

        dv_ptrs = odv_base + offs_k_abs[:, None] * stride_dv_s + offs_d_inner[None, :] * stride_dv_d
        tl.store(dv_ptrs, acc_dv, mask=mask_k[:, None])


@triton.jit
def _dq_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, do_ptr, l_ptr,
    dq_ptr,
    seq_len, head_dim,
    stride_q_s, stride_q_d,
    stride_k_s, stride_k_d,
    stride_v_s, stride_v_d,
    stride_o_s, stride_o_d,
    stride_do_s, stride_do_d,
    stride_l_s,
    stride_dq_s, stride_dq_d,
    total_assignments,
    ns_k_blocks,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    work_id = tl.program_id(0)
    inv_scale = 1.0 / tl.sqrt(tl.cast(head_dim, tl.float32))

    offs_q_inner = tl.arange(0, BLOCK_Q)
    offs_d_inner = tl.arange(0, BLOCK_D)

    while work_id < total_assignments:
        idx = work_id
        work_id += tl.num_programs(0)

        q_blk = idx // BLOCK_Q
        bh_idx = idx % BLOCK_Q
        q_offset = q_blk * BLOCK_Q

        hq_base = q_ptr + bh_idx
        hk_base = k_ptr + bh_idx
        hv_base = v_ptr + bh_idx
        ho_base = o_ptr + bh_idx
        hdo_base = do_ptr + bh_idx
        hl_base = l_ptr + bh_idx
        hdq_base = dq_ptr + bh_idx

        offs_q_abs = q_offset + offs_q_inner
        mask_q = offs_q_abs < seq_len

        q_ptrs = hq_base + offs_q_abs[:, None] * stride_q_s + offs_d_inner[None, :] * stride_q_d
        q_t = tl.load(q_ptrs, mask=mask_q[:, None], other=0.0)

        do_ptrs = hdo_base + offs_q_abs[:, None] * stride_do_s + offs_d_inner[None, :] * stride_do_d
        do_t = tl.load(do_ptrs, mask=mask_q[:, None], other=0.0)

        l_ptrs = hl_base + offs_q_abs * stride_l_s
        l_t = tl.load(l_ptrs, mask=mask_q, other=float("-inf"))

        # D = rowsum(dO * O) => [BLOCK_Q, 1]
        o_ptrs = ho_base + offs_q_abs[:, None] * stride_o_s + offs_d_inner[None, :] * stride_o_d
        o_t = tl.load(o_ptrs, mask=mask_q[:, None], other=0.0)
        do_o = do_t.to(tl.float32) * o_t.to(tl.float32)
        d_row = tl.sum(do_o, axis=1, keep_dims=True)

        q_f32 = q_t.to(tl.float32)
        do_f32 = do_t.to(tl.float32)

        acc_dq = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)

        for kb in range(ns_k_blocks):
            kv_offset = kb * BLOCK_K
            offs_k_abs = kv_offset + tl.arange(0, BLOCK_K)
            mask_k = offs_k_abs < seq_len

            causal_mask = (offs_q_abs[:, None] >= offs_k_abs[None, :])

            k_ptrs = hk_base + offs_k_abs[:, None] * stride_k_s + offs_d_inner[None, :] * stride_k_d
            k_t = tl.load(k_ptrs, mask=mask_k[:, None], other=0.0)

            v_ptrs = hv_base + offs_k_abs[:, None] * stride_v_s + offs_d_inner[None, :] * stride_v_d
            v_t = tl.load(v_ptrs, mask=mask_k[:, None], other=0.0)

            s = tl.dot(q_f32, k_t.T)
            s *= inv_scale

            p = tl.exp(tl.minimum(tl.maximum(-60.0, s - l_t[:, None]), 60.0))
            p = p * causal_mask

            dp = tl.dot(do_f32, v_t.T)

            ds = p * (dp - d_row) * inv_scale

            acc_dq = tl.dot(ds, k_t, acc_dq)

        dq_ptrs = hdq_base + offs_q_abs[:, None] * stride_dq_s + offs_d_inner[None, :] * stride_dq_d
        tl.store(dq_ptrs, acc_dq, mask=mask_q[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute causal multi-head attention backward: dQ, dK, dV."""
    torch.cuda.set_device(Q.device)

    B, H, S, D = Q.shape
    BH = B * H
    seq_len = S
    head_dim = D

    stride_q_s = Q.stride(2)
    stride_q_d = Q.stride(3)
    stride_k_s = K.stride(2)
    stride_k_d = K.stride(3)
    stride_v_s = V.stride(2)
    stride_v_d = V.stride(3)
    stride_o_s = O.stride(2)
    stride_o_d = O.stride(3)
    stride_do_s = dO.stride(2)
    stride_do_d = dO.stride(3)
    stride_l_s = L.stride(2)
    stride_dk_s = dK.stride(2)
    stride_dk_d = dK.stride(3)
    stride_dv_s = dV.stride(2)
    stride_dv_d = dV.stride(3)
    stride_dq_s = dQ.stride(2)
    stride_dq_d = dQ.stride(3)

    try:
        num_sms = torch.cuda.get_device_properties(Q.device).multi_processor_count
    except Exception:
        num_sms = 132

    BLOCK_Q = 64
    BLOCK_K = 64
    BLOCK_D = 128

    ns_q_blocks = triton.cdiv(seq_len, BLOCK_Q)
    ns_k_blocks = triton.cdiv(seq_len, BLOCK_K)

    # dKV: one program per (bh_idx, kv_blk), persistent scheduling
    total_dkv_assignments = BH * ns_k_blocks
    grid_dkv = max(1, min(num_sms, total_dkv_assignments))

    _dkv_kernel[(grid_dkv,)](
        Q, K, V, O, dO, L,
        dK, dV,
        seq_len, head_dim,
        stride_q_s, stride_q_d,
        stride_k_s, stride_k_d,
        stride_v_s, stride_v_d,
        stride_o_s, stride_o_d,
        stride_do_s, stride_do_d,
        stride_l_s,
        stride_dk_s, stride_dk_d,
        stride_dv_s, stride_dv_d,
        total_dkv_assignments,
        ns_q_blocks,
        BLOCK_Q=BLOCK_Q,
        BLOCK_K=BLOCK_K,
        BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=3,
    )

    # dQ: one program per (bh_idx, q_blk), persistent scheduling
    total_dq_assignments = BH * ns_q_blocks
    grid_dq = max(1, min(num_sms, total_dq_assignments))

    _dq_kernel[(grid_dq,)](
        Q, K, V, O, dO, L,
        dQ,
        seq_len, head_dim,
        stride_q_s, stride_q_d,
        stride_k_s, stride_k_d,
        stride_v_s, stride_v_d,
        stride_o_s, stride_o_d,
        stride_do_s, stride_do_d,
        stride_l_s,
        stride_dq_s, stride_dq_d,
        total_dq_assignments,
        ns_k_blocks,
        BLOCK_Q=BLOCK_Q,
        BLOCK_K=BLOCK_K,
        BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=3,
    )