import torch
import triton
import triton.language as tl


@triton.jit
def _dkv_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, do_ptr, l_ptr,
    dk_ptr, dv_ptr,
    seq_len, head_dim, n_bh, n_heads,
    str_q_b, str_q_h, str_q_s, str_q_d,
    str_k_b, str_k_h, str_k_s, str_k_d,
    str_v_b, str_v_h, str_v_s, str_v_d,
    str_o_b, str_o_h, str_o_s, str_o_d,
    str_do_b, str_do_h, str_do_s, str_do_d,
    str_l_b, str_l_h, str_l_s,
    str_dk_b, str_dk_h, str_dk_s, str_dk_d,
    str_dv_b, str_dv_h, str_dv_s, str_dv_d,
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

        # Decode assignment: each idx corresponds to one (bh_idx, kv_blk)
        kv_block_in_seq = idx // n_bh
        bh_idx = idx % n_bh
        b_idx = bh_idx // n_heads
        h_idx = bh_idx % n_heads
        kv_offset = kv_block_in_seq * BLOCK_K

        # Base offsets for this (b, h)
        q_base = q_ptr + b_idx * str_q_b + h_idx * str_q_h
        k_base = k_ptr + b_idx * str_k_b + h_idx * str_k_h
        v_base = v_ptr + b_idx * str_v_b + h_idx * str_v_h
        o_base = o_ptr + b_idx * str_o_b + h_idx * str_o_h
        do_base = do_ptr + b_idx * str_do_b + h_idx * str_do_h
        l_base = l_ptr + b_idx * str_l_b + h_idx * str_l_h
        dk_base = dk_ptr + b_idx * str_dk_b + h_idx * str_dk_h
        dv_base = dv_ptr + b_idx * str_dv_b + h_idx * str_dv_h

        offs_k_abs = kv_offset + offs_k_inner
        mask_k = offs_k_abs < seq_len

        # Load K and V for this (bh_idx, kv_blk)
        k_ptrs = k_base + offs_k_abs[:, None] * str_k_s + offs_d_inner[None, :] * str_k_d
        k_tile = tl.load(k_ptrs, mask=mask_k[:, None], other=0.0)
        k_f32 = k_tile.to(tl.float32)

        v_ptrs = v_base + offs_k_abs[:, None] * str_v_s + offs_d_inner[None, :] * str_v_d
        v_tile = tl.load(v_ptrs, mask=mask_k[:, None], other=0.0)
        v_f32 = v_tile.to(tl.float32)

        acc_dk = tl.zeros((BLOCK_K, BLOCK_D), dtype=tl.float32)
        acc_dv = tl.zeros((BLOCK_K, BLOCK_D), dtype=tl.float32)

        # Loop over all Q blocks for this KV block
        for qb in range(ns_q_blocks):
            q_base_offset = qb * BLOCK_Q
            offs_q_abs = q_base_offset + tl.arange(0, BLOCK_Q)
            mask_q = offs_q_abs < seq_len

            q_ptrs = q_base + offs_q_abs[:, None] * str_q_s + offs_d_inner[None, :] * str_q_d
            q_t = tl.load(q_ptrs, mask=mask_q[:, None], other=0.0)
            q_f32 = q_t.to(tl.float32)

            do_ptrs = do_base + offs_q_abs[:, None] * str_do_s + offs_d_inner[None, :] * str_do_d
            do_t = tl.load(do_ptrs, mask=mask_q[:, None], other=0.0)
            do_f32 = do_t.to(tl.float32)

            l_ptrs = l_base + offs_q_abs * str_l_s
            l_t = tl.load(l_ptrs, mask=mask_q, other=float("-inf"))

            causal_mask = (offs_q_abs[:, None] >= offs_k_abs[None, :])

            # S = Q @ K^T / sqrt(d)
            s = tl.dot(q_f32, k_f32.T)
            s *= inv_scale

            # P = softmax(S) = exp(S - L), clipped
            p = tl.exp(tl.minimum(tl.maximum(-60.0, s - l_t[:, None]), 60.0))
            p = p * causal_mask

            # dp = dO @ V^T
            dp = tl.dot(do_f32, v_f32.T)

            # D_i = rowsum(dO_i * O_i) for dK computation
            o_ptrs = o_base + offs_q_abs[:, None] * str_o_s + offs_d_inner[None, :] * str_o_d
            o_t = tl.load(o_ptrs, mask=mask_q[:, None], other=0.0)
            o_f32 = o_t.to(tl.float32)
            d_row = tl.sum(do_f32 * o_f32, axis=1, keep_dims=True)

            # ds = P * (dp - D) / sqrt(d)
            ds = p * (dp - d_row) * inv_scale

            # dK += ds^T @ Q
            acc_dk += tl.dot(ds.T, q_f32)

            # dV += P^T @ dO
            acc_dv += tl.dot(p.T, do_f32)

        # Write back dK and dV
        dk_ptrs = dk_base + offs_k_abs[:, None] * str_dk_s + offs_d_inner[None, :] * str_dk_d
        tl.store(dk_ptrs, acc_dk, mask=mask_k[:, None])

        dv_ptrs = dv_base + offs_k_abs[:, None] * str_dv_s + offs_d_inner[None, :] * str_dv_d
        tl.store(dv_ptrs, acc_dv, mask=mask_k[:, None])


@triton.jit
def _dq_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, do_ptr, l_ptr,
    dq_ptr,
    seq_len, head_dim, n_bh, n_heads,
    str_q_b, str_q_h, str_q_s, str_q_d,
    str_k_b, str_k_h, str_k_s, str_k_d,
    str_v_b, str_v_h, str_v_s, str_v_d,
    str_o_b, str_o_h, str_o_s, str_o_d,
    str_do_b, str_do_h, str_do_s, str_do_d,
    str_l_b, str_l_h, str_l_s,
    str_dq_b, str_dq_h, str_dq_s, str_dq_d,
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

        # Decode: each idx -> (bh_idx, q_blk)
        q_block_in_seq = idx // n_bh
        bh_idx = idx % n_bh
        b_idx = bh_idx // n_heads
        h_idx = bh_idx % n_heads
        q_offset = q_block_in_seq * BLOCK_Q

        q_base = q_ptr + b_idx * str_q_b + h_idx * str_q_h
        k_base = k_ptr + b_idx * str_k_b + h_idx * str_k_h
        v_base = v_ptr + b_idx * str_v_b + h_idx * str_v_h
        o_base = o_ptr + b_idx * str_o_b + h_idx * str_o_h
        do_base = do_ptr + b_idx * str_do_b + h_idx * str_do_h
        l_base = l_ptr + b_idx * str_l_b + h_idx * str_l_h
        dq_base = dq_ptr + b_idx * str_dq_b + h_idx * str_dq_h

        offs_q_abs = q_offset + offs_q_inner
        mask_q = offs_q_abs < seq_len

        q_ptrs = q_base + offs_q_abs[:, None] * str_q_s + offs_d_inner[None, :] * str_q_d
        q_t = tl.load(q_ptrs, mask=mask_q[:, None], other=0.0)
        q_f32 = q_t.to(tl.float32)

        do_ptrs = do_base + offs_q_abs[:, None] * str_do_s + offs_d_inner[None, :] * str_do_d
        do_t = tl.load(do_ptrs, mask=mask_q[:, None], other=0.0)
        do_f32 = do_t.to(tl.float32)

        l_ptrs = l_base + offs_q_abs * str_l_s
        l_t = tl.load(l_ptrs, mask=mask_q, other=float("-inf"))

        # D = rowsum(dO * O) => [BLOCK_Q, 1]
        o_ptrs = o_base + offs_q_abs[:, None] * str_o_s + offs_d_inner[None, :] * str_o_d
        o_t = tl.load(o_ptrs, mask=mask_q[:, None], other=0.0)
        o_f32 = o_t.to(tl.float32)
        d_row = tl.sum(do_f32 * o_f32, axis=1, keep_dims=True)

        acc_dq = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)

        for kb in range(ns_k_blocks):
            kv_offset = kb * BLOCK_K
            offs_k_abs = kv_offset + tl.arange(0, BLOCK_K)
            mask_k = offs_k_abs < seq_len

            causal_mask = (offs_q_abs[:, None] >= offs_k_abs[None, :])

            k_ptrs = k_base + offs_k_abs[:, None] * str_k_s + offs_d_inner[None, :] * str_k_d
            k_t = tl.load(k_ptrs, mask=mask_k[:, None], other=0.0)
            k_f32 = k_t.to(tl.float32)

            v_ptrs = v_base + offs_k_abs[:, None] * str_v_s + offs_d_inner[None, :] * str_v_d
            v_t = tl.load(v_ptrs, mask=mask_k[:, None], other=0.0)
            v_f32 = v_t.to(tl.float32)

            s = tl.dot(q_f32, k_f32.T)
            s *= inv_scale

            p = tl.exp(tl.minimum(tl.maximum(-60.0, s - l_t[:, None]), 60.0))
            p = p * causal_mask

            dp = tl.dot(do_f32, v_f32.T)

            ds = p * (dp - d_row) * inv_scale

            dq_inc = tl.dot(ds, k_f32)
            acc_dq += dq_inc

        dq_ptrs = dq_base + offs_q_abs[:, None] * str_dq_s + offs_d_inner[None, :] * str_dq_d
        tl.store(dq_ptrs, acc_dq, mask=mask_q[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute causal multi-head attention backward: dQ, dK, dV."""
    torch.cuda.set_device(Q.device)

    B, H, S, D = Q.shape
    BH = B * H
    seq_len = S
    head_dim = D

    try:
        num_sms = torch.cuda.get_device_properties(Q.device).multi_processor_count
    except Exception:
        num_sms = 132

    BLOCK_Q = 64
    BLOCK_K = 64
    BLOCK_D = 128

    ns_q_blocks = triton.cdiv(seq_len, BLOCK_Q)
    ns_k_blocks = triton.cdiv(seq_len, BLOCK_K)

    # Strides
    str_q_b, str_q_h, str_q_s, str_q_d = Q.stride()
    str_k_b, str_k_h, str_k_s, str_k_d = K.stride()
    str_v_b, str_v_h, str_v_s, str_v_d = V.stride()
    str_o_b, str_o_h, str_o_s, str_o_d = O.stride()
    str_do_b, str_do_h, str_do_s, str_do_d = dO.stride()
    str_l_b, str_l_h, str_l_s = L.stride(), L.stride(), L.stride()
    str_dk_b, str_dk_h, str_dk_s, str_dk_d = dK.stride()
    str_dv_b, str_v_h2, str_dv_s, str_dv_d = dV.stride()
    str_dq_b, str_dq_h, str_dq_s, str_dq_d = dQ.stride()

    # Fix: extract L strides properly
    str_l_b, str_l_h, str_l_s = L.stride(0), L.stride(1), L.stride(2)

    # dKV kernel
    total_dkv_assignments = BH * ns_k_blocks
    grid_dkv = max(1, min(num_sms, total_dkv_assignments))

    _dkv_kernel[(grid_dkv,)](
        Q, K, V, O, dO, L,
        dK, dV,
        seq_len, head_dim, BH, H,
        str_q_b, str_q_h, str_q_s, str_q_d,
        str_k_b, str_k_h, str_k_s, str_k_d,
        str_v_b, str_v_h, str_v_s, str_v_d,
        str_o_b, str_o_h, str_o_s, str_o_d,
        str_do_b, str_do_h, str_do_s, str_do_d,
        str_l_b, str_l_h, str_l_s,
        str_dk_b, str_dk_h, str_dk_s, str_dk_d,
        str_dv_b, str_v_h2, str_dv_s, str_dv_d,
        total_dkv_assignments,
        ns_q_blocks,
        BLOCK_Q=BLOCK_Q,
        BLOCK_K=BLOCK_K,
        BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=3,
    )

    # dQ kernel
    total_dq_assignments = BH * ns_q_blocks
    grid_dq = max(1, min(num_sms, total_dq_assignments))

    _dq_kernel[(grid_dq,)](
        Q, K, V, O, dO, L,
        dQ,
        seq_len, head_dim, BH, H,
        str_q_b, str_q_h, str_q_s, str_q_d,
        str_k_b, str_k_h, str_k_s, str_k_d,
        str_v_b, str_v_h, str_v_s, str_v_d,
        str_o_b, str_o_h, str_o_s, str_o_d,
        str_do_b, str_do_h, str_do_s, str_do_d,
        str_l_b, str_l_h, str_l_s,
        str_dq_b, str_dq_h, str_dq_s, str_dq_d,
        total_dq_assignments,
        ns_k_blocks,
        BLOCK_Q=BLOCK_Q,
        BLOCK_K=BLOCK_K,
        BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=3,
    )