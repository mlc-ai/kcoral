import torch
import triton
import triton.language as tl


@triton.jit
def _dkv_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, do_ptr, l_ptr,
    dk_ptr, dv_ptr,
    seq_len, head_dim, n_heads,
    str_q_b, str_q_h, str_q_s, str_q_d,
    str_k_b, str_k_h, str_k_s, str_k_d,
    str_v_b, str_v_h, str_v_s, str_v_d,
    str_o_b, str_o_h, str_o_s, str_o_d,
    str_do_b, str_do_h, str_do_s, str_do_d,
    str_l_b, str_l_h, str_l_s,
    str_dk_b, str_dk_h, str_dk_s, str_dk_d,
    str_dv_b, str_dv_h, str_dv_s, str_dv_d,
    ns_q_blocks,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_k = tl.program_id(1)

    n_bh = tl.num_programs(0)
    
    b_idx = pid_bh // n_heads
    h_idx = pid_bh % n_heads
    
    kv_offset = pid_k * BLOCK_K

    inv_scale = 1.0 / tl.sqrt(tl.cast(head_dim, tl.float32))

    offs_k_inner = tl.arange(0, BLOCK_K)
    offs_d_inner = tl.arange(0, BLOCK_D)

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

    # Load K and V for this block
    k_ptrs = k_base + offs_k_abs[:, None] * str_k_s + offs_d_inner[None, :] * str_k_d
    k_tile = tl.load(k_ptrs, mask=mask_k[:, None], other=0.0).to(tl.float32)

    v_ptrs = v_base + offs_k_abs[:, None] * str_v_s + offs_d_inner[None, :] * str_v_d
    v_tile = tl.load(v_ptrs, mask=mask_k[:, None], other=0.0).to(tl.float32)

    acc_dk = tl.zeros((BLOCK_K, BLOCK_D), dtype=tl.float32)
    acc_dv = tl.zeros((BLOCK_K, BLOCK_D), dtype=tl.float32)

    for qb in range(ns_q_blocks):
        q_offset = qb * BLOCK_Q
        offs_q_abs = q_offset + tl.arange(0, BLOCK_Q)
        mask_q = offs_q_abs < seq_len

        q_ptrs = q_base + offs_q_abs[:, None] * str_q_s + offs_d_inner[None, :] * str_q_d
        q_tile = tl.load(q_ptrs, mask=mask_q[:, None], other=0.0).to(tl.float32)

        do_ptrs = do_base + offs_q_abs[:, None] * str_do_s + offs_d_inner[None, :] * str_do_d
        do_tile = tl.load(do_ptrs, mask=mask_q[:, None], other=0.0).to(tl.float32)

        l_ptrs = l_base + offs_q_abs * str_l_s
        l_tile = tl.load(l_ptrs, mask=mask_q, other=float("-inf"))

        o_ptrs = o_base + offs_q_abs[:, None] * str_o_s + offs_d_inner[None, :] * str_o_d
        o_tile = tl.load(o_ptrs, mask=mask_q[:, None], other=0.0).to(tl.float32)

        # Causal mask: q_pos >= k_pos
        causal_mask = (offs_q_abs[:, None] >= offs_k_abs[None, :])

        # S = Q @ K^T / sqrt(d)
        s = tl.dot(q_tile, k_tile.T)
        s = s * inv_scale

        # P = exp(S - L) * causal_mask
        p = tl.exp(s - l_tile[:, None])
        p = p * causal_mask

        # dp = dO @ V^T
        dp = tl.dot(do_tile, v_tile.T)

        # D = rowsum(dO * O)
        d_row = tl.sum(do_tile * o_tile, axis=1, keep_dims=True)

        # dS = P * (dp - D)
        ds = p * (dp - d_row)

        # dK += dS^T @ Q / sqrt(d)
        acc_dk = tl.dot(ds.T, q_tile, acc_dk)

        # dV += P^T @ dO
        acc_dv = tl.dot(p.T, do_tile, acc_dv)

    # Store results divided by sqrt(d) for dK
    dk_ptrs = dk_base + offs_k_abs[:, None] * str_dk_s + offs_d_inner[None, :] * str_dk_d
    tl.store(dk_ptrs, acc_dk * inv_scale, mask=mask_k[:, None])

    dv_ptrs = dv_base + offs_k_abs[:, None] * str_dv_s + offs_d_inner[None, :] * str_dv_d
    tl.store(dv_ptrs, acc_dv, mask=mask_k[:, None])


@triton.jit
def _dq_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, do_ptr, l_ptr,
    dq_ptr,
    seq_len, head_dim, n_heads,
    str_q_b, str_q_h, str_q_s, str_q_d,
    str_k_b, str_k_h, str_k_s, str_k_d,
    str_v_b, str_v_h, str_v_s, str_v_d,
    str_o_b, str_o_h, str_o_s, str_o_d,
    str_do_b, str_do_h, str_do_s, str_do_d,
    str_l_b, str_l_h, str_l_s,
    str_dq_b, str_dq_h, str_dq_s, str_dq_d,
    ns_k_blocks,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)

    n_heads_val = tl.num_programs(0)
    
    b_idx = pid_bh // n_heads_val
    h_idx = pid_bh % n_heads_val

    q_offset = pid_q * BLOCK_Q

    inv_scale = 1.0 / tl.sqrt(tl.cast(head_dim, tl.float32))

    offs_q_inner = tl.arange(0, BLOCK_Q)
    offs_d_inner = tl.arange(0, BLOCK_D)

    # Base offsets for this (b, h)
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
    q_tile = tl.load(q_ptrs, mask=mask_q[:, None], other=0.0).to(tl.float32)

    do_ptrs = do_base + offs_q_abs[:, None] * str_do_s + offs_d_inner[None, :] * str_do_d
    do_tile = tl.load(do_ptrs, mask=mask_q[:, None], other=0.0).to(tl.float32)

    l_ptrs = l_base + offs_q_abs * str_l_s
    l_tile = tl.load(l_ptrs, mask=mask_q, other=float("-inf"))

    o_ptrs = o_base + offs_q_abs[:, None] * str_o_s + offs_d_inner[None, :] * str_o_d
    o_tile = tl.load(o_ptrs, mask=mask_q[:, None], other=0.0).to(tl.float32)

    # D = rowsum(dO * O)
    d_row = tl.sum(do_tile * o_tile, axis=1, keep_dims=True)

    acc_dq = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)

    for kb in range(ns_k_blocks):
        kv_offset = kb * BLOCK_K
        offs_k_abs = kv_offset + tl.arange(0, BLOCK_K)
        mask_k = offs_k_abs < seq_len

        # Causal mask: q_pos >= k_pos
        causal_mask = (offs_q_abs[:, None] >= offs_k_abs[None, :])

        k_ptrs = k_base + offs_k_abs[:, None] * str_k_s + offs_d_inner[None, :] * str_k_d
        k_tile = tl.load(k_ptrs, mask=mask_k[:, None], other=0.0).to(tl.float32)

        v_ptrs = v_base + offs_k_abs[:, None] * str_v_s + offs_d_inner[None, :] * str_v_d
        v_tile = tl.load(v_ptrs, mask=mask_k[:, None], other=0.0).to(tl.float32)

        # S = Q @ K^T / sqrt(d)
        s = tl.dot(q_tile, k_tile.T)
        s = s * inv_scale

        # P = exp(S - L) * causal_mask
        p = tl.exp(s - l_tile[:, None])
        p = p * causal_mask

        # dp = dO @ V^T
        dp = tl.dot(do_tile, v_tile.T)

        # dS = P * (dp - D)
        ds = p * (dp - d_row)

        # dQ += dS @ K / sqrt(d)
        acc_dq = tl.dot(ds, k_tile, acc_dq)

    dq_ptrs = dq_base + offs_q_abs[:, None] * str_dq_s + offs_d_inner[None, :] * str_dq_d
    tl.store(dq_ptrs, acc_dq * inv_scale, mask=mask_q[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute causal multi-head attention backward: dQ, dK, dV."""
    torch.cuda.set_device(Q.device)

    B, H, S, D = Q.shape
    BH = B * H
    seq_len = S
    head_dim = D

    BLOCK_Q = 64
    BLOCK_K = 64
    BLOCK_D = 128

    ns_q_blocks = triton.cdiv(seq_len, BLOCK_Q)
    ns_k_blocks = triton.cdiv(seq_len, BLOCK_K)

    # Strides for all tensors
    sq = Q.stride()
    sk = K.stride()
    sv = V.stride()
    so = O.stride()
    sdo = dO.stride()
    sl = L.stride()
    sdq = dQ.stride()
    sdk = dK.stride()
    sdv = dV.stride()

    # dKV kernel: grid = (BH, ns_k_blocks)
    grid_dkv = (BH, ns_k_blocks)

    _dkv_kernel[grid_dkv](
        Q, K, V, O, dO, L,
        dK, dV,
        seq_len, head_dim, H,
        sq[0], sq[1], sq[2], sq[3],
        sk[0], sk[1], sk[2], sk[3],
        sv[0], sv[1], sv[2], sv[3],
        so[0], so[1], so[2], so[3],
        sdo[0], sdo[1], sdo[2], sdo[3],
        sl[0], sl[1], sl[2],
        sdk[0], sdk[1], sdk[2], sdk[3],
        sdv[0], sdv[1], sdv[2], sdv[3],
        ns_q_blocks,
        BLOCK_Q=BLOCK_Q,
        BLOCK_K=BLOCK_K,
        BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=3,
    )

    # dQ kernel: grid = (BH, ns_q_blocks)
    grid_dq = (BH, ns_q_blocks)

    _dq_kernel[grid_dq](
        Q, K, V, O, dO, L,
        dQ,
        seq_len, head_dim, H,
        sq[0], sq[1], sq[2], sq[3],
        sk[0], sk[1], sk[2], sk[3],
        sv[0], sv[1], sv[2], sv[3],
        so[0], so[1], so[2], so[3],
        sdo[0], sdo[1], sdo[2], sdo[3],
        sl[0], sl[1], sl[2],
        sdq[0], sdq[1], sdq[2], sdq[3],
        ns_k_blocks,
        BLOCK_Q=BLOCK_Q,
        BLOCK_K=BLOCK_K,
        BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=3,
    )