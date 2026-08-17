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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_n = tl.program_id(1)

    b_idx = pid_bh // n_heads
    h_idx = pid_bh % n_heads
    n_offset = pid_n * BLOCK_N

    inv_scale = 1.0 / tl.sqrt(tl.cast(head_dim, tl.float32))

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # Base pointers for this (b, h)
    base_q = q_ptr + b_idx * str_q_b + h_idx * str_q_h
    base_k = k_ptr + b_idx * str_k_b + h_idx * str_k_h
    base_v = v_ptr + b_idx * str_v_b + h_idx * str_v_h
    base_o = o_ptr + b_idx * str_o_b + h_idx * str_o_h
    base_do = do_ptr + b_idx * str_do_b + h_idx * str_do_h
    base_l = l_ptr + b_idx * str_l_b + h_idx * str_l_h
    base_dk = dk_ptr + b_idx * str_dk_b + h_idx * str_dk_h
    base_dv = dv_ptr + b_idx * str_dv_b + h_idx * str_dv_h

    # Load K and V tile once
    mask_n = offs_n < seq_len
    k_ptrs = base_k + offs_n[:, None] * str_k_s + offs_d[None, :] * str_k_d
    k_t = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)

    v_ptrs = base_v + offs_n[:, None] * str_v_s + offs_d[None, :] * str_v_d
    v_t = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)

    acc_dk = tl.zeros((BLOCK_N, BLOCK_DMODEL), dtype=tl.float32)
    acc_dv = tl.zeros((BLOCK_N, BLOCK_DMODEL), dtype=tl.float32)

    num_m_blocks = tl.cdiv(seq_len, BLOCK_M)
    for m_block in range(num_m_blocks):
        m_offset = m_block * BLOCK_M

        # Load Q, dO, L, O for this query block
        mask_m = (m_offset + offs_m) < seq_len
        q_ptrs = base_q + offs_m[:, None] * str_q_s + offs_d[None, :] * str_q_d
        q_t = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)

        do_ptrs = base_do + offs_m[:, None] * str_do_s + offs_d[None, :] * str_do_d
        do_t = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)

        l_ptrs = base_l + offs_m * str_l_s
        l_t = tl.load(l_ptrs, mask=mask_m, other=float("-inf"))

        o_ptrs = base_o + offs_m[:, None] * str_o_s + offs_d[None, :] * str_o_d
        o_t = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)

        # Causal mask: query pos >= key pos
        q_pos = m_offset + offs_m[:, None]
        k_pos = n_offset + offs_n[None, :]
        causal_mask = q_pos >= k_pos

        # Score matrix S = Q @ K^T / sqrt(d)
        s = tl.dot(q_t, k_t.T) * inv_scale

        # Softmax probs P = exp(S - L) * causal
        p = tl.exp(s - l_t[:, None])
        p = p * causal_mask.to(tl.float32)

        # dP = dO @ V^T
        dp = tl.dot(do_t, v_t.T)

        # D = rowsum(dO * O)
        d_row = tl.sum(do_t * o_t, axis=1, keep_dims=True)

        # dS = P * (dP - D)
        ds = p * (dp - d_row)

        # Accumulate gradients
        acc_dk += tl.dot(ds.T, q_t)
        acc_dv += tl.dot(p.T, do_t)

    # Store results with proper scaling
    dk_ptrs = base_dk + offs_n[:, None] * str_dk_s + offs_d[None, :] * str_dk_d
    tl.store(dk_ptrs, (acc_dk * inv_scale).to(tl.bfloat16), mask=mask_n[:, None])

    dv_ptrs = base_dv + offs_n[:, None] * str_dv_s + offs_d[None, :] * str_dv_d
    tl.store(dv_ptrs, acc_dv.to(tl.bfloat16), mask=mask_n[:, None])


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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    b_idx = pid_bh // n_heads
    h_idx = pid_bh % n_heads
    m_offset = pid_m * BLOCK_M

    inv_scale = 1.0 / tl.sqrt(tl.cast(head_dim, tl.float32))

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # Base pointers for this (b, h)
    base_q = q_ptr + b_idx * str_q_b + h_idx * str_q_h
    base_k = k_ptr + b_idx * str_k_b + h_idx * str_k_h
    base_v = v_ptr + b_idx * str_v_b + h_idx * str_v_h
    base_o = o_ptr + b_idx * str_o_b + h_idx * str_o_h
    base_do = do_ptr + b_idx * str_do_b + h_idx * str_do_h
    base_l = l_ptr + b_idx * str_l_b + h_idx * str_l_h
    base_dq = dq_ptr + b_idx * str_dq_b + h_idx * str_dq_h

    # Load Q, dO, L, O for this query block
    mask_m = (m_offset + offs_m) < seq_len
    q_ptrs = base_q + offs_m[:, None] * str_q_s + offs_d[None, :] * str_q_d
    q_t = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)

    do_ptrs = base_do + offs_m[:, None] * str_do_s + offs_d[None, :] * str_do_d
    do_t = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)

    l_ptrs = base_l + offs_m * str_l_s
    l_t = tl.load(l_ptrs, mask=mask_m, other=float("-inf"))

    o_ptrs = base_o + offs_m[:, None] * str_o_s + offs_d[None, :] * str_o_d
    o_t = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)

    # D = rowsum(dO * O)
    d_row = tl.sum(do_t * o_t, axis=1, keep_dims=True)

    acc_dq = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)

    num_n_blocks = tl.cdiv(seq_len, BLOCK_N)
    for n_block in range(num_n_blocks):
        n_offset = n_block * BLOCK_N
        mask_n = (n_offset + offs_n) < seq_len

        k_ptrs = base_k + offs_n[:, None] * str_k_s + offs_d[None, :] * str_k_d
        k_t = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)

        v_ptrs = base_v + offs_n[:, None] * str_v_s + offs_d[None, :] * str_v_d
        v_t = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)

        # Causal mask: query pos >= key pos
        q_pos = m_offset + offs_m[:, None]
        k_pos = n_offset + offs_n[None, :]
        causal_mask = q_pos >= k_pos

        s = tl.dot(q_t, k_t.T) * inv_scale

        p = tl.exp(s - l_t[:, None])
        p = p * causal_mask.to(tl.float32)

        dp = tl.dot(do_t, v_t.T)

        ds = p * (dp - d_row)

        acc_dq += tl.dot(ds, k_t)

    dq_ptrs = base_dq + offs_m[:, None] * str_dq_s + offs_d[None, :] * str_dq_d
    tl.store(dq_ptrs, (acc_dq * inv_scale).to(tl.bfloat16), mask=mask_m[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute causal multi-head attention backward: dQ, dK, dV."""
    torch.cuda.set_device(Q.device)

    B, H, S, D = Q.shape
    BH = B * H

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_DMODEL = 128

    ns_blocks = triton.cdiv(S, BLOCK_N)
    mq_blocks = triton.cdiv(S, BLOCK_M)

    sq = list(Q.stride())
    sk = list(K.stride())
    sv = list(V.stride())
    so = list(O.stride())
    sdo = list(dO.stride())
    sl = list(L.stride())
    sdq = list(dQ.stride())
    sdk = list(dK.stride())
    sdv = list(dV.stride())

    # Launch dKV kernel: grid = (BH, ns_blocks)
    grid_dkv = (BH, ns_blocks)
    _dkv_kernel[grid_dkv](
        Q, K, V, O, dO, L,
        dK, dV,
        S, D, H,
        sq[0], sq[1], sq[2], sq[3],
        sk[0], sk[1], sk[2], sk[3],
        sv[0], sv[1], sv[2], sv[3],
        so[0], so[1], so[2], so[3],
        sdo[0], sdo[1], sdo[2], sdo[3],
        sl[0], sl[1], sl[2],
        sdk[0], sdk[1], sdk[2], sdk[3],
        sdv[0], sdv[1], sdv[2], sdv[3],
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=4,
        num_stages=3,
    )

    # Launch dQ kernel: grid = (BH, mq_blocks)
    grid_dq = (BH, mq_blocks)
    _dq_kernel[grid_dq](
        Q, K, V, O, dO, L,
        dQ,
        S, D, H,
        sq[0], sq[1], sq[2], sq[3],
        sk[0], sk[1], sk[2], sk[3],
        sv[0], sv[1], sv[2], sv[3],
        so[0], so[1], so[2], so[3],
        sdo[0], sdo[1], sdo[2], sdo[3],
        sl[0], sl[1], sl[2],
        sdq[0], sdq[1], sdq[2], sdq[3],
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=4,
        num_stages=3,
    )