import torch
import triton
import triton.language as tl


@triton.jit
def _dkdv_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, do_ptr, lse_ptr, dk_ptr, dv_ptr,
    b_total, h_total, seq, d,
    stride_bq, stride_hq, stride_sq, stride_dq,
    stride_bk, stride_hk, stride_sk, stride_dk_in,
    stride_bv, stride_hv, stride_sv, stride_dv,
    stride_bo, stride_ho, stride_so, stride_do_in,
    stride_bdo, stride_hdo, stride_sdo, stride_ddo,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    cdiv_seq_n = tl.cdiv(seq, BLOCK_N)

    b_idx = pid // (h_total * cdiv_seq_n)
    rem = pid % (h_total * cdiv_seq_n)
    h_idx = rem // cdiv_seq_n
    n_chunk = rem % cdiv_seq_n

    offs_n = n_chunk * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < seq
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < d

    base_k = k_ptr + b_idx * stride_bk + h_idx * stride_hk
    k_ptrs = base_k + offs_n[:, None] * stride_sk + offs_d[None, :] * stride_dk_in
    k_tile = tl.load(k_ptrs, mask=(mask_n[:, None] & mask_d[None, :]), other=0.0).to(tl.float32)

    base_v = v_ptr + b_idx * stride_bv + h_idx * stride_hv
    v_ptrs = base_v + offs_n[:, None] * stride_sv + offs_d[None, :] * stride_dv
    v_tile = tl.load(v_ptrs, mask=(mask_n[:, None] & mask_d[None, :]), other=0.0).to(tl.float32)

    dK_acc = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    dV_acc = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)

    cdiv_seq_m = tl.cdiv(seq, BLOCK_M)
    for m_chunk in range(cdiv_seq_m):
        offs_m = m_chunk * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < seq

        base_q = q_ptr + b_idx * stride_bq + h_idx * stride_hq
        q_ptrs = base_q + offs_m[:, None] * stride_sq + offs_d[None, :] * stride_dq
        q_tile = tl.load(q_ptrs, mask=(mask_m[:, None] & mask_d[None, :]), other=0.0).to(tl.float32)

        base_do = do_ptr + b_idx * stride_bdo + h_idx * stride_hdo
        do_ptrs = base_do + offs_m[:, None] * stride_sdo + offs_d[None, :] * stride_ddo
        do_tile = tl.load(do_ptrs, mask=(mask_m[:, None] & mask_d[None, :]), other=0.0).to(tl.float32)

        base_o = o_ptr + b_idx * stride_bo + h_idx * stride_ho
        o_ptrs = base_o + offs_m[:, None] * stride_so + offs_d[None, :] * stride_do_in
        o_tile = tl.load(o_ptrs, mask=(mask_m[:, None] & mask_d[None, :]), other=0.0).to(tl.float32)

        d_vals = tl.sum(do_tile * o_tile, axis=1)

        lse_ptrs = lse_ptr + b_idx * (h_total * seq) + h_idx * seq + offs_m
        l_vals = tl.load(lse_ptrs, mask=mask_m, other=0.0)

        S = tl.dot(q_tile, k_tile.T) * scale

        causal_mask = (offs_m[:, None] >= offs_n[None, :]) & (mask_m[:, None] & mask_n[None, :])

        P = tl.exp(S - l_vals[:, None])
        P = tl.where(causal_mask, P, 0.0)

        dP = tl.dot(do_tile, v_tile.T)

        dS = P * (dP - d_vals[:, None]) * scale
        dS = tl.where(causal_mask, dS, 0.0)

        dK_acc = tl.dot(dS.T, q_tile, acc=dK_acc)
        dV_acc = tl.dot(P.T, do_tile, acc=dV_acc)

    dk_ptrs = dk_ptr + b_idx * stride_bk + h_idx * stride_hk + offs_n[:, None] * stride_sk + offs_d[None, :] * stride_dk_in
    tl.store(dk_ptrs, dK_acc.to(tl.bfloat16), mask=(mask_n[:, None] & mask_d[None, :]))

    dv_ptrs = dv_ptr + b_idx * stride_bv + h_idx * stride_hv + offs_n[:, None] * stride_sv + offs_d[None, :] * stride_dv
    tl.store(dv_ptrs, dV_acc.to(tl.bfloat16), mask=(mask_n[:, None] & mask_d[None, :]))


@triton.jit
def _dq_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, do_ptr, lse_ptr, dq_ptr,
    b_total, h_total, seq, d,
    stride_bq, stride_hq, stride_sq, stride_dq,
    stride_bk, stride_hk, stride_sk, stride_dk_in,
    stride_bv, stride_hv, stride_sv, stride_dv,
    stride_bo, stride_ho, stride_so, stride_do_in,
    stride_bdo, stride_hdo, stride_sdo, stride_ddo,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    cdiv_seq_m = tl.cdiv(seq, BLOCK_M)

    b_idx = pid // (h_total * cdiv_seq_m)
    rem = pid % (h_total * cdiv_seq_m)
    h_idx = rem // cdiv_seq_m
    m_chunk = rem % cdiv_seq_m

    offs_m = m_chunk * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < seq
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < d

    base_q = q_ptr + b_idx * stride_bq + h_idx * stride_hq
    q_ptrs = base_q + offs_m[:, None] * stride_sq + offs_d[None, :] * stride_dq
    q_tile = tl.load(q_ptrs, mask=(mask_m[:, None] & mask_d[None, :]), other=0.0).to(tl.float32)

    base_do = do_ptr + b_idx * stride_bdo + h_idx * stride_hdo
    do_ptrs = base_do + offs_m[:, None] * stride_sdo + offs_d[None, :] * stride_ddo
    do_tile = tl.load(do_ptrs, mask=(mask_m[:, None] & mask_d[None, :]), other=0.0).to(tl.float32)

    base_o = o_ptr + b_idx * stride_bo + h_idx * stride_ho
    o_ptrs = base_o + offs_m[:, None] * stride_so + offs_d[None, :] * stride_do_in
    o_tile = tl.load(o_ptrs, mask=(mask_m[:, None] & mask_d[None, :]), other=0.0).to(tl.float32)

    d_vals = tl.sum(do_tile * o_tile, axis=1)

    lse_ptrs = lse_ptr + b_idx * (h_total * seq) + h_idx * seq + offs_m
    l_vals = tl.load(lse_ptrs, mask=mask_m, other=0.0)

    dQ_acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    cdiv_seq_n = tl.cdiv(seq, BLOCK_N)
    for n_chunk in range(cdiv_seq_n):
        offs_n = n_chunk * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < seq

        base_k = k_ptr + b_idx * stride_bk + h_idx * stride_hk
        k_ptrs = base_k + offs_n[:, None] * stride_sk + offs_d[None, :] * stride_dk_in
        k_tile = tl.load(k_ptrs, mask=(mask_n[:, None] & mask_d[None, :]), other=0.0).to(tl.float32)

        base_v = v_ptr + b_idx * stride_bv + h_idx * stride_hv
        v_ptrs = base_v + offs_n[:, None] * stride_sv + offs_d[None, :] * stride_dv
        v_tile = tl.load(v_ptrs, mask=(mask_n[:, None] & mask_d[None, :]), other=0.0).to(tl.float32)

        S = tl.dot(q_tile, k_tile.T) * scale

        causal_mask = (offs_m[:, None] >= offs_n[None, :]) & (mask_m[:, None] & mask_n[None, :])

        P = tl.exp(S - l_vals[:, None])
        P = tl.where(causal_mask, P, 0.0)

        dP = tl.dot(do_tile, v_tile.T)

        dS = P * (dP - d_vals[:, None]) * scale
        dS = tl.where(causal_mask, dS, 0.0)

        dQ_acc = tl.dot(dS, k_tile, acc=dQ_acc)

    dq_ptrs = dq_ptr + b_idx * stride_bq + h_idx * stride_hq + offs_m[:, None] * stride_sq + offs_d[None, :] * stride_dq
    tl.store(dq_ptrs, dQ_acc.to(tl.bfloat16), mask=(mask_m[:, None] & mask_d[None, :]))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Causal multi-head attention backward pass."""
    torch.cuda.set_device(Q.device)

    b, h, seq, d = Q.shape

    scale = 1.0 / (d ** 0.5)

    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_D = 128

    sb_q = Q.stride(0); sh_q = Q.stride(1); ss_q = Q.stride(2); sd_q = Q.stride(3)
    sb_k = K.stride(0); sh_k = K.stride(1); ss_k = K.stride(2); sd_k = K.stride(3)
    sb_v = V.stride(0); sh_v = V.stride(1); ss_v = V.stride(2); sd_v = V.stride(3)
    sb_o = O.stride(0); sh_o = O.stride(1); ss_o = O.stride(2); sd_o = O.stride(3)
    sb_do = dO.stride(0); sh_do = dO.stride(1); ss_do = dO.stride(2); sd_do = dO.stride(3)

    L = L.contiguous()

    # Compute dK and dV: one program per (batch, head, kv_chunk)
    n_kv_chunks = triton.cdiv(seq, BLOCK_N)
    grid_dkdv = (b * h * n_kv_chunks,)

    _dkdv_kernel[grid_dkdv](
        Q, K, V, O, dO, L, dK, dV,
        b, h, seq, d,
        sb_q, sh_q, ss_q, sd_q,
        sb_k, sh_k, ss_k, sd_k,
        sb_v, sh_v, ss_v, sd_v,
        sb_o, sh_o, ss_o, sd_o,
        sb_do, sh_do, ss_do, sd_do,
        scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        num_warps=8,
        num_stages=3,
    )

    # Compute dQ: one program per (batch, head, q_chunk)
    n_q_chunks = triton.cdiv(seq, BLOCK_M)
    grid_dq = (b * h * n_q_chunks,)

    _dq_kernel[grid_dq](
        Q, K, V, O, dO, L, dQ,
        b, h, seq, d,
        sb_q, sh_q, ss_q, sd_q,
        sb_k, sh_k, ss_k, sd_k,
        sb_v, sh_v, ss_v, sd_v,
        sb_o, sh_o, ss_o, sd_o,
        sb_do, sh_do, ss_do, sd_do,
        scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        num_warps=8,
        num_stages=3,
    )