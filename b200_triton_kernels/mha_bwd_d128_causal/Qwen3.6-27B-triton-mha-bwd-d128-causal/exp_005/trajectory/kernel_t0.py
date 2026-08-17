import torch
import triton
import triton.language as tl


@triton.jit
def _preprocess_kernel(
    o_ptr, do_ptr, d_ptr,
    n_elements,
    stride_b_o, stride_h_o, stride_s_o, stride_d_o,
    BLOCK_D: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = idx < n_elements
    d_accum = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for di in range(0, BLOCK_D):
        ptr_o = o_ptr + idx * stride_b_o + tl.full((BLOCK_D,), 0.0).to(tl.int64)
        # This approach is wrong - need different structuring
        pass
    # Re-do with simpler per-element loop


@triton.jit
def _preprocess_element_kernel(
    o_ptr, do_ptr, d_ptr,
    n_elements, stride_b, stride_h, stride_s, stride_d,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute D = row_sum(dO * O) for each (b,h,s) element."""
    elem_id = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = elem_id < n_elements

    b = (elem_id // (stride_b)) % (stride_b // stride_h) if stride_b > stride_h else 0
    # Simpler: just iterate over d dimension for each (b,h,s)
    # Use direct offset-based computation
    d_val = tl.sum(
        tl.load(o_ptr + elem_id * stride_d, mask=mask, other=0.0) *
        tl.load(do_ptr + elem_id * stride_d, mask=mask, other=0.0),
        axis=0
    )
    # This won't work for vectorized - need element-by-element approach


# Final clean implementation below


@triton.jit
def _preprocess_kernel(
    o_ptr, do_ptr, d_ptr,
    b, h, seq, d,
    stride_bo, stride_ho, stride_so, stride_do,
    stride_bd, stride_hd, stride_sd,
    BLOCK_SEQ: tl.constexpr,
):
    pid = tl.program_id(0)
    b_idx = pid // (h * tl.cdiv(seq, BLOCK_SEQ))
    remainder = pid % (h * tl.cdiv(seq, BLOCK_SEQ))
    h_idx = remainder // tl.cdiv(seq, BLOCK_SEQ)
    sq_chunk = remainder % tl.cdiv(seq, BLOCK_SEQ)

    offs_s = sq_chunk * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
    mask_s = offs_s < seq

    d_block = tl.zeros((BLOCK_SEQ,), dtype=tl.float32)

    base_o = o_ptr + b_idx * stride_bo + h_idx * stride_ho
    base_do = do_ptr + b_idx * stride_bd + h_idx * stride_hd

    for di in range(0, d):
        o_offs = base_o + offs_s * stride_so + di * stride_do
        do_offs = base_do + offs_s * stride_sd + di
        o_vals = tl.load(o_offs, mask=mask_s, other=0.0)
        do_vals = tl.load(do_offs, mask=mask_s, other=0.0)
        d_block = d_block + o_vals * do_vals

    d_offs = d_ptr + b_idx * (h * seq) + h_idx * seq + offs_s
    tl.store(d_offs, d_block, mask=mask_s)


@triton.jit
def _dkdv_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, do_ptr, l_ptr, d_ptr, dk_ptr, dv_ptr,
    b_total, h_total, seq, d,
    stride_bq, stride_hq, stride_sq, stride_dq,
    stride_bk, stride_hk, stride_sk, stride_dk,
    stride_bv, stride_hv, stride_sv, stride_dv,
    stride_bd, stride_hd, stride_sd,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute dK and dV.
    Each program handles one (batch, head, kv_chunk) and loops over q chunks.
    """
    pid = tl.program_id(0)
    cdiv_seq_n = tl.cdiv(seq, BLOCK_N)
    
    b_idx = pid // (h_total * cdiv_seq_n)
    rem = pid % (h_total * cdiv_seq_n)
    h_idx = rem // cdiv_seq_n
    n_chunk = rem % cdiv_seq_n

    offs_n = n_chunk * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < seq

    offs_d = tl.arange(0, BLOCK_D)

    # Load K tile once per program
    base_k = k_ptr + b_idx * stride_bk + h_idx * stride_hk
    k_ptrs = base_k + offs_n[:, None] * stride_sk + offs_d[None, :] * stride_dk
    k_tile = tl.load(k_ptrs, mask=(mask_n[:, None] & (offs_d[None, :] < d)), other=0.0)

    # Load V tile once per program
    base_v = v_ptr + b_idx * stride_bv + h_idx * stride_hv
    v_ptrs = base_v + offs_n[:, None] * stride_sv + offs_d[None, :] * stride_dv
    v_tile = tl.load(v_ptrs, mask=(mask_n[:, None] & (offs_d[None, :] < d)), other=0.0)

    dK_acc = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    dV_acc = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)

    cdiv_seq_m = tl.cdiv(seq, BLOCK_M)
    for m_chunk in range(cdiv_seq_m):
        offs_m = m_chunk * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < seq

        # Load Q
        base_q = q_ptr + b_idx * stride_bq + h_idx * stride_hq
        q_ptrs = base_q + offs_m[:, None] * stride_sq + offs_d[None, :] * stride_dq
        q_tile = tl.load(q_ptrs, mask=(mask_m[:, None] & (offs_d[None, :] < d)), other=0.0)

        # Load dO
        base_do = do_ptr + b_idx * stride_bd + h_idx * stride_hd
        do_ptrs = base_do + offs_m[:, None] * stride_sd + offs_d[None, :]
        do_tile = tl.load(do_ptrs, mask=(mask_m[:, None] & (offs_d[None, :] < d)), other=0.0)

        # Load L (logsumexp)
        l_ptrs = l_ptr + b_idx * (h_total * seq) + h_idx * seq + offs_m
        l_vals = tl.load(l_ptrs, mask=mask_m, other=0.0)

        # Load D
        d_ptrs = d_ptr + b_idx * (h_total * seq) + h_idx * seq + offs_m
        d_vals = tl.load(d_ptrs, mask=mask_m, other=0.0)

        # S = Q @ K.T * scale -> [BM, BN]
        S = tl.dot(q_tile, k_tile.T) * scale

        # Apply causal mask: P[m,n]=0 when m<n
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        causal_mask = causal_mask & (mask_m[:, None] & mask_n[None, :])

        # P = exp(S - L[:,None])
        P = tl.exp(S - l_vals[:, None])
        P = tl.where(causal_mask, P, 0.0)

        # dP = dO @ V.T -> [BM, BN]
        dP = tl.dot(do_tile, v_tile.T)

        # dS = P * (dP - D[:,None]) * scale
        dS = P * (dP - d_vals[:, None]) * scale
        dS = tl.where(causal_mask, dS, 0.0)

        # dK += dS.T @ Q
        dK_acc = tl.dot(dS.T, q_tile, dK_acc)

        # dV += P.T @ dO
        dV_acc = tl.dot(P.T, do_tile, dV_acc)

    # Store dK
    dk_ptrs = dk_ptr + b_idx * stride_bk + h_idx * stride_hk + offs_n[:, None] * stride_sk + offs_d[None, :] * stride_dk
    tl.store(dk_ptrs, dK_acc.to(tl.bfloat16), mask=mask_n[:, None] & (offs_d[None, :] < d))

    # Store dV
    dv_ptrs = dv_ptr + b_idx * stride_bv + h_idx * stride_hv + offs_n[:, None] * stride_sv + offs_d[None, :] * stride_dv
    tl.store(dv_ptrs, dV_acc.to(tl.bfloat16), mask=mask_n[:, None] & (offs_d[None, :] < d))


@triton.jit
def _dq_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, do_ptr, l_ptr, d_ptr, dq_ptr,
    b_total, h_total, seq, d,
    stride_bq, stride_hq, stride_sq, stride_dq,
    stride_bk, stride_hk, stride_sk, stride_dk_in,
    stride_bv, stride_hv, stride_sv, stride_dv,
    stride_bd, stride_hd, stride_sd,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute dQ.
    Each program handles one (batch, head, q_chunk) and loops over kv chunks.
    """
    pid = tl.program_id(0)
    cdiv_seq_m = tl.cdiv(seq, BLOCK_M)
    
    b_idx = pid // (h_total * cdiv_seq_m)
    rem = pid % (h_total * cdiv_seq_m)
    h_idx = rem // cdiv_seq_m
    m_chunk = rem % cdiv_seq_m

    offs_m = m_chunk * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < seq

    offs_d = tl.arange(0, BLOCK_D)

    # Load Q tile once per program
    base_q = q_ptr + b_idx * stride_bq + h_idx * stride_hq
    q_ptrs = base_q + offs_m[:, None] * stride_sq + offs_d[None, :] * stride_dq
    q_tile = tl.load(q_ptrs, mask=(mask_m[:, None] & (offs_d[None, :] < d)), other=0.0)

    # Load dO tile once per program
    base_do = do_ptr + b_idx * stride_bd + h_idx * stride_hd
    do_ptrs = base_do + offs_m[:, None] * stride_sd + offs_d[None, :]
    do_tile = tl.load(do_ptrs, mask=(mask_m[:, None] & (offs_d[None, :] < d)), other=0.0)

    # Load L
    l_ptrs = l_ptr + b_idx * (h_total * seq) + h_idx * seq + offs_m
    l_vals = tl.load(l_ptrs, mask=mask_m, other=0.0)

    # Load D
    d_ptrs = d_ptr + b_idx * (h_total * seq) + h_idx * seq + offs_m
    d_vals = tl.load(d_ptrs, mask=mask_m, other=0.0)

    dQ_acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    cdiv_seq_n = tl.cdiv(seq, BLOCK_N)
    for n_chunk in range(cdiv_seq_n):
        offs_n = n_chunk * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < seq

        # Load K
        base_k = k_ptr + b_idx * stride_bk + h_idx * stride_hk
        k_ptrs = base_k + offs_n[:, None] * stride_sk + offs_d[None, :] * stride_dk_in
        k_tile = tl.load(k_ptrs, mask=(mask_n[:, None] & (offs_d[None, :] < d)), other=0.0)

        # Load V
        base_v = v_ptr + b_idx * stride_bv + h_idx * stride_hv
        v_ptrs = base_v + offs_n[:, None] * stride_sv + offs_d[None, :] * stride_dv
        v_tile = tl.load(v_ptrs, mask=(mask_n[:, None] & (offs_d[None, :] < d)), other=0.0)

        # S = Q @ K.T * scale
        S = tl.dot(q_tile, k_tile.T) * scale

        # Causal mask
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        causal_mask = causal_mask & (mask_m[:, None] & mask_n[None, :])

        # P = exp(S - L)
        P = tl.exp(S - l_vals[:, None])
        P = tl.where(causal_mask, P, 0.0)

        # dP = dO @ V.T
        dP = tl.dot(do_tile, v_tile.T)

        # dS = P * (dP - D) * scale
        dS = P * (dP - d_vals[:, None]) * scale
        dS = tl.where(causal_mask, dS, 0.0)

        # dQ += dS @ K
        dQ_acc = tl.dot(dS, k_tile, dQ_acc)

    # Store dQ
    dq_ptrs = dq_ptr + b_idx * stride_bq + h_idx * stride_hq + offs_m[:, None] * stride_sq + offs_d[None, :] * stride_dq
    tl.store(dq_ptrs, dQ_acc.to(tl.bfloat16), mask=mask_m[:, None] & (offs_d[None, :] < d))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Causal multi-head attention backward pass.
    
    Computes dQ, dK, dV gradients given forward pass results and upstream gradient dO.
    """
    torch.cuda.set_device(Q.device)
    
    b, h, seq, d = Q.shape
    
    assert Q.shape == (b, h, seq, d), f"Q shape mismatch: {Q.shape}"
    assert K.shape == (b, h, seq, d), f"K shape mismatch: {K.shape}"
    assert V.shape == (b, h, seq, d), f"V shape mismatch: {V.shape}"
    assert O.shape == (b, h, seq, d), f"O shape mismatch: {O.shape}"
    assert dO.shape == (b, h, seq, d), f"dO shape mismatch: {dO.shape}"
    
    # Handle L shape - reference unsqueezes if 3D
    if L.dim() == 3:
        l_shape = (b, h, seq)
    else:
        l_shape = L.shape[:3]
    
    scale = 1.0 / (d ** 0.5)
    
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = 128
    
    # Strides
    sb_q = Q.stride(0)
    sh_q = Q.stride(1)
    ss_q = Q.stride(2)
    sd_q = Q.stride(3)
    
    sb_k = K.stride(0)
    sh_k = K.stride(1)
    ss_k = K.stride(2)
    sd_k = K.stride(3)
    
    sb_v = V.stride(0)
    sh_v = V.stride(1)
    ss_v = V.stride(2)
    sd_v = V.stride(3)
    
    sb_o = O.stride(0)
    sh_o = O.stride(1)
    ss_o = O.stride(2)
    sd_o = O.stride(3)
    
    # Preallocate D = row_sum(dO * O) with shape (B, H, S)
    D = torch.empty((b, h, seq), dtype=torch.float32, device=Q.device)
    sd_b = D.stride(0)
    sd_h = D.stride(1)
    sd_s = D.stride(2)
    
    # Step 1: Compute D
    n_elements_1d = b * h * seq
    grid_preproc = (triton.cdiv(n_elements_1d, BLOCK_M),)
    _preprocess_kernel[grid_preproc](
        O, dO, D,
        b, h, seq, d,
        sb_o, sh_o, ss_o, sd_o,
        sd_b, sd_h, sd_s,
        BLOCK_SEQ=BLOCK_M,
        num_warps=2,
        num_stages=1,
    )
    
    # Step 2: Compute dK and dV
    # Grid: one program per (batch, head, kv_chunk)
    n_kv_chunks = triton.cdiv(seq, BLOCK_N)
    grid_dkdv = (b * h * n_kv_chunks,)
    
    _dkdv_kernel[grid_dkdv](
        Q, K, V, O, dO, L, D, dK, dV,
        b, h, seq, d,
        sb_q, sh_q, ss_q, sd_q,
        sb_k, sh_k, ss_k, sd_k,
        sb_v, sh_v, ss_v, sd_v,
        sd_b, sd_h, sd_s,
        scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=2,
    )
    
    # Step 3: Compute dQ
    # Grid: one program per (batch, head, q_chunk)
    n_q_chunks = triton.cdiv(seq, BLOCK_M)
    grid_dq = (b * h * n_q_chunks,)
    
    _dq_kernel[grid_dq](
        Q, K, V, O, dO, L, D, dQ,
        b, h, seq, d,
        sb_q, sh_q, ss_q, sd_q,
        sb_k, sh_k, ss_k, sd_k,
        sb_v, sh_v, ss_v, sd_v,
        sd_b, sd_h, sd_s,
        scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=2,
    )