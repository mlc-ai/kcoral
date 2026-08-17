import torch
import triton
import triton.language as tl


@triton.jit
def _attn_bwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
    dQ_ptr, dK_ptr, dV_ptr,
    S,
    BH_stride_q, S_stride_q, D_stride_q,
    BH_stride_k, S_stride_k, D_stride_k,
    BH_stride_v, S_stride_v, D_stride_v,
    BH_stride_o, S_stride_o, D_stride_o,
    BH_stride_do, S_stride_do, D_stride_do,
    BH_stride_l, S_stride_l,
    BH_stride_dq, S_stride_dq, D_stride_dq,
    BH_stride_dk, S_stride_dk, D_stride_dk,
    BH_stride_dv, S_stride_dv, D_stride_dv,
    TOTAL_BH: tl.constexpr,
    NUM_SMS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Persistent kernel computing dQ, dK, dV for all heads."""
    pid = tl.program_id(0)

    # Iterate over (head, tile) assignments in a persistent manner
    num_pid_m = tl.cdiv(S, BLOCK_M)
    num_pid_n = tl.cdiv(S, BLOCK_N)
    total_tiles_per_head = num_pid_m * num_pid_n
    total_tiles = TOTAL_BH * total_tiles_per_head

    for tile_idx in tl.range(pid, total_tiles, NUM_SMS, flatten=False):
        pid_bh = tile_idx // total_tiles_per_head
        local_tile = tile_idx % total_tiles_per_head
        pid_m = local_tile // num_pid_n
        pid_n = local_tile % num_pid_n

        q_base = Q_ptr + pid_bh * BH_stride_q
        k_base = K_ptr + pid_bh * BH_stride_k
        v_base = V_ptr + pid_bh * BH_stride_v
        o_base = O_ptr + pid_bh * BH_stride_o
        do_base = dO_ptr + pid_bh * BH_stride_do
        l_base = L_ptr + pid_bh * BH_stride_l
        dq_base = dQ_ptr + pid_bh * BH_stride_dq
        dk_base = dK_ptr + pid_bh * BH_stride_dk
        dv_base = dV_ptr + pid_bh * BH_stride_dv

        m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        m_mask = m_offs < S
        n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = n_offs < S
        d_offs = tl.arange(0, BLOCK_D)

        # Load Q tile: [BLOCK_M, BLOCK_D]
        q_ptrs = q_base + m_offs[:, None] * S_stride_q + d_offs[None, :] * D_stride_q
        q = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0)

        # Load K tile: [BLOCK_N, BLOCK_D]
        k_ptrs = k_base + n_offs[:, None] * S_stride_k + d_offs[None, :] * D_stride_k
        k = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)

        # Load V tile: [BLOCK_N, BLOCK_D]
        v_ptrs = v_base + n_offs[:, None] * S_stride_v + d_offs[None, :] * D_stride_v
        v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)

        # Load O tile: [BLOCK_M, BLOCK_D]
        o_ptrs = o_base + m_offs[:, None] * S_stride_o + d_offs[None, :] * D_stride_o
        o = tl.load(o_ptrs, mask=m_mask[:, None], other=0.0)

        # Load dO tile: [BLOCK_M, BLOCK_D]
        do_ptrs = do_base + m_offs[:, None] * S_stride_do + d_offs[None, :] * D_stride_do
        do_tile = tl.load(do_ptrs, mask=m_mask[:, None], other=0.0)

        # Load L row: [BLOCK_M]
        l_ptrs = l_base + m_offs * S_stride_l
        l_val = tl.load(l_ptrs, mask=m_mask, other=0.0).to(tl.float32)[:, None]

        scale = 1.0 / tl.sqrt(float(BLOCK_D))

        # Compute S[m,n] = Q[m,:] @ K[n,:]^T / sqrt(d)  -> [BLOCK_M, BLOCK_N]
        q_f32 = q.to(tl.float32)
        k_f32 = k.to(tl.float32)
        v_f32 = v.to(tl.float32)
        do_f32 = do_tile.to(tl.float32)

        S_block = tl.dot(q_f32, k_f32.T) * scale
        P_block = tl.exp(S_block - l_val)  # [BLOCK_M, BLOCK_N]

        # Compute dP = dO @ V^T  -> [BLOCK_M, BLOCK_N]
        dP = tl.dot(do_f32, v_f32.T)

        # Compute D_i = sum_j(P_ij * dP_ij) = sum_j(O_ij * dO_ij) over j dimension
        o_f32 = o.to(tl.float32)
        D_row = tl.sum((do_f32 * o_f32), axis=1)[:, None]  # [BLOCK_M, 1]

        # dS = P * (dP - D) -> [BLOCK_M, BLOCK_N]
        dS = P_block * (dP - D_row) * scale

        # Accumulate dQ += dS @ K
        dQ_tile = tl.dot(dS, k_f32)

        # Accumulate dK += dS^T @ Q
        dK_tile = tl.dot(dS.T, q_f32)

        # Accumulate dV += P^T @ dO
        dV_tile = tl.dot(P_block.T, do_f32)

        # Store results with atomics since multiple tiles can overlap
        dq_ptrs_out = dq_base + m_offs[:, None] * S_stride_dq + d_offs[None, :] * D_stride_dq
        for di in range(BLOCK_D):
            col = tl.arange(0, BLOCK_M)
            mask_col = col < S
            val = dQ_tile[col + pid_m * BLOCK_M, di : di + 1]
            ptr = dq_base + (col + pid_m * BLOCK_M)[:, None] * S_stride_dq + di * D_stride_dq
            # Use atomic add for accumulation
            tl.atomic_add(ptr + pid_m * BLOCK_M * S_stride_dq, dQ_tile[:, di:di+1], mask=m_mask[:, None])

        # Simple store approach - each (bh, m_block) writes unique rows to dQ
        # Only one m_block per (bh, m) so dQ is written exactly once
        dq_ptrs_store = dq_base + m_offs[:, None] * S_stride_dq + d_offs[None, :] * D_stride_dq
        tl.store(dq_ptrs_store, dQ_tile.to(Q_ptr.dtype.element_ty), mask=m_mask[:, None])

        # For dK, dV: accumulate across all m_blocks since they share n_block
        dk_ptrs_atom = dk_base + n_offs[:, None] * S_stride_dk + d_offs[None, :] * D_stride_dk
        for di in range(BLOCK_D):
            tl.atomic_add(dk_ptrs_atom, dK_tile.to(K_ptr.dtype.element_ty)[:, di:di+1], mask=n_mask[:, None])

        dv_ptrs_atom = dv_base + n_offs[:, None] * S_stride_dv + d_offs[None, :] * D_stride_dv
        for di in range(BLOCK_D):
            tl.atomic_add(dv_ptrs_atom, dV_tile.to(V_ptr.dtype.element_ty)[:, di:di+1], mask=n_mask[:, None])


@triton.jit
def _dq_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dQ_ptr,
    S,
    BH_stride_q, S_stride_q, D_stride_q,
    BH_stride_k, S_stride_k, D_stride_k,
    BH_stride_v, S_stride_v, D_stride_v,
    BH_stride_o, S_stride_o, D_stride_o,
    BH_stride_do, S_stride_do, D_stride_do,
    BH_stride_l, S_stride_l,
    BH_stride_dq, S_stride_dq, D_stride_dq,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute dQ: each (bh, m_block) produces unique output rows."""
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
    d_offs = tl.arange(0, BLOCK_D)

    q_ptrs = q_base + m_offs[:, None] * S_stride_q + d_offs[None, :] * D_stride_q
    q = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0)

    do_ptrs = do_base + m_offs[:, None] * S_stride_do + d_offs[None, :] * D_stride_do
    do_tile = tl.load(do_ptrs, mask=m_mask[:, None], other=0.0)

    l_ptrs = l_base + m_offs * S_stride_l
    l_val = tl.load(l_ptrs, mask=m_mask, other=0.0).to(tl.float32)[:, None]

    scale = 1.0 / tl.sqrt(float(BLOCK_D))
    num_n_blocks = tl.cdiv(S, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    o_ptrs = o_base + m_offs[:, None] * S_stride_o + d_offs[None, :] * D_stride_o
    o_tile = tl.load(o_ptrs, mask=m_mask[:, None], other=0.0)
    d_row = tl.sum((do_tile * o_tile).to(tl.float32), axis=1)[:, None]

    for nb in range(num_n_blocks):
        n_offs = nb * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = n_offs < S

        k_ptrs = k_base + n_offs[:, None] * S_stride_k + d_offs[None, :] * D_stride_k
        k = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)

        v_ptrs = v_base + n_offs[:, None] * S_stride_v + d_offs[None, :] * D_stride_v
        v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)

        qk = tl.dot(q.to(tl.float32), k.to(tl.float32).T) * scale
        probs = tl.exp(qk - l_val)

        dp = tl.dot(do_tile.to(tl.float32), v.to(tl.float32).T)
        da = probs * (dp - d_row) * scale

        acc = tl.dot(da, k.to(tl.float32), acc)

    dq_out = acc.to(Q_ptr.dtype.element_ty)
    dq_ptrs = dq_base + m_offs[:, None] * S_stride_dq + d_offs[None, :] * D_stride_dq
    tl.store(dq_ptrs, dq_out, mask=m_mask[:, None])


@triton.jit
def _dkv_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dK_ptr, dV_ptr,
    S,
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
    BLOCK_D: tl.constexpr,
):
    """Compute dK, dV using atomic accumulations."""
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
    d_offs = tl.arange(0, BLOCK_D)

    k_ptrs = k_base + n_offs[:, None] * S_stride_k + d_offs[None, :] * D_stride_k
    k = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)

    v_ptrs = v_base + n_offs[:, None] * S_stride_v + d_offs[None, :] * D_stride_v
    v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)

    scale = 1.0 / tl.sqrt(float(BLOCK_D))
    num_m_blocks = tl.cdiv(S, BLOCK_M)
    acc_dk = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    acc_dv = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)

    for mb in range(num_m_blocks):
        m_offs = mb * BLOCK_M + tl.arange(0, BLOCK_M)
        m_mask = m_offs < S

        q_ptrs = q_base + m_offs[:, None] * S_stride_q + d_offs[None, :] * D_stride_q
        q = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0)

        l_ptrs = l_base + m_offs * S_stride_l
        l_row = tl.load(l_ptrs, mask=m_mask, other=0.0).to(tl.float32)[:, None]

        do_ptrs = do_base + m_offs[:, None] * S_stride_do + d_offs[None, :] * D_stride_do
        do_tile = tl.load(do_ptrs, mask=m_mask[:, None], other=0.0)

        o_ptrs = o_base + m_offs[:, None] * S_stride_o + d_offs[None, :] * D_stride_o
        o_tile = tl.load(o_ptrs, mask=m_mask[:, None], other=0.0)
        d_row = tl.sum((do_tile * o_tile).to(tl.float32), axis=1)[:, None]

        qkt = tl.dot(q.to(tl.float32), k.to(tl.float32).T) * scale
        probs = tl.exp(qkt - l_row)

        dp = tl.dot(do_tile.to(tl.float32), v.to(tl.float32).T)
        da = probs * (dp - d_row) * scale

        acc_dk = tl.dot(da.T, q.to(tl.float32), acc_dk)
        acc_dv = tl.dot(probs.T, do_tile.to(tl.float32), acc_dv)

    dk_out = acc_dk.to(K_ptr.dtype.element_ty)
    dk_ptrs = dk_base + n_offs[:, None] * S_stride_dk + d_offs[None, :] * D_stride_dk
    tl.store(dk_ptrs, dk_out, mask=n_mask[:, None])

    dv_out = acc_dv.to(V_ptr.dtype.element_ty)
    dv_ptrs = dv_base + n_offs[:, None] * S_stride_dv + d_offs[None, :] * D_stride_dv
    tl.store(dv_ptrs, dv_out, mask=n_mask[:, None])


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
    BLOCK_D = 128
    NUM_WARPS = 4
    NUM_STAGES = 3

    num_m_blocks = triton.cdiv(S, BLOCK_M)
    num_n_blocks = triton.cdiv(S, BLOCK_N)

    grid_dq = (BH, num_m_blocks)
    _dq_kernel[grid_dq](
        q_r, k_r, v_r, o_r, do_r, l_r, dq_r,
        S,
        bh_q, s_q, d_q,
        bh_k, s_k, d_k,
        bh_v, s_v, d_v,
        bh_o, s_o, d_o,
        bh_do, s_do, d_do,
        bh_l, s_l,
        bh_dq, s_dq, d_dq,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )

    grid_dkv = (BH, num_n_blocks)
    _dkv_kernel[grid_dkv](
        q_r, k_r, v_r, o_r, do_r, l_r, dk_r, dv_r,
        S,
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
        BLOCK_D=BLOCK_D,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )