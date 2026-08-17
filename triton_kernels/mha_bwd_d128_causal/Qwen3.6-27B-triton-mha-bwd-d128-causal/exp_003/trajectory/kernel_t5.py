import torch
import triton
import triton.language as tl


@triton.jit
def _bwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
    dQ_out_ptr, dK_out_ptr, dV_out_ptr,
    B, H, S, D,
    HEAD_STRIDE_Q, BATCH_STRIDE_Q,
    HEAD_STRIDE_L, BATCH_STRIDE_L,
    inv_sqrt_d,
    NUM_SMS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """Fused causal MHA backward: computes dQ, dK, dV in one kernel.
    
    Each program processes a single (m_tile, n_tile) across ALL batch-head pairs
    using an outer loop. This reduces grid size from ~400K to just S/BLOCK_M * S/BLOCK_N = 64 tiles.
    
    Thread block layout:
      - axis 0: batch-head group identifier
      - axis 1: m-tile index
      - axis 2: n-tile index
    
    Actually we collapse bh into a persistent inner loop for maximum throughput.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m_base = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n_base = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m_base < S
    mask_n = offs_n_base < S

    offs_d = tl.arange(0, D)

    # Total number of batch-head combos
    num_bh = B * H

    # Iterate over all batch-head pairs persistently
    for bih in range(num_bh):
        b = bih // H
        h = bih % H

        # Compute base offsets for this (b,h) pair
        q_base = Q_ptr + b * BATCH_STRIDE_Q + h * HEAD_STRIDE_Q
        k_base = K_ptr + b * BATCH_STRIDE_Q + h * HEAD_STRIDE_Q
        v_base = V_ptr + b * BATCH_STRIDE_Q + h * HEAD_STRIDE_Q
        o_base = O_ptr + b * BATCH_STRIDE_Q + h * HEAD_STRIDE_Q
        do_base = dO_ptr + b * BATCH_STRIDE_Q + h * HEAD_STRIDE_Q
        l_base = L_ptr + b * BATCH_STRIDE_L + h * HEAD_STRIDE_L
        dq_base = dQ_out_ptr + b * BATCH_STRIDE_Q + h * HEAD_STRIDE_Q
        dk_base = dK_out_ptr + b * BATCH_STRIDE_Q + h * HEAD_STRIDE_Q
        dv_base = dV_out_ptr + b * BATCH_STRIDE_Q + h * HEAD_STRIDE_Q

        m_idx = offs_m_base[:, None]
        n_idx = offs_n_base[:, None]
        d_idx = offs_d[None, :]

        # Load logsumexp for query rows
        lse = tl.load(l_base + offs_m_base, mask=mask_m, other=0.0)

        # Preload Q, dO, O [BLOCK_M, D]
        q_ptrs = q_base + offs_m_base[:, None] * D + offs_d[None, :]
        q_block = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)

        do_ptrs = do_base + offs_m_base[:, None] * D + offs_d[None, :]
        do_block = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)

        o_ptrs = o_base + offs_m_base[:, None] * D + offs_d[None, :]
        o_block = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)

        # pD[m] = dO[m,:] · O[m,:]
        pD = tl.sum(do_block * o_block, axis=1)

        # --- Phase 1: accumulate dQ contributions ---
        acc_t1_q = tl.zeros((BLOCK_M, D), dtype=tl.float32)
        acc_t3_q = tl.zeros((BLOCK_M, D), dtype=tl.float32)

        num_n_tiles = tl.cdiv(S, BLOCK_N)
        for pi in range(num_n_tiles):
            cur_offs_n = pi * BLOCK_N + tl.arange(0, BLOCK_N)
            cur_mask_n = cur_offs_n < S
            cur_n_idx = cur_offs_n[:, None]

            k_ptrs = k_base + cur_offs_n[:, None] * D + offs_d[None, :]
            k_block = tl.load(k_ptrs, mask=cur_mask_n[:, None], other=0.0).to(tl.float32)

            v_ptrs = v_base + cur_offs_n[:, None] * D + offs_d[None, :]
            v_block = tl.load(v_ptrs, mask=cur_mask_n[:, None], other=0.0).to(tl.float32)

            attn = tl.dot(q_block, k_block.T) * inv_sqrt_d

            causal = offs_m_base[:, None] >= cur_offs_n[None, :]
            valid = causal & mask_m[:, None] & cur_mask_n[None, :]

            P = tl.where(valid, tl.exp(attn - lse[:, None]), 0.0)

            dP_bar = tl.dot(do_block, v_block.T)

            PdP = P * dP_bar
            acc_t1_q = tl.dot(PdP, k_block, acc_t1_q)
            acc_t3_q = tl.dot(P, k_block, acc_t3_q)

        dQ_val = (acc_t1_q - pD[:, None] * acc_t3_q) * inv_sqrt_d
        dq_ptrs = dq_base + offs_m_base[:, None] * D + offs_d[None, :]
        tl.store(dq_ptrs, dQ_val.to(tl.bfloat16), mask=mask_m[:, None])

        # --- Phase 2: accumulate dK and dV contributions ---
        # Preload K, V [BLOCK_N, D]
        k_ptrs_kv = k_base + offs_n_base[:, None] * D + offs_d[None, :]
        k_block_kv = tl.load(k_ptrs_kv, mask=mask_n[:, None], other=0.0).to(tl.float32)

        v_ptrs_kv = v_base + offs_n_base[:, None] * D + offs_d[None, :]
        v_block_kv = tl.load(v_ptrs_kv, mask=mask_n[:, None], other=0.0).to(tl.float32)

        acc_dk = tl.zeros((BLOCK_N, D), dtype=tl.float32)
        acc_dv = tl.zeros((BLOCK_N, D), dtype=tl.float32)

        num_m_tiles = tl.cdiv(S, BLOCK_M)
        for pi in range(num_m_tiles):
            cur_offs_m = pi * BLOCK_M + tl.arange(0, BLOCK_M)
            cur_mask_m = cur_offs_m < S
            cur_m_idx = cur_offs_m[:, None]

            q_ptrs = q_base + cur_offs_m[:, None] * D + offs_d[None, :]
            q_block_qkv = tl.load(q_ptrs, mask=cur_mask_m[:, None], other=0.0).to(tl.float32)

            do_ptrs = do_base + cur_offs_m[:, None] * D + offs_d[None, :]
            do_block_qkv = tl.load(do_ptrs, mask=cur_mask_m[:, None], other=0.0).to(tl.float32)

            o_ptrs = o_base + cur_offs_m[:, None] * D + offs_d[None, :]
            o_block_qkv = tl.load(o_ptrs, mask=cur_mask_m[:, None], other=0.0).to(tl.float32)

            lse_cur = tl.load(l_base + cur_offs_m, mask=cur_mask_m, other=0.0)

            pD_cur = tl.sum(do_block_qkv * o_block_qkv, axis=1)

            attn = tl.dot(q_block_qkv, k_block_kv.T) * inv_sqrt_d

            causal = cur_offs_m[:, None] >= offs_n_base[None, :]
            valid = causal & cur_mask_m[:, None] & mask_n[None, :]

            P = tl.where(valid, tl.exp(attn - lse_cur[:, None]), 0.0)

            dP_bar = tl.dot(do_block_qkv, v_block_kv.T)

            dscore = P * (dP_bar - pD_cur[:, None])

            acc_dk = tl.dot(dscore.T, q_block_qkv, acc_dk)
            acc_dv = tl.dot(P.T, do_block_qkv, acc_dv)

        dk_ptrs = dk_base + offs_n_base[:, None] * D + offs_d[None, :]
        tl.store(dk_ptrs, (acc_dk * inv_sqrt_d).to(tl.bfloat16), mask=mask_n[:, None])

        dv_ptrs = dv_base + offs_n_base[:, None] * D + offs_d[None, :]
        tl.store(dv_ptrs, acc_dv.to(tl.bfloat16), mask=mask_n[:, None])


def run(Q, K, V, O, dO, L, dQ_out, dK_out, dV_out):
    """Causal multi-head attention backward."""
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape

    import math
    inv_sqrt_d = 1.0 / math.sqrt(D)

    BLOCK_M = 64
    BLOCK_N = 64

    num_m_tiles = triton.cdiv(S, BLOCK_M)
    num_n_tiles = triton.cdiv(S, BLOCK_N)

    grid = (num_m_tiles, num_n_tiles)

    BATCH_STRIDE_Q = H * S * D
    HEAD_STRIDE_Q = S * D
    BATCH_STRIDE_L = H * S
    HEAD_STRIDE_L = S

    _bwd_kernel[grid](
        Q, K, V, O, dO, L,
        dQ_out, dK_out, dV_out,
        B, H, S, D,
        HEAD_STRIDE_Q, BATCH_STRIDE_Q,
        HEAD_STRIDE_L, BATCH_STRIDE_L,
        inv_sqrt_d,
        NUM_SMS=132,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=8, num_stages=3,
    )