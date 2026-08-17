import torch
import triton
import triton.language as tl


@triton.jit
def _causal_attn_kernel(
    Q, K, V, O, LSE,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lsb, stride_lsh, stride_ls,
    SEQ_LEN, D, NUM_HEADS,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Full-grid causal attention kernel. Grid: (batch×head, query-tile)."""

    pid_bh = tl.program_id(0)
    pid_m  = tl.program_id(1)

    batch = pid_bh // NUM_HEADS
    head  = pid_bh % NUM_HEADS

    # ---- coordinate arrays ----
    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    off_d = tl.arange(0, BLOCK_D)                     # [BLOCK_D]

    row_mask = (off_m < SEQ_LEN)[:, None]             # [BM, 1]
    col_mask = (off_d < D)[None, :]                    # [1, BD]
    row_mask_1d = off_m < SEQ_LEN                      # [BM]

    # ---- per-(batch,head) base offsets ----
    base_q  = batch * stride_qb + head * stride_qh
    base_k  = batch * stride_kb + head * stride_kh
    base_v  = batch * stride_vb + head * stride_vh
    base_o  = batch * stride_ob + head * stride_oh
    base_lse = batch * stride_lsb + head * stride_lsh

    # ---- accumulator state (fp32) ----
    acc   = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    m_i   = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)
    l_i   = tl.full((BLOCK_M,), 1.0,                  dtype=tl.float32)

    inv_scale = 1.0 / tl.sqrt(D.to(tl.float32))
    n_kv_tiles = tl.cdiv(SEQ_LEN, BLOCK_N)

    # ---- hoist Q load outside KV loop ----
    q_ptrs = base_q + off_m[:, None] * stride_qs + off_d[None, :] * stride_qd
    q_tile = tl.load(q_ptrs, mask=row_mask & col_mask, other=0.0)

    # ================================================================
    # Main loop – iterate over key/value block tiles
    # ================================================================
    for start_n in range(n_kv_tiles):
        off_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

        # --- load K [BLOCK_N, BLOCK_D] -----------------------------
        col_mask_n = (off_n < SEQ_LEN)[:, None]
        k_ptrs = base_k + off_n[:, None] * stride_ks + off_d[None, :] * stride_kd
        k_tile = tl.load(k_ptrs, mask=col_mask_n & col_mask, other=0.0)

        # --- Q · Kᵀ -> [BLOCK_M, BLOCK_N] -------------------------
        scores = tl.dot(q_tile, k_tile.T) * inv_scale

        # --- causal mask -------------------------------------------
        causal_mask = off_m[:, None] >= off_n[None, :]
        scores = tl.where(causal_mask, scores, float('-inf'))

        # --- online softmax update ---------------------------------
        cur_max = tl.max(scores, axis=1)
        new_max = tl.maximum(m_i, cur_max)

        alpha = tl.exp(m_i - new_max)
        p_exp = tl.exp(scores - new_max[:, None])
        p_sum = tl.sum(p_exp, axis=1)

        l_i = l_i * alpha + p_sum
        m_i = new_max

        # --- load V [BLOCK_N, BLOCK_D] ----------------------------
        v_ptrs = base_v + off_n[:, None] * stride_vs + off_d[None, :] * stride_vd
        v_tile = tl.load(v_ptrs, mask=col_mask_n & col_mask, other=0.0)

        # --- accumulate: previous * alpha + p_exp · V -------------
        acc = acc * alpha[:, None] + tl.dot(p_exp, v_tile.to(tl.float32))

    # ---- normalise and store O (bf16) ----------------------------
    denom = l_i[:, None]
    o_val = acc / denom

    o_ptrs = base_o + off_m[:, None] * stride_os + off_d[None, :] * stride_od
    tl.store(o_ptrs, o_val.to(tl.bfloat16), mask=row_mask & col_mask)

    # ---- store LSE (fp32) ----------------------------------------
    lse_val = m_i + tl.log(l_i)
    lse_ptrs = base_lse + off_m * stride_ls
    tl.store(lse_ptrs, lse_val, mask=row_mask_1d)


def run(Q, K, V, O, LSE):
    """
    Destination-passing entry point for causal MHA forward.

    Inputs (definition order):  Q, K, V  – bf16 [B,H,S,D]
    Outputs (definition order): O, LSE   – bf16 [B,H,S,D], fp32 [B,H,S]
    """
    torch.cuda.set_device(Q.device)

    B, H, S, D = Q.shape
    assert K.shape == (B, H, S, D)
    assert V.shape == (B, H, S, D)
    assert O.shape == (B, H, S, D)
    assert LSE.shape == (B, H, S)

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_D = 128

    num_bh = B * H
    num_q_tiles = triton.cdiv(S, BLOCK_M)

    grid = (num_bh, num_q_tiles)

    _causal_attn_kernel[grid](
        Q, K, V, O, LSE,
        stride_qb=Q.stride(0), stride_qh=Q.stride(1),
        stride_qs=Q.stride(2), stride_qd=Q.stride(3),
        stride_kb=K.stride(0), stride_kh=K.stride(1),
        stride_ks=K.stride(2), stride_kd=K.stride(3),
        stride_vb=V.stride(0), stride_vh=V.stride(1),
        stride_vs=V.stride(2), stride_vd=V.stride(3),
        stride_ob=O.stride(0), stride_oh=O.stride(1),
        stride_os=O.stride(2), stride_od=O.stride(3),
        stride_lsb=LSE.stride(0), stride_lsh=LSE.stride(1),
        stride_ls=LSE.stride(2),
        SEQ_LEN=S,
        D=D,
        NUM_HEADS=H,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        num_warps=8,
        num_stages=3,
    )