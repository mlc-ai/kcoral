import torch
import triton
import triton.language as tl


@triton.jit
def _causal_attn_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr,
    seq_len, d,
    sq, sk, sv, so, sd, sls,
    N_BATCH_HEAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    All-(batch,head) causal attention kernel.
    Grid: (num_batch_heads, num_query_tiles)
    """
    pid_bh = tl.program_id(0)       # batch-head index
    pid_m  = tl.program_id(1)       # query tile index

    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    off_d = tl.arange(0, BLOCK_D)                     # [BLOCK_D]

    row_mask = (off_m < seq_len)[:, None]             # [BM,1]
    col_mask = (off_d < d)[None, :]                    # [1,BD]
    row_mask_1d = off_m < seq_len                      # [BM]

    # ---- base offset for this (batch,head) ----
    bh_base = pid_bh * sq                              # because sq == S*D for flattened view

    # ---- accumulators (fp32) ----
    acc   = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    m_i   = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)
    l_i   = tl.full((BLOCK_M,), 1.0,                  dtype=tl.float32)

    inv_scale = 1.0 / tl.sqrt(d.to(tl.float32))
    n_kv_tiles = tl.cdiv(seq_len, BLOCK_N)

    # ---- hoist Q load ----
    q_ptrs = q_ptr + bh_base + off_m[:, None] * sq + off_d[None, :] * sd
    q_tile = tl.load(q_ptrs, mask=row_mask & col_mask, other=0.0)

    # ================================================================
    # KV main loop
    # ================================================================
    for start_n in range(n_kv_tiles):
        off_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
        col_mask_n = (off_n < seq_len)[:, None]

        # --- load K ------------------------------------------------
        k_ptrs = k_ptr + bh_base + off_n[:, None] * sk + off_d[None, :] * sd
        k_tile = tl.load(k_ptrs, mask=col_mask_n & col_mask, other=0.0)

        # --- Q · Kᵀ -> [BLOCK_M, BLOCK_N] -------------------------
        scores = tl.dot(q_tile, k_tile.T) * inv_scale

        # --- causal mask -------------------------------------------
        causal_mask = off_m[:, None] >= off_n[None, :]
        scores = tl.where(causal_mask, scores, float('-inf'))

        # --- online softmax ----------------------------------------
        cur_max = tl.max(scores, axis=1)
        new_max = tl.maximum(m_i, cur_max)

        alpha = tl.exp(m_i - new_max)
        p_exp = tl.exp(scores - new_max[:, None])
        p_sum = tl.sum(p_exp, axis=1)

        l_i = l_i * alpha + p_sum
        m_i = new_max

        # --- load V ------------------------------------------------
        v_ptrs = v_ptr + bh_base + off_n[:, None] * sv + off_d[None, :] * sd
        v_tile = tl.load(v_ptrs, mask=col_mask_n & col_mask, other=0.0)

        # --- accumulate O ------------------------------------------
        acc = acc * alpha[:, None] + tl.dot(p_exp, v_tile.to(tl.float32))

    # ---- normalise & store O (bf16) ------------------------------
    denom = l_i[:, None]
    o_val = acc / denom

    o_ptrs = o_ptr + bh_base + off_m[:, None] * so + off_d[None, :] * sd
    tl.store(o_ptrs, o_val.to(tl.bfloat16), mask=row_mask & col_mask)

    # ---- store LSE (fp32) ----------------------------------------
    lse_val = m_i + tl.log(l_i)
    lse_ptrs = lse_ptr + pid_bh * sls + off_m
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

    BH = B * H

    # Reshape to [BH, S, D] for safe pointer math (zero-copy view when contiguous)
    Q_bh = Q.view(BH, S, D)
    K_bh = K.view(BH, S, D)
    V_bh = V.view(BH, S, D)
    O_bh = O.view(BH, S, D)
    LSE_bh = LSE.view(BH, S)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = 128

    num_q_tiles = triton.cdiv(S, BLOCK_M)
    grid = (BH, num_q_tiles)

    _causal_attn_kernel[grid](
        Q_bh, K_bh, V_bh, O_bh, LSE_bh,
        seq_len=S,
        d=D,
        sq=Q_bh.stride(0),
        sk=K_bh.stride(0),
        sv=V_bh.stride(0),
        so=O_bh.stride(0),
        sd=Q_bh.stride(1),
        sls=LSE_bh.stride(0),
        N_BATCH_HEAD=BH,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        num_warps=8,
        num_stages=3,
    )