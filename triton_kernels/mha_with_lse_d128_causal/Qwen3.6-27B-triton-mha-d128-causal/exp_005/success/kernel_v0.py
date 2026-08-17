import torch
import triton
import triton.language as tl


@triton.jit
def _causal_attn_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    lse_ptr,
    seq_len,
    d,
    sq,
    sk,
    sv,
    so,
    sd,
    sls,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Single-(batch,head) causal attention kernel on 2D [seq,d] tensors."""

    pid_m = tl.program_id(0)

    # ---- offsets within this query tile ----
    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    off_d = tl.arange(0, BLOCK_D)                     # [BLOCK_D]

    row_mask  = (off_m < seq_len)[:, None]            # [BLOCK_M, 1]
    col_mask  = (off_d < d)[None, :]                   # [1, BLOCK_D]
    row_mask_1d = off_m < seq_len                      # [BLOCK_M]

    # ---- accumulator state (fp32) ----
    acc   = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    m_i   = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)
    l_i   = tl.full((BLOCK_M,), 1.0,                  dtype=tl.float32)

    inv_scale = 1.0 / tl.sqrt(d.to(tl.float32))
    n_kv_tiles = tl.cdiv(seq_len, BLOCK_N)

    # ---- hoist Q load outside KV loop ----
    q_ptrs = q_ptr + off_m[:, None] * sq + off_d[None, :] * sd
    q_tile = tl.load(q_ptrs, mask=row_mask & col_mask, other=0.0)

    # ================================================================
    # Main loop – iterate over key/value block tiles
    # ================================================================
    for start_n in range(n_kv_tiles):
        off_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)   # [BLOCK_N]

        # --- load K [BLOCK_N, BLOCK_D] -----------------------------
        col_mask_n = (off_n < seq_len)[:, None]
        k_ptrs = k_ptr + off_n[:, None] * sk + off_d[None, :] * sd
        k_tile = tl.load(k_ptrs, mask=col_mask_n & col_mask, other=0.0)

        # --- Q · Kᵀ -> [BLOCK_M, BLOCK_N] -------------------------
        scores = tl.dot(q_tile, k_tile.T) * inv_scale

        # --- causal mask -------------------------------------------
        causal_mask = off_m[:, None] >= off_n[None, :]        # [BM,BN]
        scores = tl.where(causal_mask, scores, float('-inf'))

        # --- online softmax update ---------------------------------
        cur_max = tl.max(scores, axis=1)                      # [BM]
        new_max = tl.maximum(m_i, cur_max)                    # [BM]

        alpha = tl.exp(m_i - new_max)
        p_exp = tl.exp(scores - new_max[:, None])             # [BM,BN]
        p_sum = tl.sum(p_exp, axis=1)                         # [BM]

        l_i = l_i * alpha + p_sum
        m_i = new_max

        # --- load V [BLOCK_N, BLOCK_D] ----------------------------
        v_ptrs = v_ptr + off_n[:, None] * sv + off_d[None, :] * sd
        v_tile = tl.load(v_ptrs, mask=col_mask_n & col_mask, other=0.0)

        # --- accumulate: previous * alpha + p_exp · V -------------
        acc = acc * alpha[:, None] + tl.dot(p_exp, v_tile.to(tl.float32))

    # ---- normalise and store O (bf16) ----------------------------
    denom = l_i[:, None]                                      # [BM,1]
    o_val = acc / denom                                        # fp32

    o_ptrs = o_ptr + off_m[:, None] * so + off_d[None, :] * sd
    tl.store(o_ptrs, o_val.to(tl.bfloat16), mask=row_mask & col_mask)

    # ---- store LSE (fp32) ----------------------------------------
    lse_val = m_i + tl.log(l_i)                               # [BM] fp32
    lse_ptrs = lse_ptr + off_m * sls
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

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = 128
    num_q_tiles = triton.cdiv(S, BLOCK_M)

    kernel = lambda META: (triton.cdiv(S, META["BLOCK_M"]),)

    for b in range(B):
        for h in range(H):
            q_h = Q[b, h]                       # [S, D] bf16 contiguous
            k_h = K[b, h]                       # [S, D] bf16 contiguous
            v_h = V[b, h]                       # [S, D] bf16 contiguous
            o_h = O[b, h]                       # [S, D] bf16 contiguous
            lse_bh = LSE[b, h]                  # [S]   fp32 contiguous

            _causal_attn_kernel[kernel](
                q_h, k_h, v_h, o_h, lse_bh,
                seq_len=S,
                d=D,
                sq=q_h.stride(0),
                sk=k_h.stride(0),
                sv=v_h.stride(0),
                so=o_h.stride(0),
                sd=q_h.stride(1),
                sls=lse_bh.stride(0),
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                BLOCK_D=BLOCK_D,
                num_warps=4,
                num_stages=2,
            )