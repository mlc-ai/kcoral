import torch
import triton
import triton.language as tl


@triton.jit
def _causal_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    lse_ptr,
    seq_len,
    D,
    stride_q_seq,
    stride_k_seq,
    stride_v_seq,
    stride_o_seq,
    stride_lse_seq,
    HEAD_STRIDE_ELEMS,
    NEG_INF_SENTINEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Single-(batch,head) causal attention kernel.
    Grid axis 0 iterates over query-block tiles.
    """
    # ---- program coordinates ----
    q_block_id = tl.program_id(0)
    q_start = q_block_id * BLOCK_M

    q_offsets = q_start + tl.arange(0, BLOCK_M)       # [BLOCK_M]
    d_offsets = tl.arange(0, BLOCK_D)                  # [BLOCK_D]

    row_in_bounds  = q_offsets[:, None] < seq_len      # [BLOCK_M, 1]
    col_in_bounds  = d_offsets[None, :] < D             # [1, BLOCK_D]
    row_in_bounds_1 = q_offsets < seq_len              # [BLOCK_M]   for LSE store

    # ---- initialise accumulators ----
    acc_o = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    m_i   = tl.full((BLOCK_M,), NEG_INF_SENTINEL, dtype=tl.float32)
    l_i   = tl.full((BLOCK_M,), 1.0,                    dtype=tl.float32)

    # ---- per-(B,H) base-offset (in elements from the whole-tensor base) ----
    bh_offset = HEAD_STRIDE_ELEMS                       # batch is always 0

    inv_scale = 1.0 / tl.sqrt(D.to(tl.float32))

    num_kv_tiles = tl.cdiv(seq_len, BLOCK_N)

    # ============================================================
    # main loop – iterate over key/value block tiles
    # ============================================================
    for kv_idx in range(num_kv_tiles):
        kv_start  = kv_idx * BLOCK_N
        kv_offsets = kv_start + tl.arange(0, BLOCK_N)   # [BLOCK_N]

        # --- load Q tile  [BLOCK_M, BLOCK_D] ---------------------
        q_tile = tl.load(
            q_ptr + bh_offset + q_offsets[:, None] * stride_q_seq + d_offsets[None, :],
            mask=row_in_bounds & col_in_bounds,
            other=0.0,
        )

        # --- load K tile  [BLOCK_N, BLOCK_D] ---------------------
        k_tile = tl.load(
            k_ptr + bh_offset + kv_offsets[:, None] * stride_k_seq + d_offsets[None, :],
            mask=(kv_offsets[:, None] < seq_len) & col_in_bounds,
            other=0.0,
        )

        # --- Q · Kᵀ  →  [BLOCK_M, BLOCK_N] ----------------------
        scores = tl.dot(q_tile, k_tile.T) * inv_scale

        # --- causal mask -----------------------------------------
        causal = q_offsets[:, None] >= kv_offsets[None, :]      # [BLOCK_M, BLOCK_N]
        scores = tl.where(causal, scores, NEG_INF_SENTINEL)

        # --- numerically-stable online softmax --------------------
        cur_max = tl.max(scores, axis=1)                        # [BLOCK_M]
        new_max = tl.maximum(m_i, cur_max)                      # [BLOCK_M]

        alpha   = tl.exp(m_i - new_max)                         # decay-factor
        p_exp   = tl.exp(scores - new_max[:, None])             # [BLOCK_M, BLOCK_N]
        p_sum   = tl.sum(p_exp, axis=1)                         # [BLOCK_M]

        l_i     = l_i * alpha + p_sum
        m_i     = new_max

        # --- load V tile  [BLOCK_N, BLOCK_D] --------------------
        v_tile = tl.load(
            v_ptr + bh_offset + kv_offsets[:, None] * stride_v_seq + d_offsets[None, :],
            mask=(kv_offsets[:, None] < seq_len) & col_in_bounds,
            other=0.0,
        )

        # --- accumulate weighted-V --------------------------------
        acc_o = acc_o * alpha[:, None] + tl.dot(p_exp, v_tile)

    # ---- normalise & write-back ----------------------------------
    # divide by final row-sums  (guard against div-by-zero for empty windows)
    denom = l_i[..., None] + 1e-30                                # [BLOCK_M, 1]
    o_val = acc_o / denom                                        # [BLOCK_M, BLOCK_D] fp32

    tl.store(
        o_ptr + bh_offset + q_offsets[:, None] * stride_o_seq + d_offsets[None, :],
        o_val,
        mask=row_in_bounds & col_in_bounds,
    )

    lse_val = m_i + tl.log(l_i + 1e-30)                          # [BLOCK_M] fp32
    tl.store(
        lse_ptr + bh_offset + q_offsets * stride_lse_seq,
        lse_val,
        mask=row_in_bounds_1,
    )


def run(Q, K, V, O, LSE):
    """
    Destination-passing entry-point for causal MHA forward.

    Parameters (in definition order):
        Q, K, V  – bf16 tensors [B, H, S, D]
        O        – preallocated bf16 tensor [B, H, S, D]
        LSE      – preallocated f32  tensor [B, H, S]
    """
    torch.cuda.set_device(Q.device)

    B, H, S, D = Q.shape
    assert K.shape == (B, H, S, D)
    assert V.shape == (B, H, S, D)
    assert O.shape == (B, H, S, D)
    assert LSE.shape == (B, H, S)

    num_q_tiles = triton.cdiv(S, 64)

    for b in range(B):
        for h in range(H):
            # slice helpers – isolate one (batch, head) view
            q_h = Q[b, h]      # [S, D]
            k_h = K[b, h]
            v_h = V[b, h]
            o_h = O[b, h]
            lse_bh = LSE[b, h] # [S]

            grid = (num_q_tiles,)

            _causal_attention_kernel[grid](
                q_h,
                k_h,
                v_h,
                o_h,
                lse_bh,
                seq_len=S,
                D=D,
                stride_q_seq=q_h.stride(0),
                stride_k_seq=k_h.stride(0),
                stride_v_seq=v_h.stride(0),
                stride_o_seq=o_h.stride(0),
                stride_lse_seq=lse_bh.stride(0),
                HEAD_STRIDE_ELEMS=0,          # already sliced – no extra offset
                BLOCK_M=64,
                BLOCK_N=64,
                BLOCK_D=128,
                NEG_INF_SENTINEL=-1e10,
                num_warps=4,
                num_stages=2,
            )