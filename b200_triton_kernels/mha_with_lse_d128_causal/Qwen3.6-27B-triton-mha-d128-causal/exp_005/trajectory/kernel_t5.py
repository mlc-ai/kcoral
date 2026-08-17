import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _causal_attn_kernel(
    q_desc, k_desc, v_desc, o_desc,
    lse_ptr, seq_len, d, stride_ls,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Single-(batch,head) causal attention kernel using tensor descriptors."""

    pid_m = tl.program_id(0)

    # ---- coordinate offsets ----
    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    off_d = tl.arange(0, BLOCK_D)                     # [BLOCK_D]

    row_mask_1d = off_m < seq_len

    # ---- accumulator state (fp32) ----
    acc   = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    m_i   = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)
    l_i   = tl.full((BLOCK_M,), 1.0,                  dtype=tl.float32)

    inv_scale = 1.0 / tl.sqrt(d.to(tl.float32))
    n_kv_tiles = tl.cdiv(seq_len, BLOCK_N)

    # ---- hoist Q load outside KV loop ----
    q_tile = q_desc.load([pid_m * BLOCK_M, 0])            # [BM, BD]

    # ================================================================
    # Main loop – iterate over key/value block tiles
    # ================================================================
    for start_n in range(n_kv_tiles):
        off_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

        # --- load K [BLOCK_N, BLOCK_D] -----------------------------
        k_tile = k_desc.load([start_n * BLOCK_N, 0])       # [BN, BD]

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
        v_tile = v_desc.load([start_n * BLOCK_N, 0])       # [BN, BD]

        # --- accumulate: previous * alpha + p_exp · V -------------
        acc = acc * alpha[:, None] + tl.dot(p_exp, v_tile.to(tl.float32))

    # ---- normalise and store O (bf16) ----------------------------
    denom = l_i[:, None]
    o_val = acc / denom                                     # fp32

    # Store O via descriptor
    o_desc.store([pid_m * BLOCK_M, 0], o_val.to(tl.bfloat16))

    # ---- store LSE (fp32) ----------------------------------------
    lse_val = m_i + tl.log(l_i)                             # [BM] fp32
    lse_ptrs = lse_ptr + off_m * stride_ls
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

    # Pre-build tensor descriptors per head tile shape
    q_td_base = TensorDescriptor.from_tensor(Q, [BLOCK_M, BLOCK_D])
    k_td_base = TensorDescriptor.from_tensor(K, [BLOCK_N, BLOCK_D])
    v_td_base = TensorDescriptor.from_tensor(V, [BLOCK_N, BLOCK_D])
    o_td_base = TensorDescriptor.from_tensor(O, [BLOCK_M, BLOCK_D])

    def _mk_heads():
        """Yield (q_h, k_h, v_h, o_h, lse_bh) per (b,h) with updated descriptors."""
        for b in range(B):
            for h in range(H):
                yield (Q[b, h], K[b, h], V[b, h], O[b, h], LSE[b, h])

    # Warm up with compilation
    warm_q, warm_k, warm_v, warm_o, warm_lse = next(_mk_heads())
    
    # Build descriptors for warm
    q_d = TensorDescriptor.from_tensor(warm_q, [BLOCK_M, BLOCK_D])
    k_d = TensorDescriptor.from_tensor(warm_k, [BLOCK_N, BLOCK_D])
    v_d = TensorDescriptor.from_tensor(warm_v, [BLOCK_N, BLOCK_D])
    o_d = TensorDescriptor.from_tensor(warm_o, [BLOCK_M, BLOCK_D])
    
    num_q_tiles = triton.cdiv(S, BLOCK_M)
    grid = (num_q_tiles,)

    _causal_attn_kernel[grid](
        q_d, k_d, v_d, o_d,
        warm_lse,
        seq_len=S,
        d=D,
        stride_ls=warm_lse.stride(0),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        num_warps=8,
        num_stages=4,
    )

    # Now process all heads with fresh generators
    def _all_heads():
        for b in range(B):
            for h in range(H):
                q_h = Q[b, h]
                k_h = K[b, h]
                v_h = V[b, h]
                o_h = O[b, h]
                lse_bh = LSE[b, h]
                
                q_d = TensorDescriptor.from_tensor(q_h, [BLOCK_M, BLOCK_D])
                k_d = TensorDescriptor.from_tensor(k_h, [BLOCK_N, BLOCK_D])
                v_d = TensorDescriptor.from_tensor(v_h, [BLOCK_N, BLOCK_D])
                o_d = TensorDescriptor.from_tensor(o_h, [BLOCK_M, BLOCK_D])

                _causal_attn_kernel[grid](
                    q_d, k_d, v_d, o_d,
                    lse_bh,
                    seq_len=S,
                    d=D,
                    stride_ls=lse_bh.stride(0),
                    BLOCK_M=BLOCK_M,
                    BLOCK_N=BLOCK_N,
                    BLOCK_D=BLOCK_D,
                    num_warps=8,
                    num_stages=4,
                )

    # Re-run: reset outputs to zero first since we warmed up the first head
    # Actually, the warm-up wrote correct results too. We just need all other heads.
    # Re-do from scratch cleanly.
    # Reset outputs
    O.zero_()
    LSE.zero_()

    processed = False
    for q_h, k_h, v_h, o_h, lse_bh in _all_heads():
        q_d = TensorDescriptor.from_tensor(q_h, [BLOCK_M, BLOCK_D])
        k_d = TensorDescriptor.from_tensor(k_h, [BLOCK_N, BLOCK_D])
        v_d = TensorDescriptor.from_tensor(v_h, [BLOCK_N, BLOCK_D])
        o_d = TensorDescriptor.from_tensor(o_h, [BLOCK_M, BLOCK_D])

        _causal_attn_kernel[grid](
            q_d, k_d, v_d, o_d,
            lse_bh,
            seq_len=S,
            d=D,
            stride_ls=lse_bh.stride(0),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_D=BLOCK_D,
            num_warps=8,
            num_stages=4,
        )