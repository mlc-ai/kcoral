import torch
import triton
import triton.language as tl


@triton.jit
def _attention_kernel(
    q_ptr,
    stride_bq, stride_sq, stride_qd,
    k_ptr,
    stride_bk, stride_sk, stride_kd,
    v_ptr,
    stride_bv, stride_sv, stride_vd,
    o_ptr,
    stride_bo, stride_so, stride_od,
    lse_ptr,
    stride_bl, stride_sl,
    seq_len,
    scale,
    n_bh,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Multi-head attention forward pass.
    
    Grid: (n_batch_heads, n_query_tiles)
    Each program processes one query block of one (batch, head) pair.
    Inner loop sweeps over key/value tiles.
    
    Uses online softmax: maintain running (m_i, l_i, acc_o) across K iterations.
    """
    bh_idx = tl.program_id(0)
    tile_idx = tl.program_id(1)

    # Output row offsets [BLOCK_M]
    off_m = tile_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    off_d = tl.arange(0, BLOCK_D)
    off_k = tl.arange(0, BLOCK_N)

    # Base pointers for this (batch, head) slice
    q_base = q_ptr + bh_idx * stride_bq
    k_base = k_ptr + bh_idx * stride_bk
    v_base = v_ptr + bh_idx * stride_bv
    o_base = o_ptr + bh_idx * stride_bo
    lse_base = lse_ptr + bh_idx * stride_bl

    # Load Q tile [BLOCK_M, BLOCK_D]
    q_ptrs = q_base + off_m[:, None] * stride_sq + off_d[None, :] * stride_qd
    q_tile = tl.load(q_ptrs, mask=off_m[:, None] < seq_len, other=0.0)

    # Accumulators
    acc_o = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), -1e10, dtype=tl.float32)

    n_kv_tiles = tl.cdiv(seq_len, BLOCK_N)

    for kv in range(n_kv_tiles):
        kv_off = kv * BLOCK_N + off_k
        kv_mask = (kv_off < seq_len)[:, None]

        # Load K tile [BLOCK_N, BLOCK_D]
        k_ptrs = k_base + kv_off * stride_sk + off_d[None, :] * stride_kd
        k_tile = tl.load(k_ptrs, mask=kv_mask, other=0.0)

        # Load V tile [BLOCK_N, BLOCK_D]
        v_ptrs = v_base + kv_off * stride_sv + off_d[None, :] * stride_vd
        v_tile = tl.load(v_ptrs, mask=kv_mask, other=0.0)

        # Scaled QK^T  [BLOCK_M, BLOCK_D] x [BLOCK_D, BLOCK_N] -> [BLOCK_M, BLOCK_N]
        s = tl.dot(q_tile, tl.trans(k_tile)) * scale

        # Row max for numerical stability [BLOCK_M, 1]
        m_new = tl.max(s, axis=1, keep_dims=True)

        if kv == 0:
            # First block: no rescaling needed
            p = tl.exp(s - m_new)
            acc_o = tl.dot(p, v_tile)
            l_i = tl.sum(p, axis=1)
        else:
            # Rescale previous accumulators by exp(m_old - m_new)
            exp_diff = tl.exp(m_i.squeeze() - m_new.squeeze())
            acc_o = acc_o * exp_diff[:, None]
            l_i = l_i * exp_diff

            # New attention weights
            p = tl.exp(s - m_new)
            acc_o = tl.dot(p, v_tile, acc=acc_o)
            l_i = l_i + tl.sum(p, axis=1)

        m_i = m_new

    # Final normalization: O = acc_o / l_i
    o_val = acc_o / l_i[:, None]

    # Write output O [BLOCK_M, BLOCK_D]
    o_ptrs = o_base + off_m[:, None] * stride_so + off_d[None, :] * stride_od
    tl.store(o_ptrs, o_val.to(tl.bfloat16), mask=off_m[:, None] < seq_len)

    # Write LSE = m_i + log(l_i) [BLOCK_M]
    lse_val = m_i.squeeze() + tl.log(l_i)
    lse_ptrs = lse_base + off_m * stride_sl
    tl.store(lse_ptrs, lse_val, mask=off_m < seq_len)


def run(Q, K, V, O, LSE):
    """Destination-passing entry point for multi-head attention.
    
    Computes O = softmax(Q @ K^T / sqrt(D)) @ V and LSE = logsumexp(Q @ K^T / sqrt(D)).
    All inputs are [B, H, S, D] bf16; O is bf16, LSE is fp32 [B, H, S].
    """
    torch.cuda.set_device(Q.device)

    B, H, S, D = Q.shape
    n_bh = B * H

    # Scale factor for attention: 1/sqrt(D)
    scale = 1.0 / float(D ** 0.5)

    # Block sizes tuned for D=128 head dimension on Hopper
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = D  # Must be power of two for head dim 128

    # Stride helpers: combined (B,H) stride assumes B and H are adjacent dims
    def bh_stride(t):
        return t.stride(0) * H + t.stride(1)

    stride_bq, stride_sq, stride_qd = bh_stride(Q), Q.stride(2), Q.stride(3)
    stride_bk, stride_sk, stride_kd = bh_stride(K), K.stride(2), K.stride(3)
    stride_bv, stride_sv, stride_vd = bh_stride(V), V.stride(2), V.stride(3)
    stride_bo, stride_so, stride_od = bh_stride(O), O.stride(2), O.stride(3)
    stride_bl, stride_sl = bh_stride(LSE), LSE.stride(2)

    # Grid: (batch_heads, query_blocks)
    grid = (n_bh, triton.cdiv(S, BLOCK_M))

    _attention_kernel[grid](
        Q, stride_bq, stride_sq, stride_qd,
        K, stride_bk, stride_sk, stride_kd,
        V, stride_bv, stride_sv, stride_vd,
        O, stride_bo, stride_so, stride_od,
        LSE, stride_bl, stride_sl,
        S, scale, n_bh,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=3,
    )