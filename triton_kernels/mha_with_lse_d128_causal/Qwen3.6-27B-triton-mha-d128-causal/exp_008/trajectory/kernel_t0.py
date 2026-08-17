import torch
import triton
import triton.language as tl


@triton.jit
def _flash_attn_fwd_kernel(
    Q,
    K,
    V,
    O,
    LSE,
    seqlen,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lse_b, stride_lse_h, stride_lse_s,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """FlashAttention-3 style causal attention kernel."""

    pid = tl.program_id(0)
    total_bh = tl.num_programs(0)
    pid_bm = pid // total_bh
    pid_bh = pid % total_bh
    num_heads = (total_bh - pid_bm) if pid_bm < total_bh else 0

    # Derive batch/head from flattened BH index
    h = pid_bh
    # Need to know num_heads - encode it differently
    # Use a different approach: B and H as constexpr or derived

    # Actually, let's use the grid layout: (tiles_m, B, H)
    # Simpler: flatten and decode
    # Since B*H is known, we can compute:
    # total_tiles_m per (B,H) is implicit
    
    # Let me use a cleaner 3D grid approach instead
    pass


@triton.jit
def _flash_attn_fwd_kernel(
    Q,
    K,
    V,
    O,
    LSE,
    num_heads,
    seqlen,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lse_b, stride_lse_h, stride_lse_s,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """FlashAttention-3 style causal attention forward kernel.
    
    Grid: (num_tiles_m, num_heads, batch_size)
    Each CTA processes one (b, h) pair over a tile of M query positions.
    Iterates over N blocks for keys/values.
    Uses online softmax with FP32 accumulation.
    Applies causal mask: query pos can only attend to key pos <= query pos.
    """

    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)

    inv_sqrt_d = 1.0 / tl.sqrt(128.0)

    # Query offsets
    q_start_m = pid_m * BLOCK_M
    q_off = q_start_m + tl.arange(0, BLOCK_M)

    # Base pointers for this (b, h) slice
    base_q = Q + pid_b * stride_qb + pid_h * stride_qh
    base_k = K + pid_b * stride_kb + pid_h * stride_kh
    base_v = V + pid_b * stride_vb + pid_h * stride_vh
    base_o = O + pid_b * stride_ob + pid_h * stride_oh
    base_lse = LSE + pid_b * stride_lse_b + pid_h * stride_lse_h

    # Load Q tile [BLOCK_M, D] -- constant across K-loop iterations
    q_ptrs = base_q + q_off[:, None] * stride_qs + tl.arange(0, 128)[None, :] * stride_qd
    q_tile = tl.load(q_ptrs, mask=q_off[:, None] < seqlen, other=0.0)

    # Online softmax state
    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)

    # Accumulator for output projection
    acc = tl.zeros([BLOCK_M, 128], dtype=tl.float32)

    # Number of K/V blocks to iterate over
    num_blocks_n = tl.cdiv(seqlen, BLOCK_N)

    for start_n in range(num_blocks_n):
        n_off = start_n * BLOCK_N + tl.arange(0, BLOCK_N)

        # Load K tile [BLOCK_N, D]
        k_ptrs = base_k + n_off[:, None] * stride_ks + tl.arange(0, 128)[None, :] * stride_kd
        k_tile = tl.load(k_ptrs, mask=n_off[:, None] < seqlen, other=0.0)

        # Compute attention scores: Q @ K^T  ->  [BLOCK_M, BLOCK_N]
        s = tl.dot(q_tile, k_tile.T) * inv_sqrt_d

        # Causal mask: query_pos >= key_pos  (attend to self and past)
        causal_mask = (q_off[:, None] + q_start_m) >= (n_off[None, :] + start_n)
        s = tl.where(causal_mask, s, float("-inf"))

        # New per-row max
        m_i_new = tl.maximum(m_i, tl.max(s, axis=1))

        # Scale factor for previous accumulation
        m_i_old = m_i
        m_i = m_i_new
        alpha = tl.exp(m_i_old - m_i)

        # Normalized attention weights
        p = tl.exp(s - m_i[:, None])

        # Accumulated scale (denominator of normalized softmax)
        l_i = alpha * l_i + tl.sum(p, axis=1)

        # Load V tile [BLOCK_N, D]
        v_ptrs = base_v + n_off[:, None] * stride_vs + tl.arange(0, 128)[None, :] * stride_vd
        v_tile = tl.load(v_ptrs, mask=n_off[:, None] < seqlen, other=0.0)

        # Weighted sum: previous_output * alpha + attention @ V
        acc = alpha[:, None] * acc + tl.dot(p, v_tile)

    # Write output O [BLOCK_M, D] -> bf16
    o_ptrs = base_o + q_off[:, None] * stride_os + tl.arange(0, 128)[None, :] * stride_od
    tl.store(o_ptrs, acc.to(tl.bfloat16), mask=q_off[:, None] < seqlen)

    # Write LSE [BLOCK_M] -> fp32
    # LSE = max(P) + log(sum(exp(P - max(P)))) = m_i + log(l_i)
    lse_ptrs = base_lse + q_off * stride_lse_s
    tl.store(lse_ptrs, m_i + tl.log(l_i), mask=q_off < seqlen)


def run(Q, K, V, O, LSE):
    """Causal multi-head attention forward pass with LSE output.
    
    Implements FlashAttention-3 style online softmax algorithm:
    - Processes one (batch, head) pair per CTA tile group
    - Iterates over key/value blocks sequentially
    - Accumulates output in FP32, stores as BF16
    - Returns log-sum-exp in FP32
    """
    torch.cuda.set_device(Q.device)

    bsz, num_heads, seqlen, head_dim = Q.shape

    assert K.shape == Q.shape, f"K shape {K.shape} != Q shape {Q.shape}"
    assert V.shape == Q.shape, f"V shape {V.shape} != Q shape {Q.shape}"
    assert O.shape == Q.shape, f"O shape {O.shape} != Q shape {Q.shape}"
    assert LSE.shape == (bsz, num_heads, seqlen), \
        f"LSE shape {LSE.shape} != ({bsz}, {num_heads}, {seqlen})"
    assert Q.dtype == torch.bfloat16, f"Expected bf16, got {Q.dtype}"
    assert head_dim == 128, f"Only head_dim=128 supported, got {head_dim}"

    # Strides
    sq, sk, sv, so = Q.stride(), K.stride(), V.stride(), O.stride()
    sl = LSE.stride()

    BLOCK_M = 64
    BLOCK_N = 64

    # Grid: (tiles_along_query, num_heads, batch_size)
    num_tiles_m = triton.cdiv(seqlen, BLOCK_M)
    grid = (num_tiles_m, num_heads, bsz)

    _flash_attn_fwd_kernel[grid](
        Q, K, V, O, LSE,
        num_heads,
        seqlen,
        sq[0], sq[1], sq[2], sq[3],
        sk[0], sk[1], sk[2], sk[3],
        sv[0], sv[1], sv[2], sv[3],
        so[0], so[1], so[2], so[3],
        sl[0], sl[1], sl[2],
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=3,
    )