import torch
import triton
import triton.language as tl


@triton.jit
def _mha_kernel(
    Q,
    K,
    V,
    O,
    LSE,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    seq_len,
    num_heads,
    SCALE_INV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Tiled multi-head attention kernel.
    Each program handles one (batch, head, query_tile) combination.
    
    Computes attention in-place using online softmax:
      - Iterate over key/value blocks
      - Accumulate scaled partial results with running max/sum for stability
      - Finalize by normalization
    """
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    pid_b = pid_bh // num_heads
    pid_h = pid_bh % num_heads

    # Offset arrays for indexing
    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)       # query row offsets [BLOCK_M]
    off_n = tl.arange(0, BLOCK_N)                          # key col offsets [BLOCK_N]
    off_d = tl.arange(0, BLOCK_D)                          # feature dim offsets [BLOCK_D]

    # Build base pointers for this (batch, head) pair
    q_ptr = Q + pid_b * stride_qb + pid_h * stride_qh
    k_ptr = K + pid_b * stride_kb + pid_h * stride_kh
    v_ptr = V + pid_b * stride_vb + pid_h * stride_vh
    o_ptr = O + pid_b * stride_ob + pid_h * stride_oh
    lse_ptr = LSE + pid_b * stride_lseb + pid_h * stride_lseh

    # Initialize accumulators
    acc_o = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), value=float('-inf'), dtype=tl.float32)
    l_i = tl.full((BLOCK_M,), value=1.0, dtype=tl.float32)

    # Validity mask for query rows (outside seq_len → skipped)
    q_row_valid = off_m[:, None] < seq_len                 # [BLOCK_M, 1]

    # Load Q tile ONCE outside the loop (Q doesn't depend on key block)
    q_ptrs = q_ptr + off_m[:, None] * stride_qs + off_d[None, :] * stride_qd
    q_tile = tl.load(q_ptrs, mask=q_row_valid & (off_d[None, :] < BLOCK_D), other=0.0)

    # Loop over key/value sequence blocks
    for start_n in range(0, seq_len, BLOCK_N):
        n_idx = start_n + off_n                            # [BLOCK_N]
        col_valid = n_idx[None, :] < seq_len                # [1, BLOCK_N]

        # Load K tile [BLOCK_N, BLOCK_D]
        k_ptrs = k_ptr + n_idx[:, None] * stride_ks + off_d[None, :] * stride_kd
        k_tile = tl.load(k_ptrs, mask=col_valid & (off_d[None, :] < BLOCK_D), other=0.0)

        # Load V tile [BLOCK_N, BLOCK_D]
        v_ptrs = v_ptr + n_idx[:, None] * stride_vs + off_d[None, :] * stride_vd
        v_tile = tl.load(v_ptrs, mask=col_valid & (off_d[None, :] < BLOCK_D), other=0.0)

        # Attention scores: S = Q @ K^T / sqrt(D) -> [BLOCK_M, BLOCK_N]
        # tl.dot contracts last dims; both q_tile and k_tile have last dim BLOCK_D
        s = tl.dot(q_tile, k_tile) * SCALE_INV

        # --- Online softmax update ---
        m_prev = m_i                                       # save previous max

        # New per-row max (ignore out-of-bounds cols via -inf sentinel)
        m_i = tl.maximum(m_i, tl.max(s, axis=1))

        # Correction factor: exp(m_prev - m_i), broadcast to [BLOCK_M, 1]
        alpha = tl.exp(m_prev - m_i)

        # Softmax weights for this block
        p = tl.exp(s - m_i[:, None])                       # [BLOCK_M, BLOCK_N]

        # Accumulate denominator and numerator
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc_o = acc_o * alpha[:, None] + tl.dot(p, v_tile)

    # --- Finalization ---
    # Guard against degenerate case (no valid keys)
    l_i = tl.maximum(l_i, 1e-12)

    # Normalize output
    acc_o = acc_o / l_i[:, None]

    # LSE = max + log(sum(exp(P - max)))
    lse_out = m_i + tl.log(l_i)

    # Write O tile [BLOCK_M, BLOCK_D]
    o_ptrs = o_ptr + off_m[:, None] * stride_os + off_d[None, :] * stride_od
    tl.store(o_ptrs, acc_o.to(O.dtype.element_ty), mask=q_row_valid & (off_d[None, :] < BLOCK_D))

    # Write LSE [BLOCK_M]
    lse_ptrs = lse_ptr + off_m * stride_lses
    tl.store(lse_ptrs, lse_out, mask=off_m < seq_len)


def run(Q, K, V, O, LSE):
    """
    Multi-head attention forward pass: O = softmax(Q @ K^T / sqrt(D)) @ V
    
    Args (destination-passing):
        Q, K, V : bf16 tensors of shape (B, H, S, D)
        O       : preallocated bf16 tensor of shape (B, H, S, D)
        LSE     : preallocated fp32 tensor of shape (B, H, S)
    """
    torch.cuda.set_device(Q.device)

    B = Q.shape[0]
    H = Q.shape[1]
    S = Q.shape[2]
    D = Q.shape[3]

    # Tile sizes tuned for Hopper SM90 with bf16 and D=128
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = D  # Full head dimension as reduction tile

    # Precompute 1/sqrt(D) in fp32 for numerical accuracy
    scale_inv = 1.0 / (D ** 0.5)

    # Grid: axis 0 = batch x head, axis 1 = query tiles
    grid = (B * H, triton.cdiv(S, BLOCK_M))

    _mha_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        S, H,
        SCALE_INV=scale_inv,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=3,
    )