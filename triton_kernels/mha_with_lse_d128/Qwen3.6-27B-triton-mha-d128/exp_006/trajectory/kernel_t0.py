import torch
import triton
import triton.language as tl


@triton.jit
def _attention_kernel(
    Q, K, V, O, LSE,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lsb, stride_lsh, stride_lss,
    S,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
):
    """FlashAttention forward kernel using online softmax algorithm."""

    # Decompose 1D program id into (query_tile_idx, batch_idx, head_idx)
    pid = tl.program_id(0)

    off_m = tl.arange(0, BLOCK_M)  # [BLOCK_M]
    off_d = tl.arange(0, BLOCK_D)  # [BLOCK_D]

    # Pointers to Q (tile [BLOCK_M, BLOCK_D]) - loaded once outside loop
    q_ptrs = (Q
              + off_m[:, None] * stride_qs
              + off_d[None, :] * stride_qd)  # [BLOCK_M, BLOCK_D]

    # Initialize accumulators for online softmax
    m_i = tl.full((BLOCK_M,), float("-inf"), ACC_DTYPE)
    l_i = tl.zeros((BLOCK_M,), ACC_DTYPE)
    acc_o = tl.zeros((BLOCK_M, BLOCK_D), ACC_DTYPE)

    # Iterate over K, V blocks along sequence dimension
    for start_n in range(0, S, BLOCK_N):
        # Current N offset for this iteration
        cur_n = start_n + off_m  # [BLOCK_M] used for masking against S

        # Build K and V pointers for this KV tile
        k_ptrs = (K
                  + cur_n[:, None] * stride_ks
                  + off_d[None, :] * stride_kd)  # [BLOCK_N, BLOCK_D] -> masks to tile
        v_ptrs = (V
                  + cur_n[:, None] * stride_vs
                  + off_d[None, :] * stride_vd)  # [BLOCK_N, BLOCK_D]

        # Proper N offsets for load [BLOCK_N]
        n_off = start_n + tl.arange(0, BLOCK_N)  # [BLOCK_N]

        # Actual load pointers with correct tile indexing
        k_ptrs = (K
                  + n_off[:, None] * stride_ks
                  + off_d[None, :] * stride_kd)  # [BLOCK_N, BLOCK_D]
        v_ptrs = (V
                  + n_off[:, None] * stride_vs
                  + off_d[None, :] * stride_vd)  # [BLOCK_N, BLOCK_D]

        # Masks for boundaries
        m_mask = off_m < S
        n_mask = n_off < S

        # Load K tile [BLOCK_N, BLOCK_D], mask extra elements
        k = tl.load(k_ptrs,
                     mask=(n_mask[:, None] & (off_d < BLOCK_D)),
                     other=0.0)  # Will broadcast correctly

        # Compute Q @ K^T with scale
        # q_ptrs already built above, need Q pointers indexed with batch/head
        # Recompute with batch/head included
        q = tl.load(q_ptrs, mask=(m_mask[:, None]), other=0.0)

        # Scaled dot product: [BLOCK_M, BLOCK_D] @ [BLOCK_D, BLOCK_N]^T -> [BLOCK_M, BLOCK_N]
        s = tl.dot(q, k.T, acc=None, out_dtype=ACC_DTYPE)

        # Apply scaling factor 1/sqrt(D)
        scale = 1.0 / tl.sqrt(float(BLOCK_D))
        s = s * scale

        # Online softmax step 1: new running maximum
        m_ij = tl.max(s, axis=1, keep_dims=True)  # [BLOCK_M, 1]
        m_i_new = tl.maximum(m_i[:, None], m_ij).squeeze(1)  # [BLOCK_M]

        # Online softmax step 2: exponentiate with correction
        p = tl.exp(s - m_i_new[:, None])  # [BLOCK_M, BLOCK_N]

        # Correct old l_i contribution: l_i * exp(m_i_old - m_i_new)
        old_scale = tl.exp(m_i - m_i_new)  # [BLOCK_M]
        l_i_new = old_scale * l_i + tl.sum(p, axis=1, keep_dims=False)

        # Load V tile [BLOCK_N, BLOCK_D]
        v = tl.load(v_ptrs,
                     mask=(n_mask[:, None]),
                     other=0.0)

        # acc_o = old_scale * acc_o + p @ V
        acc_o = old_scale[:, None] * acc_o + tl.dot(p, v, out_dtype=ACC_DTYPE)

        # Update running statistics
        m_i = m_i_new
        l_i = l_i_new

    # Final normalization: O = acc_o / l_i
    o_final = acc_o / l_i[:, None]

    # Convert to output dtype and store
    o_store = o_final.to(O.dtype.element_ty)
    tl.store(O + q_ptrs, o_store, mask=m_mask[:, None])

    # Store LSE = m_i + log(l_i)
    lse_val = m_i + tl.log(l_i)
    lse_ptrs = LSE + off_m * stride_lss
    tl.store(lse_ptrs, lse_val, mask=m_mask)


def run(Q, K, V, O, LSE):
    """
    Compute multi-head attention forward pass:
      O = softmax(Q @ K^T / sqrt(D)) @ V   (bfloat16, [B, H, S, D])
      LSE = logsumexp(Q @ K^T / sqrt(D))   (float32, [B, H, S])

    Args:
        Q: [B, H, S, D] bfloat16 queries
        K: [B, H, S, D] bfloat16 keys
        V: [B, H, S, D] bfloat16 values
        O: [B, H, S, D] bfloat16 preallocated output
        LSE: [B, H, S] float32 preallocated output
    """
    torch.cuda.set_device(Q.device)

    B = Q.shape[0]
    H = Q.shape[1]
    S = Q.shape[2]
    D = Q.shape[3]

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = D  # Tile entire head dimension

    # Grid: one program per (batch, head, query_tile)
    num_q_tiles = triton.cdiv(S, BLOCK_M)
    grid = (num_q_tiles * B * H,)

    _attention_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        S,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        ACC_DTYPE=tl.float32,
        num_warps=4,
        num_stages=3,
    )