import torch
import triton
import triton.language as tl


@triton.jit
def _mha_kernel(
    Q, K, V, O, LSE,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lse_b, stride_lse_h, stride_lse_s,
    H, S,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Causal MHA forward kernel using online softmax (FlashAttention style)."""
    pid = tl.program_id(0)
    num_blocks_m = tl.cdiv(S, BLOCK_M)

    # Decode (batch, head, query-block) from flattened program id
    bh = pid // num_blocks_m
    block_m = pid % num_blocks_m
    b = bh // H
    h = bh % H

    # Indexing helpers
    off_m = block_m * BLOCK_M + tl.arange(0, BLOCK_M)   # [BLOCK_M] – query positions
    off_d = tl.arange(0, BLOCK_D)                        # [BLOCK_D] – head-dim positions

    # --- Load Q tile [BLOCK_M, BLOCK_D] ----------------------------------------
    q_ptrs = Q + b * stride_qb + h * stride_qh + \
             off_m[:, None] * stride_qs + off_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=off_m[:, None] < S, other=0.0).to(tl.float32)

    # --- Online softmax state --------------------------------------------------
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)   # running max
    l_i = tl.full((BLOCK_M,), 0.0, dtype=tl.float32)             # running sum

    num_blocks_n = tl.cdiv(S, BLOCK_N)

    for start_n in range(num_blocks_n):
        n_off = start_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N] – key positions

        # Load K tile [BLOCK_N, BLOCK_D]
        k_ptrs = K + b * stride_kb + h * stride_kh + \
                 n_off[:, None] * stride_ks + off_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=n_off[:, None] < S, other=0.0).to(tl.float32)

        # Load V tile [BLOCK_N, BLOCK_D]
        v_ptrs = V + b * stride_vb + h * stride_vh + \
                 n_off[:, None] * stride_vs + off_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=n_off[:, None] < S, other=0.0).to(tl.float32)

        # Q @ K^T  →  [BLOCK_M, BLOCK_N]
        qk = tl.dot(q, k.T) * scale

        # Causal mask: keep only entries where key_pos <= query_pos
        mask_2d = n_off[None, :] <= off_m[:, None]
        qk = tl.where(mask_2d, qk, float('-inf'))

        # --- Online softmax step ------------------------------------------------
        m_ij = tl.max(qk, axis=1)                    # [BLOCK_M]
        m_i_new = tl.maximum(m_i, m_ij)               # [BLOCK_M]

        alpha = tl.exp(m_i - m_i_new)                # [BLOCK_M]
        p = tl.exp(qk - m_i_new[:, None])            # [BLOCK_M, BLOCK_N]

        acc = alpha[:, None] * acc + tl.dot(p, v)    # [BLOCK_M, BLOCK_D]
        l_i = alpha * l_i + tl.sum(p, axis=1)        # [BLOCK_M]
        m_i = m_i_new                                # [BLOCK_M]

    # --- Normalize and store ----------------------------------------------------
    l_i_safe = tl.where(l_i > 0.0, l_i, 1.0)
    m_i_safe = tl.where(l_i > 0.0, m_i, 0.0)
    acc = acc / l_i_safe[:, None]

    # Output O [BLOCK_M, BLOCK_D] → bf16
    o_ptrs = O + b * stride_ob + h * stride_oh + \
             off_m[:, None] * stride_os + off_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(tl.bfloat16), mask=off_m[:, None] < S)

    # Output LSE [BLOCK_M] → float32
    lse_val = m_i_safe + tl.log(l_i_safe)
    lse_ptrs = LSE + b * stride_lse_b + h * stride_lse_h + off_m * stride_lse_s
    tl.store(lse_ptrs, lse_val, mask=off_m < S)


def run(Q, K, V, O, LSE):
    """
    Causal multi-head attention forward.

    Computes O = softmax(Q @ K^T / sqrt(D)) @ V  (bf16)
    and      LSE = logsumexp(Q @ K^T / sqrt(D))   (float32)
    with a lower-triangular (causal) mask over (query_pos, key_pos).
    """
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape

    scale = 1.0 / float(D ** 0.5)

    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_D = 128  # matches fixed head dimension D

    num_blocks_m = triton.cdiv(S, BLOCK_M)
    grid = (B * H * num_blocks_m,)

    _mha_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        H, S,
        scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=3,
    )