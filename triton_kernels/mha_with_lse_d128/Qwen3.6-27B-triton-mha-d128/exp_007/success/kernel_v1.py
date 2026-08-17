import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=8, num_stages=3),
    ],
    key=["S"],
)
@triton.jit
def _attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    lse_ptr,
    stride_bh,
    stride_seq,
    S,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
):
    """FlashAttention kernel: one CTA per (batch, head) pair, persistent scheduling.
    
    Each program processes all query positions for its assigned head via tiled GEMM
    with online softmax normalization. Full D loaded at once since D<=128.
    """

    pid_bh = tl.program_id(0)
    bh_offset = pid_bh * stride_bh

    # Query sequence tiling: iterate over all query tiles for this (b,h)
    num_qtiles = tl.cdiv(S, BLOCK_M)

    for qtile in tl.range(num_qtiles, num_stages=1):
        m_abs = qtile * BLOCK_M
        m_idx = m_abs + tl.arange(0, BLOCK_M)
        m_mask = m_idx < S

        # Load Q tile [BLOCK_M, D]
        q_ptrs = q_ptr + bh_offset + m_idx[:, None] * stride_seq + tl.arange(0, D)[None, :]
        q = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0).to(tl.float32)

        # Online softmax accumulators for this query tile
        m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
        l_i = tl.full((BLOCK_M,), 1.0, dtype=tl.float32)
        acc_o = tl.zeros((BLOCK_M, D), dtype=tl.float32)

        # Iterate over key sequence tiles
        num_ktiles = tl.cdiv(S, BLOCK_N)
        for ktile in tl.range(num_ktiles, num_stages=3):
            n_abs = ktile * BLOCK_N
            n_idx = n_abs + tl.arange(0, BLOCK_N)
            n_mask = n_idx < S

            # Load K tile [BLOCK_N, D] and V tile [BLOCK_N, D]
            k_ptrs = k_ptr + bh_offset + n_idx[:, None] * stride_seq + tl.arange(0, D)[None, :]
            k = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0).to(tl.float32)

            v_ptrs = v_ptr + bh_offset + n_idx[:, None] * stride_seq + tl.arange(0, D)[None, :]
            v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0).to(tl.float32)

            # Attention scores: Q @ K^T * scale -> [BLOCK_M, BLOCK_N]
            s = tl.dot(q, k.T) * scale

            # Online softmax: stable update
            m_ij = tl.max(s, axis=1)
            m_new = tl.maximum(m_i, m_ij)

            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s - m_new[:, None])

            acc_o = alpha[:, None] * acc_o + tl.dot(p, v)

            beta = tl.sum(p, axis=1)
            l_i = alpha * l_i + beta
            m_i = m_new

        # Final normalize and write output O tile
        l_safe = tl.where(l_i > 0.0, l_i, 1.0)
        out = (acc_o / l_safe[:, None]).to(tl.bfloat16)
        o_ptrs = o_ptr + bh_offset + m_idx[:, None] * stride_seq + tl.arange(0, D)[None, :]
        tl.store(o_ptrs, out, mask=m_mask[:, None])

        # Write LSE = m + log(l) for this query tile
        lse_val = m_i + tl.log(l_safe)
        lse_offsets = pid_bh * S + m_idx
        tl.store(lse_ptr + lse_offsets, lse_val, mask=m_mask)


def run(Q, K, V, O, LSE):
    """Multi-head attention forward: O = softmax(Q@K^T/sqrt(D))@V with LSE.

    Inputs: Q, K, V : [B, H, S, D] bf16
    Outputs: O : [B, H, S, D] bf16, LSE : [B, H, S] f32
    Destination-passing: writes into preallocated O and LSE tensors.
    """
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    BH = B * H

    # Contiguous reshapes to [BH, S, D]
    Q_c = Q.reshape(BH, S, D).contiguous()
    K_c = K.reshape(BH, S, D).contiguous()
    V_c = V.reshape(BH, S, D).contiguous()
    O_c = O.reshape(BH, S, D).contiguous()

    stride_bh = S * D  # stride between (b,h) pairs
    stride_seq = D     # stride between sequence positions

    scale = 1.0 / (float(D) ** 0.5)

    # One CTA per (batch, head) pair — persistent, processes all seq positions
    grid = (BH,)

    _attention_kernel[grid](
        Q_c, K_c, V_c, O_c, LSE,
        stride_bh, stride_seq,
        S, scale,
        D=D,
    )