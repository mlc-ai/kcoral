import torch
import triton
import triton.language as tl


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
    """Optimized FlashAttention kernel for Hopper SM90.
    
    Each program handles one (batch*head, query_tile). Iterates over key tiles
    with online softmax. All tile dims are constexpr for optimal codegen.
    Q/K/V loaded as bf16; fp32 accumulation via tl.dot native behavior.
    """
    pid_bh = tl.program_id(0)
    pid_qtile = tl.program_id(1)

    bh_offset = pid_bh * stride_bh
    m_abs = pid_qtile * BLOCK_M

    # Row masks and indices
    m_idx = m_abs + tl.arange(0, BLOCK_M)
    m_mask = m_idx < S
    d_idx = tl.arange(0, D)

    # Load Q tile [BLOCK_M, D] once
    q_ptrs = q_ptr + bh_offset + m_idx[:, None] * stride_seq + d_idx[None, :]
    q = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0)

    # Online softmax accumulators (fp32)
    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.full((BLOCK_M,), 1.0, dtype=tl.float32)
    acc_o = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    # Key tile loop
    for start_n in range(0, S, BLOCK_N):
        n_idx = start_n + tl.arange(0, BLOCK_N)
        n_mask = n_idx < S

        # Load K and V tiles as bf16
        k_ptrs = k_ptr + bh_offset + n_idx[:, None] * stride_seq + d_idx[None, :]
        k = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)

        v_ptrs = v_ptr + bh_offset + n_idx[:, None] * stride_seq + d_idx[None, :]
        v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)

        # Q @ K^T : bf16 x bf16 -> fp32 natually via Tensor Cores
        s = tl.dot(q, k.T) * scale

        # Mask out-of-bounds keys by zeroing their contribution
        s = tl.where(n_mask[None, :], s, float("-inf"))

        # Online softmax
        m_ij = tl.max(s, axis=1)
        m_new = tl.maximum(m_i, m_ij)

        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])

        # P @ V : fp32(f32 from exp) x bf16 -> need matching types
        # Cast V to fp32 for the dot with fp32 probabilities
        v_f32 = v.to(tl.float32)
        acc_o = alpha[:, None] * acc_o + tl.dot(p, v_f32)

        beta = tl.sum(p, axis=1)
        l_i = alpha * l_i + beta
        m_i = m_new

    # Normalize and store output
    l_safe = tl.where(l_i > 0.0, l_i, 1.0)
    out = (acc_o / l_safe[:, None]).to(tl.bfloat16)
    o_ptrs = o_ptr + bh_offset + m_idx[:, None] * stride_seq + d_idx[None, :]
    tl.store(o_ptrs, out, mask=m_mask[:, None])

    # Write LSE
    lse_val = m_i + tl.log(l_safe)
    lse_offsets = pid_bh * S + m_idx
    tl.store(lse_ptr + lse_offsets, lse_val, mask=m_mask)


def run(Q, K, V, O, LSE):
    """Multi-head attention forward with LSE output.

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

    stride_bh = S * D
    stride_seq = D
    scale = 1.0 / (float(D) ** 0.5)

    # For S=4096, D=128 on Hopper: tuned values
    BLOCK_M = 64
    BLOCK_N = 64

    num_qtiles = triton.cdiv(S, BLOCK_M)
    grid = (BH, num_qtiles)

    _attention_kernel[grid](
        Q_c, K_c, V_c, O_c, LSE,
        stride_bh, stride_seq,
        S, scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        D=D,
        num_warps=8,
        num_stages=3,
    )