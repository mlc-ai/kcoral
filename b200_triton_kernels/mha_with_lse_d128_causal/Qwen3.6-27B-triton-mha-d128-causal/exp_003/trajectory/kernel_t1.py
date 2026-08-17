import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=3),
    ],
    key=["S", "H"],
)
@triton.jit
def _mha_kernel(
    Q, K, V, O, LSE,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lse_b, stride_lse_h, stride_lse_s,
    H, S, D,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Causal multi-head attention forward with online softmax (FlashAttention style).

    Strategy:
    - Each program handles one (batch, head, query-block) tile of shape (BLOCK_M, D).
    - Online softmax updates m_i/l_i/acc incrementally across key-block sweeps.
    - Causal mask applied per-key-block; early termination when key block is fully past all queries.
    - Outputs O (bf16) and LSE (fp32 natural-log).
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(S, BLOCK_M)

    # Decode flat program id -> (batch, head, query-block index)
    bh_idx = pid // num_pid_m
    block_m = pid % num_pid_m
    b = bh_idx // H
    h = bh_idx % H

    # Query-position offsets: [BLOCK_M]
    off_m = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_m_last = off_m[-1]  # largest query pos in this tile

    # Head-dimension offsets: [D]
    off_d = tl.arange(0, D)

    # Load Q tile [BLOCK_M, D] once – reuse across all K-V sweeps
    q_ptrs = Q + b * stride_qb + h * stride_qh + \
             off_m[:, None] * stride_qs + off_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=(off_m[:, None] < S) & (off_d[None, :] < D), other=0.0).to(tl.float32)

    # Online softmax initial state
    acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)   # running per-row max
    l_i = tl.full((BLOCK_M,), 1.0, dtype=tl.float32)             # running per-row sum of exps

    num_pid_n = tl.cdiv(S, BLOCK_N)

    # ---- sweep across key blocks --------------------------------------------------
    for start_n in range(num_pid_n):
        n_off = start_n * BLOCK_N + tl.arange(0, BLOCK_N)

        # Early exit: if every key position in this block is past every query in our tile,
        # the causal mask would zero everything → safe to stop.
        if start_n * BLOCK_N > off_m_last:
            break

        # Load K tile [BLOCK_N, D]
        k_ptrs = K + b * stride_kb + h * stride_kh + \
                 n_off[:, None] * stride_ks + off_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=(n_off[:, None] < S) & (off_d[None, :] < D), other=0.0).to(tl.float32)

        # Load V tile [BLOCK_N, D]
        v_ptrs = V + b * stride_vb + h * stride_vh + \
                 n_off[:, None] * stride_vs + off_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=(n_off[:, None] < S) & (off_d[None, :] < D), other=0.0).to(tl.float32)

        # --- attention scores [BLOCK_M, BLOCK_N] ------------------------------------
        qk = tl.dot(q, k.T) * scale

        # Apply causal mask  (key_pos <= query_pos)
        is_causal = n_off[None, :] <= off_m[:, None]
        qk = tl.where(is_causal, qk, float('-inf'))

        # --- online softmax update ---------------------------------------------------
        m_ij = tl.max(qk, axis=1)                              # [BLOCK_M]
        m_i_new = tl.maximum(m_i, m_ij)                         # [BLOCK_M]

        alpha = tl.exp(m_i - m_i_new)                           # [BLOCK_M]
        p = tl.exp(qk - m_i_new[:, None])                       # [BLOCK_M, BLOCK_N]

        acc = alpha[:, None] * acc + tl.dot(p, v)               # [BLOCK_M, D]
        l_i = alpha * l_i + tl.sum(p, axis=1)                   # [BLOCK_M]
        m_i = m_i_new                                           # [BLOCK_M]

    # ---- epilogue -----------------------------------------------------------------
    # Normalize accumulated output
    acc_norm = acc / l_i[:, None]                               # [BLOCK_M, D]

    # Write O [BLOCK_M, D] -> bf16
    o_ptrs = O + b * stride_ob + h * stride_oh + \
             off_m[:, None] * stride_os + off_d[None, :] * stride_od
    tl.store(o_ptrs, acc_norm.to(tl.bfloat16),
             mask=(off_m[:, None] < S) & (off_d[None, :] < D))

    # Write LSE [BLOCK_M] -> fp32  (natural log)
    lse_val = m_i + tl.log(l_i)                                 # [BLOCK_M]
    lse_ptrs = LSE + b * stride_lse_b + h * stride_lse_h + off_m * stride_lse_s
    tl.store(lse_ptrs, lse_val, mask=off_m < S)


def run(Q, K, V, O, LSE):
    """
    Causal multi-head attention forward.

    Computes:
      O   = softmax(Q @ K^T / sqrt(D)) @ V          (bf16, [B,H,S,D])
      LSE = logsumexp(Q @ K^T / sqrt(D))  per-row   (fp32,  [B,H,S])

    with lower-triangular (causal) masking over (query_pos, key_pos).
    """
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape

    scale = 1.0 / float(D ** 0.5)

    BLOCK_M = 128
    BLOCK_N = 64

    num_pid_m = triton.cdiv(S, BLOCK_M)
    grid = (B * H * num_pid_m,)

    _mha_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        H, S, D,
        scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )