import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
    ],
    key=['S'],
)
@triton.jit
def _mha_fwd(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, LSE_ptr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lb, stride_lh, stride_ls,
    B, H, S,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Non-causal MHA forward kernel with numerically-stable online softmax."""
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    pid_b = pid_bh // H
    pid_h = pid_bh % H

    # Base offsets for this (batch, head)
    q_base = Q_ptr + pid_b * stride_qb + pid_h * stride_qh
    k_base = K_ptr + pid_b * stride_kb + pid_h * stride_kh
    v_base = V_ptr + pid_b * stride_vb + pid_h * stride_vh
    o_base = O_ptr + pid_b * stride_ob + pid_h * stride_oh
    lse_base = LSE_ptr + pid_b * stride_lb + pid_h * stride_lh

    # Index tiles
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    # Load Q tile [BLOCK_M, BLOCK_D]
    q_ptrs = q_base + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    mask_m = offs_m[:, None] < S
    q = tl.load(q_ptrs, mask=mask_m, other=0.0)

    # Accumulators (FP32 for numerical stability)
    acc_o = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    m_i = tl.full((BLOCK_M,), -float('inf'), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)

    num_kv_blocks = tl.cdiv(S, BLOCK_N)

    for start_n in range(num_kv_blocks):
        offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n[:, None] < S

        # Load K tile [BLOCK_N, BLOCK_D]
        k_ptrs = k_base + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=mask_n, other=0.0)

        # Q @ K^T -> [BLOCK_M, BLOCK_N] fp32
        scores = tl.dot(q, k.T) * scale

        # Zero-K padding (other=0.0) must NOT contribute to softmax.
        # Set their scores to -inf so exp(-inf)=0.
        scores = tl.where(mask_n.T, scores, float('-inf'))

        # --- Online softmax update ---
        m_ij = tl.max(scores, axis=1)                         # [BLOCK_M]
        m_i_new = tl.maximum(m_i, m_ij)                       # [BLOCK_M]

        alpha = tl.exp(m_i - m_i_new)                         # scale-old-factor
        p = tl.exp(scores - m_i_new[:, None])                 # [BLOCK_M, BLOCK_N]

        l_i_new = alpha * l_i + tl.sum(p, axis=1)             # [BLOCK_M]

        # Rescale accumulated output and add new contribution
        acc_o = acc_o * alpha[:, None]

        # Load V tile [BLOCK_N, BLOCK_D]
        v_ptrs = v_base + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=mask_n, other=0.0)

        acc_o = tl.dot(p, v.to(tl.float32), acc=acc_o)

        m_i = m_i_new
        l_i = l_i_new

    # Normalise
    acc_o = acc_o / l_i[:, None]

    # Store output O [BLOCK_M, BLOCK_D] in bf16
    o_ptrs = o_base + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc_o.to(tl.bfloat16), mask=mask_m)

    # Store LSE [BLOCK_M] in fp32
    lse_val = m_i + tl.log(l_i)
    tl.store(lse_base + offs_m * stride_ls, lse_val, mask=offs_m < S)


def run(Q, K, V, O, LSE):
    """Compute multi-head attention O, LSE into preallocated output tensors."""
    torch.cuda.set_device(Q.device)

    B, H, S, D = Q.shape
    scale = 1.0 / (D ** 0.5)

    grid = lambda META: (triton.cdiv(S, META["BLOCK_M"]), B * H)

    _mha_fwd[grid](
        Q, K, V, O, LSE,
        *Q.stride(),
        *K.stride(),
        *V.stride(),
        *O.stride(),
        *LSE.stride(),
        B, H, S,
        scale,
        BLOCK_D=D,
    )