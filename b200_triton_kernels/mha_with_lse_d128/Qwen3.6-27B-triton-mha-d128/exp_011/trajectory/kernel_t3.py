import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=4, num_stages=3),
    ],
    key=['S', 'D'],
)
@triton.jit
def _mha_fwd_kernel(
    Q, K, V, O, LSE,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lb, stride_lh, stride_ls,
    S, D,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Non-causal multi-head attention with online softmax."""
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    # Decompose batch*head id
    pb = pid_bh // D
    ph = pid_bh - pb * D

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n_base = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    m_mask = offs_m < S
    m_mask_2d = m_mask[:, None]

    q_base = Q + pb * stride_qb + ph * stride_qh
    k_base = K + pb * stride_kb + ph * stride_kh
    v_base = V + pb * stride_vb + ph * stride_vh
    o_base = O + pb * stride_ob + ph * stride_oh
    lse_base = LSE + pb * stride_lb + ph * stride_lh

    # Load Q tile [BLOCK_M, BLOCK_D]
    q_ptrs = q_base + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=m_mask_2d, other=0.0)

    prev_max = tl.full((BLOCK_M,), -float('inf'), tl.float32)
    prev_sum = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    num_steps = tl.cdiv(S, BLOCK_N)

    for step in range(num_steps):
        n_start = step * BLOCK_N
        offs_n = n_start + offs_n_base
        n_mask = offs_n < S
        n_mask_row = n_mask[None, :]

        # Load K [BLOCK_N, BLOCK_D]
        k_ptrs = k_base + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=n_mask_row, other=0.0)

        # scores [BLOCK_M, BLOCK_N]
        scores = tl.dot(q, tl.trans(k)) * scale
        scores = tl.where(n_mask_row, scores, float('-inf'))

        # Online softmax update
        cur_max = tl.max(scores, axis=1)
        new_max = tl.maximum(prev_max, cur_max)

        alpha = tl.exp(prev_max - new_max)
        p = tl.exp(scores - new_max[:, None])

        prev_sum = alpha * prev_sum + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        # Load V [BLOCK_N, BLOCK_D]
        v_ptrs = v_base + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=n_mask_row, other=0.0)

        acc = tl.dot(p, v.to(tl.float32), acc=acc)

        prev_max = new_max

    # Normalize
    inv_sum = 1.0 / prev_sum
    acc = acc * inv_sum[:, None]

    # Store O [BLOCK_M, BLOCK_D]
    o_ptrs = o_base + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(tl.bfloat16), mask=m_mask_2d)

    # Store LSE [BLOCK_M]
    lse_val = prev_max + tl.log(prev_sum)
    lse_ptrs = lse_base + offs_m * stride_ls
    tl.store(lse_ptrs, lse_val, mask=m_mask)


def run(Q, K, V, O, LSE):
    """Compute multi-head attention O = softmax(QK^T/sqrt(D))V and LSE into preallocated outputs."""
    torch.cuda.set_device(Q.device)

    B, H, S, D = Q.shape
    scale = 1.0 / (D ** 0.5)

    num_bm = triton.cdiv(S, 128)
    grid = (num_bm, B * H)

    _mha_fwd_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        S, D,
        scale,
        BLOCK_D=D,
    )