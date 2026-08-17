import torch
import triton
import triton.language as tl

@triton.jit
def _mha_causal_fwd_kernel(
    Q, K, V, O, LSE,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    stride_lseb, stride_lseh, stride_lsem,
    S,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    q_ptrs = Q + pid_b * stride_qb + pid_h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q_mask = offs_m[:, None] < S
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    acc_o = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # limit_n bounds the keys we evaluate for causality and sequence length
    limit_n = tl.minimum(start_m + BLOCK_M, ((S + BLOCK_N - 1) // BLOCK_N) * BLOCK_N)

    # Precomputed constant for scale: 1 / sqrt(128) * log2(e)
    RCP_LN2 = 1.4426950408889634
    scale_ln2 = 0.08838834764831845 * RCP_LN2

    k_base = K + pid_b * stride_kb + pid_h * stride_kh
    v_base = V + pid_b * stride_vb + pid_h * stride_vh

    for start_n in range(0, limit_n, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k_mask = offs_n[:, None] < S
        k_ptrs = k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        v_ptrs = v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd

        k = tl.load(k_ptrs, mask=k_mask, other=0.0)
        v = tl.load(v_ptrs, mask=k_mask, other=0.0)

        # Q @ K^T
        acc_scores = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        scores = tl.dot(q, k.T, acc_scores)
        scores = scores * scale_ln2

        # Apply causal mask and sequence boundary mask
        valid_mask = (offs_n[None, :] <= offs_m[:, None]) & (offs_n[None, :] < S)
        scores = tl.where(valid_mask, scores, -float('inf'))

        # Online Softmax
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        safe_m_ij = tl.where(m_ij == -float('inf'), 0.0, m_ij)
        alpha = tl.math.exp2(m_i - safe_m_ij)
        p = tl.math.exp2(scores - safe_m_ij[:, None])

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc_o = acc_o * alpha[:, None]

        # P @ V
        acc_o = tl.dot(p.to(tl.bfloat16), v, acc_o)
        m_i = m_ij

    # Normalize output
    safe_l_i = tl.where(l_i == 0.0, 1.0, l_i)
    out = acc_o / safe_l_i[:, None]

    o_ptrs = O + pid_b * stride_ob + pid_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, out.to(tl.bfloat16), mask=q_mask)

    # Convert LogSumExp to natural log scale
    LN2 = 0.6931471805599453
    lse = (m_i + tl.math.log2(safe_l_i)) * LN2
    lse = tl.where(l_i == 0.0, -float('inf'), lse)

    lse_ptrs = LSE + pid_b * stride_lseb + pid_h * stride_lseh + offs_m * stride_lsem
    tl.store(lse_ptrs, lse, mask=offs_m < S)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    if S == 0:
        return
    
    grid = lambda META: (
        triton.cdiv(S, META['BLOCK_M']),
        B,
        H,
    )
    
    _mha_causal_fwd_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        S,
        BLOCK_M=64, BLOCK_N=64, BLOCK_D=128,
        num_warps=4, num_stages=3
    )