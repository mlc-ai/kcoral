import torch
import triton
import triton.language as tl

@triton.jit
def _attn_fwd_kernel(
    Q, K, V, O, LSE,
    S,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, 128)

    q_ptrs = Q + pid_b * stride_qb + pid_h * stride_qh + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=offs_m[:, None] < S, other=0.0)

    RCP_LN2: tl.constexpr = 1.4426950408889634
    scale: tl.constexpr = 0.08838834764831843  # 1.0 / sqrt(128)
    q = (q * scale * RCP_LN2).to(tl.bfloat16)

    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, 128), tl.float32)

    k_ptrs = K + pid_b * stride_kb + pid_h * stride_kh + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = V + pid_b * stride_vb + pid_h * stride_vh + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd

    for start_n in range(0, S, BLOCK_N):
        if start_n + BLOCK_N <= S:
            k = tl.load(k_ptrs)
            v = tl.load(v_ptrs)
            
            scores = tl.dot(q, k.T)
            
            m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
            safe_m_ij = tl.where(m_ij == -float("inf"), 0.0, m_ij)
            alpha = tl.math.exp2(m_i - safe_m_ij)
            p = tl.math.exp2(scores - safe_m_ij[:, None])
            
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None]
            acc = tl.dot(p.to(tl.bfloat16), v, acc)
            m_i = m_ij
        else:
            curr_n = start_n + offs_n
            mask = curr_n < S
            k = tl.load(k_ptrs, mask=mask[:, None], other=0.0)
            v = tl.load(v_ptrs, mask=mask[:, None], other=0.0)
            
            scores = tl.dot(q, k.T)
            scores = tl.where(mask[None, :], scores, -float("inf"))
            
            m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
            safe_m_ij = tl.where(m_ij == -float("inf"), 0.0, m_ij)
            alpha = tl.math.exp2(m_i - safe_m_ij)
            p = tl.math.exp2(scores - safe_m_ij[:, None])
            
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None]
            acc = tl.dot(p.to(tl.bfloat16), v, acc)
            m_i = m_ij

        k_ptrs += BLOCK_N * stride_ks
        v_ptrs += BLOCK_N * stride_vs

    safe_l_i = tl.where(l_i == 0.0, 1.0, l_i)
    output = acc / safe_l_i[:, None]
    
    LN2: tl.constexpr = 0.6931471805599453
    lse_log2 = tl.where(l_i == 0.0, -float("inf"), m_i + tl.math.log2(safe_l_i))
    lse = lse_log2 * LN2

    o_ptrs = O + pid_b * stride_ob + pid_h * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(o_ptrs, output.to(tl.bfloat16), mask=offs_m[:, None] < S)

    lse_ptrs = LSE + pid_b * stride_lseb + pid_h * stride_lseh + offs_m * stride_lses
    tl.store(lse_ptrs, lse, mask=offs_m < S)


def run(Q, K, V, O, LSE):
    """
    Computes non-causal multi-head attention forward returning O and LSE.
    """
    torch.cuda.set_device(Q.device)
    B = Q.size(0)
    H = Q.size(1)
    S = Q.size(2)
    
    BLOCK_M = 128
    BLOCK_N = 64
    
    grid = (triton.cdiv(S, BLOCK_M), B, H)
    
    _attn_fwd_kernel[grid](
        Q, K, V, O, LSE,
        S,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=3,
    )