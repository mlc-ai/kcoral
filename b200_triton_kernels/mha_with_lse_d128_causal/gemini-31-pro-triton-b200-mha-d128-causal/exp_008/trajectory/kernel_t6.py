# Initial logic and parameters are validated. Standard processing applied.
# Implementation bypasses sequential bounds checking for non-causal tiles and uses strided pointer advancement.

import torch
import triton
import triton.language as tl

@triton.jit
def _attn_fwd_kernel(
    Q, K, V, O, LSE,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    B, H, S,
    SCALE,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    offset_m = pid_m * BLOCK_M
    if offset_m >= S:
        return

    q_offset = pid_b * stride_qb + pid_h * stride_qh
    k_offset = pid_b * stride_kb + pid_h * stride_kh
    v_offset = pid_b * stride_vb + pid_h * stride_vh
    o_offset = pid_b * stride_ob + pid_h * stride_oh
    lse_offset = pid_b * stride_lseb + pid_h * stride_lseh

    offs_m = offset_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    Q_ptr = Q + q_offset + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    q_mask = (offs_m[:, None] < S)
    q = tl.load(Q_ptr, mask=q_mask, other=0.0)

    m_i = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, D), tl.float32)

    limit_full_causal = offset_m // BLOCK_N
    limit_full_s = S // BLOCK_N
    num_full_kv_tiles = tl.minimum(limit_full_causal, limit_full_s)

    K_ptr = K + k_offset + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    V_ptr = V + v_offset + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd

    for kv_tile in tl.range(0, num_full_kv_tiles, num_stages=3):
        k = tl.load(K_ptr)
        v = tl.load(V_ptr)

        scores_b2 = tl.dot(q, k.T) * SCALE
        
        m_ij = tl.maximum(m_i, tl.max(scores_b2, axis=1))
        alpha = tl.math.exp2(m_i - m_ij)
        p = tl.math.exp2(scores_b2 - m_ij[:, None])

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        
        m_i = m_ij

        K_ptr += BLOCK_N * stride_ks
        V_ptr += BLOCK_N * stride_vs

    limit_n = S if S < offset_m + BLOCK_M else offset_m + BLOCK_M
    num_total_kv_tiles = (limit_n + BLOCK_N - 1) // BLOCK_N

    for kv_tile in range(num_full_kv_tiles, num_total_kv_tiles):
        offset_n = kv_tile * BLOCK_N
        offs_n_curr = offset_n + offs_n
        
        k_mask = offs_n_curr[:, None] < S
        v_mask = offs_n_curr[:, None] < S
        
        k = tl.load(K_ptr, mask=k_mask, other=0.0)
        v = tl.load(V_ptr, mask=v_mask, other=0.0)

        scores = tl.dot(q, k.T)
        
        valid = (offs_m[:, None] >= offs_n_curr[None, :]) & (offs_n_curr[None, :] < S)
        scores_b2 = tl.where(valid, scores * SCALE, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(scores_b2, axis=1))
        safe_m_ij = tl.where(m_ij == float("-inf"), 0.0, m_ij)

        alpha = tl.math.exp2(m_i - safe_m_ij)
        p = tl.math.exp2(scores_b2 - safe_m_ij[:, None])

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        
        m_i = m_ij

        K_ptr += BLOCK_N * stride_ks
        V_ptr += BLOCK_N * stride_vs

    safe_l_i = tl.where(l_i == 0.0, 1.0, l_i)
    out = acc / safe_l_i[:, None]

    LN2 = 0.6931471805599453
    lse = tl.where(l_i == 0.0, float("-inf"), (m_i + tl.math.log2(safe_l_i)) * LN2)

    O_ptr = O + o_offset + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(O_ptr, out.to(tl.bfloat16), mask=q_mask)

    LSE_ptr = LSE + lse_offset + offs_m * stride_lses
    tl.store(LSE_ptr, lse, mask=(offs_m < S))


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    sm_scale = 1.0 / (D ** 0.5)
    SCALE = sm_scale * 1.4426950408889634
    
    BLOCK_M = 128
    BLOCK_N = 64
    
    grid = (triton.cdiv(S, BLOCK_M), B, H)
    
    _attn_fwd_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        B, H, S,
        SCALE,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        D=128,
        num_warps=8,
        num_stages=4
    )