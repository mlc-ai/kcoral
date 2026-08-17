import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
    ],
    key=["S"],
)
@triton.jit
def _attn_fwd_kernel(
    Q, K, V, O, LSE,
    S,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    softmax_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < S

    q_base = Q + pid_b * stride_qb + pid_h * stride_qh
    k_base = K + pid_b * stride_kb + pid_h * stride_kh
    v_base = V + pid_b * stride_vb + pid_h * stride_vh
    o_base = O + pid_b * stride_ob + pid_h * stride_oh
    lse_base = LSE + pid_b * stride_lseb + pid_h * stride_lseh

    offs_d = tl.arange(0, BLOCK_D)

    # Load Q tile once
    q_ptrs = q_base + (offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd)
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    RCP_LN2 = 1.4426950408889634
    scale = softmax_scale * RCP_LN2

    S_blocks = tl.cdiv(S, BLOCK_N)

    # Fast path: Loop over guaranteed full KV blocks
    for kv_idx in range(0, S_blocks - 1):
        offs_n = kv_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        
        k_ptrs = k_base + (offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd)
        v_ptrs = v_base + (offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd)
        
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)
        
        scores = tl.dot(q, k.T, out_dtype=tl.float32)
        scores = scores * scale
        
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        safe_m_ij = tl.where(m_ij == -float("inf"), 0.0, m_ij)
        
        alpha = tl.math.exp2(m_i - safe_m_ij)
        p = tl.math.exp2(scores - safe_m_ij[:, None])
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        m_i = m_ij

    # Last KV block (might be partial)
    if S_blocks > 0:
        kv_idx = S_blocks - 1
        offs_n = kv_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S
        
        k_ptrs = k_base + (offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd)
        v_ptrs = v_base + (offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd)
        
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        
        scores = tl.dot(q, k.T, out_dtype=tl.float32)
        scores = scores * scale
        
        # Mask out-of-bounds keys for valid query evaluation
        scores = tl.where(mask_n[None, :], scores, -float("inf"))
        
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        safe_m_ij = tl.where(m_ij == -float("inf"), 0.0, m_ij)
        
        alpha = tl.math.exp2(m_i - safe_m_ij)
        p = tl.math.exp2(scores - safe_m_ij[:, None])
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        m_i = m_ij

    # Epilogue
    safe_l_i = tl.where(l_i == 0.0, 1.0, l_i)
    out = acc / safe_l_i[:, None]
    lse_log2 = tl.where(l_i == 0.0, -float("inf"), m_i + tl.math.log2(safe_l_i))
    
    LN2 = 0.6931471805599453
    lse_ln = lse_log2 * LN2

    # Store normalized output and LSE for the active query chunk
    o_ptrs = o_base + (offs_m[:, None] * stride_os + offs_d[None, :] * stride_od)
    tl.store(o_ptrs, out.to(tl.bfloat16), mask=mask_m[:, None])

    lse_ptrs = lse_base + offs_m * stride_lses
    tl.store(lse_ptrs, lse_ln, mask=mask_m)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    softmax_scale = 1.0 / (D ** 0.5)

    grid = lambda META: (triton.cdiv(S, META["BLOCK_M"]), B, H)
    
    _attn_fwd_kernel[grid](
        Q, K, V, O, LSE,
        S,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        softmax_scale,
        BLOCK_D=128,
    )