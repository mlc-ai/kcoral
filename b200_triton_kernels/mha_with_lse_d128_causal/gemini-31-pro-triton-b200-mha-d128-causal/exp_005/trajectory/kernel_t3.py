import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
    ],
    key=['S']
)
@triton.jit
def _mha_causal_fwd_kernel(
    Q, K, V, O, LSE,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    stride_lseb, stride_lseh, stride_lsem,
    S,
    BLOCK_D: tl.constexpr,
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    start_m = pid_m * BLOCK_M
    if start_m >= S:
        return

    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    batch_head_offset_q = pid_b * stride_qb + pid_h * stride_qh
    batch_head_offset_k = pid_b * stride_kb + pid_h * stride_kh
    batch_head_offset_v = pid_b * stride_vb + pid_h * stride_vh

    q_ptrs = Q + batch_head_offset_q + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    k_ptrs = K + batch_head_offset_k + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
    v_ptrs = V + batch_head_offset_v + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd

    q_mask = offs_m[:, None] < S
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)
    
    # Pre-scale Q in base-2 units outside the loop
    # scale = 1 / sqrt(128) = 0.08838834764831845
    # scale_ln2 = scale * log2(e) = 0.12752189912185208
    scale_ln2: tl.constexpr = 0.12752189912185208
    q = (q * scale_ln2).to(tl.bfloat16)

    acc_o = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)

    limit_full = tl.minimum(start_m, (S // BLOCK_N) * BLOCK_N)
    
    # 1. Full blocks (No causal mask or sequence bounds mask needed, safe to avoid safe_m_ij check)
    for start_n in range(0, limit_full, BLOCK_N):
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)
        
        scores = tl.dot(q, k.T)
        
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.exp2(m_i - m_ij)
        p = tl.exp2(scores - m_ij[:, None])
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc_o = acc_o * alpha[:, None]
        acc_o = tl.dot(p.to(tl.bfloat16), v, acc_o)
        
        m_i = m_ij
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn

    # 2. Causal / Partial blocks
    limit_total = tl.minimum(start_m + BLOCK_M, ((S + BLOCK_N - 1) // BLOCK_N) * BLOCK_N)
    for start_n in range(limit_full, limit_total, BLOCK_N):
        offs_n_curr = start_n + offs_n
        k_mask = offs_n_curr[:, None] < S
        
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)
        v = tl.load(v_ptrs, mask=k_mask, other=0.0)
        
        scores = tl.dot(q, k.T)
        
        # Apply causal mask and sequence boundary check
        valid_mask = (offs_n_curr[None, :] <= offs_m[:, None]) & (offs_n_curr[None, :] < S)
        scores = tl.where(valid_mask, scores, -float('inf'))
        
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        safe_m_ij = tl.where(m_ij == -float('inf'), 0.0, m_ij)
        alpha = tl.exp2(m_i - safe_m_ij)
        p = tl.exp2(scores - safe_m_ij[:, None])
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc_o = acc_o * alpha[:, None]
        acc_o = tl.dot(p.to(tl.bfloat16), v, acc_o)
        
        m_i = m_ij
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn

    # Finalize and write output O
    safe_l_i = tl.where(l_i == 0.0, 1.0, l_i)
    out = acc_o / safe_l_i[:, None]
    
    batch_head_offset_o = pid_b * stride_ob + pid_h * stride_oh
    o_ptrs = O + batch_head_offset_o + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, out.to(tl.bfloat16), mask=q_mask)

    # Convert LogSumExp to natural log scale and write output LSE
    LN2: tl.constexpr = 0.6931471805599453
    lse = (m_i + tl.log2(safe_l_i)) * LN2
    lse = tl.where(l_i == 0.0, -float('inf'), lse)

    batch_head_offset_lse = pid_b * stride_lseb + pid_h * stride_lseh
    lse_ptrs = LSE + batch_head_offset_lse + offs_m * stride_lsem
    tl.store(lse_ptrs, lse, mask=offs_m < S)


def run(Q, K, V, O, LSE):
    """Compute causal FlashAttention forward and store in O and LSE."""
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
        BLOCK_D=128
    )