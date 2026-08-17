import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=4),
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

    # Base pointers for the current batch and head
    q_base = Q + pid_b * stride_qb + pid_h * stride_qh
    k_base = K + pid_b * stride_kb + pid_h * stride_kh
    v_base = V + pid_b * stride_vb + pid_h * stride_vh

    # Device-created tensor descriptors for Blackwell TMA access
    q_desc = tl.make_tensor_descriptor(
        q_base,
        shape=[S, BLOCK_D],
        strides=[stride_qm, 1],
        block_shape=[BLOCK_M, BLOCK_D],
        padding_option="zero"
    )
    k_desc = tl.make_tensor_descriptor(
        k_base,
        shape=[S, BLOCK_D],
        strides=[stride_kn, 1],
        block_shape=[BLOCK_N, BLOCK_D],
        padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        v_base,
        shape=[S, BLOCK_D],
        strides=[stride_vn, 1],
        block_shape=[BLOCK_N, BLOCK_D],
        padding_option="zero"
    )

    q = q_desc.load([start_m, 0])

    acc_o = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # Combine sqrt(d) scaling and base-2 exponential scaling
    # scale = 1 / sqrt(128) * log2(e) = 0.08838834764831845 * 1.4426950408889634
    scale_ln2: tl.constexpr = 0.12752189912185208

    # Limit for full blocks where no causal mask is needed
    limit_full = tl.minimum(start_m, (S // BLOCK_N) * BLOCK_N)
    
    # 1. Full blocks loop (bypasses valid bounds/causal mask instruction overhead)
    for start_n in range(0, limit_full, BLOCK_N):
        k = k_desc.load([start_n, 0])
        v = v_desc.load([start_n, 0])
        
        acc_scores = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        scores = tl.dot(q, k.T, acc_scores) * scale_ln2
        
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        # Since it's a full block, finite non-inf scores are guaranteed. No -inf safeguard is needed.
        alpha = tl.math.exp2(m_i - m_ij)
        p = tl.math.exp2(scores - m_ij[:, None])
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc_o = acc_o * alpha[:, None]
        acc_o = tl.dot(p.to(tl.bfloat16), v, acc_o)
        
        m_i = m_ij

    # 2. Partial / Causal blocks loop (includes bounds checks and bounds mask)
    limit_total = tl.minimum(start_m + BLOCK_M, ((S + BLOCK_N - 1) // BLOCK_N) * BLOCK_N)
    for start_n in range(limit_full, limit_total, BLOCK_N):
        k = k_desc.load([start_n, 0])
        v = v_desc.load([start_n, 0])
        
        acc_scores = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        scores = tl.dot(q, k.T, acc_scores) * scale_ln2
        
        offs_n_curr = start_n + offs_n
        valid_mask = (offs_n_curr[None, :] <= offs_m[:, None]) & (offs_n_curr[None, :] < S) & (offs_m[:, None] < S)
        scores = tl.where(valid_mask, scores, -float('inf'))
        
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        safe_m_ij = tl.where(m_ij == -float('inf'), 0.0, m_ij)
        alpha = tl.math.exp2(m_i - safe_m_ij)
        p = tl.math.exp2(scores - safe_m_ij[:, None])
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc_o = acc_o * alpha[:, None]
        acc_o = tl.dot(p.to(tl.bfloat16), v, acc_o)
        
        m_i = m_ij

    # Normalize output and write
    safe_l_i = tl.where(l_i == 0.0, 1.0, l_i)
    out = acc_o / safe_l_i[:, None]
    
    o_base = O + pid_b * stride_ob + pid_h * stride_oh
    o_ptrs = o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    q_mask = offs_m[:, None] < S
    tl.store(o_ptrs, out.to(tl.bfloat16), mask=q_mask)

    # Calculate LSE and log2 -> natural log transformation
    LN2: tl.constexpr = 0.6931471805599453
    lse = (m_i + tl.math.log2(safe_l_i)) * LN2
    lse = tl.where(l_i == 0.0, -float('inf'), lse)

    lse_base = LSE + pid_b * stride_lseb + pid_h * stride_lseh
    lse_ptrs = lse_base + offs_m * stride_lsem
    tl.store(lse_ptrs, lse, mask=offs_m < S)


def alloc_fn(size: int, alignment: int, stream):
    """Infrastructure storage allocator required for standard Triton device descriptors."""
    return torch.empty(size, device="cuda", dtype=torch.int8)


def run(Q, K, V, O, LSE):
    """Compute causal FlashAttention forward and store in O and LSE."""
    torch.cuda.set_device(Q.device)
    triton.set_allocator(alloc_fn)
    
    B, H, S, D = Q.shape
    if S == 0:
        return
    
    # Kernel grid definition dynamically adjusting to config blocks per autotuning iteration
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