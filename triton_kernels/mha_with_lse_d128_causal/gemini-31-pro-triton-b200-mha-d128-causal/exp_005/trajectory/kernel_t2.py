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
    if start_m >= S:
        return

    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    # Base pointers for batch and head
    q_base = Q + pid_b * stride_qb + pid_h * stride_qh
    k_base = K + pid_b * stride_kb + pid_h * stride_kh
    v_base = V + pid_b * stride_vb + pid_h * stride_vh
    
    # Tensor descriptors for TMA loads
    q_desc = tl.make_tensor_descriptor(
        q_base,
        shape=[S, BLOCK_D],
        strides=[stride_qm, 1],
        block_shape=[BLOCK_M, BLOCK_D],
        padding_option="zero",
    )
    k_desc = tl.make_tensor_descriptor(
        k_base,
        shape=[S, BLOCK_D],
        strides=[stride_kn, 1],
        block_shape=[BLOCK_N, BLOCK_D],
        padding_option="zero",
    )
    v_desc = tl.make_tensor_descriptor(
        v_base,
        shape=[S, BLOCK_D],
        strides=[stride_vn, 1],
        block_shape=[BLOCK_N, BLOCK_D],
        padding_option="zero",
    )

    # Load Q block via TMA
    q = q_desc.load([start_m, 0])
    
    # Precomputed constant for scale: 1 / sqrt(128) * log2(e)
    scale_ln2: tl.constexpr = 0.12752189912185208

    acc_o = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)

    limit_n_full = tl.minimum(start_m, (S // BLOCK_N) * BLOCK_N)
    
    # 1. Full blocks loop (No causal mask or sequence bounds mask needed)
    for start_n in range(0, limit_n_full, BLOCK_N):
        k = k_desc.load([start_n, 0])
        v = v_desc.load([start_n, 0])
        
        # Q @ K^T in FP32
        acc_scores = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        scores = tl.dot(q, k.T, acc_scores)
        scores = scores * scale_ln2
        
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        safe_m_ij = tl.where(m_ij == -float('inf'), 0.0, m_ij)
        alpha = tl.exp2(m_i - safe_m_ij)
        p = tl.exp2(scores - safe_m_ij[:, None])
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc_o = acc_o * alpha[:, None]
        acc_o = tl.dot(p.to(tl.bfloat16), v, acc_o)
        m_i = m_ij

    # 2. Partial/Causal blocks loop
    limit_n_total = tl.minimum(start_m + BLOCK_M, ((S + BLOCK_N - 1) // BLOCK_N) * BLOCK_N)
    
    for start_n in range(limit_n_full, limit_n_total, BLOCK_N):
        k = k_desc.load([start_n, 0])
        v = v_desc.load([start_n, 0])
        
        # Q @ K^T in FP32
        acc_scores = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        scores = tl.dot(q, k.T, acc_scores)
        scores = scores * scale_ln2
        
        # Apply causal mask and sequence boundary check
        offs_n = start_n + tl.arange(0, BLOCK_N)
        valid_mask = (offs_n[None, :] <= offs_m[:, None]) & (offs_n[None, :] < S) & (offs_m[:, None] < S)
        scores = tl.where(valid_mask, scores, -float('inf'))
        
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        safe_m_ij = tl.where(m_ij == -float('inf'), 0.0, m_ij)
        alpha = tl.exp2(m_i - safe_m_ij)
        p = tl.exp2(scores - safe_m_ij[:, None])
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc_o = acc_o * alpha[:, None]
        acc_o = tl.dot(p.to(tl.bfloat16), v, acc_o)
        m_i = m_ij

    # Finalize and write output O
    safe_l_i = tl.where(l_i == 0.0, 1.0, l_i)
    out = acc_o / safe_l_i[:, None]
    
    q_mask = offs_m[:, None] < S
    o_ptrs = O + pid_b * stride_ob + pid_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, out.to(tl.bfloat16), mask=q_mask)

    # Convert LogSumExp to natural log scale and write output LSE
    LN2: tl.constexpr = 0.6931471805599453
    lse = (m_i + tl.log2(safe_l_i)) * LN2
    lse = tl.where(l_i == 0.0, -float('inf'), lse)

    lse_ptrs = LSE + pid_b * stride_lseb + pid_h * stride_lseh + offs_m * stride_lsem
    tl.store(lse_ptrs, lse, mask=offs_m < S)


# Set up allocation infrastructure storage for descriptors
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


def run(Q, K, V, O, LSE):
    """Compute causal FlashAttention forward and store in O and LSE."""
    torch.cuda.set_device(Q.device)
    triton.set_allocator(alloc_fn)
    
    B, H, S, D = Q.shape
    if S == 0:
        return
    
    # 128x64 balances compute with shared memory stages for optimal Blackwell TMEM/MMA utilization
    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_D = 128

    grid = (
        triton.cdiv(S, BLOCK_M),
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
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=8, num_stages=4
    )