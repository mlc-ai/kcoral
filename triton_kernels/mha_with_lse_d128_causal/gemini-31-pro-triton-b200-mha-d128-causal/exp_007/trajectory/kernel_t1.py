import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Q, K, V, O, LSE,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    S, sm_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    start_m = tl.program_id(0)
    batch = tl.program_id(1)
    head = tl.program_id(2)

    # Base pointers for the current batch and head
    q_ptrs = Q + batch * stride_qb + head * stride_qh
    k_ptrs = K + batch * stride_kb + head * stride_kh
    v_ptrs = V + batch * stride_vb + head * stride_vh
    o_ptrs = O + batch * stride_ob + head * stride_oh
    lse_ptrs = LSE + batch * stride_lseb + head * stride_lseh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    # Load Q tile
    q_mask = offs_m[:, None] < S
    q_ptrs = q_ptrs + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # Softmax scaling factor mixed with change of base logarithm for hardware native exp2 capability
    RCP_LN2: tl.constexpr = 1.4426950408889634
    scale = sm_scale * RCP_LN2

    # Running softmax variables kept safely in FP32
    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)

    k_ptrs = k_ptrs + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = v_ptrs + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd

    # Safe bound calculation to omit fully invalid keys in the causal pattern
    max_k_idx = tl.minimum(S, (start_m + 1) * BLOCK_M)
    num_k_blocks = tl.cdiv(max_k_idx, BLOCK_N)

    m_idx = offs_m
    m_mask = m_idx < S

    for k0 in range(0, num_k_blocks):
        start_n = k0 * BLOCK_N
        n_idx = start_n + offs_n
        mask_n = n_idx[:, None] < S

        # Load Keys and Values
        k = tl.load(k_ptrs, mask=mask_n, other=0.0)
        v = tl.load(v_ptrs, mask=mask_n, other=0.0)

        # Unscaled raw QK^T exact block dot accumulation and then FP32 precise scaling
        qk = tl.dot(q, k.T)
        qk = qk * scale

        # Establish causally correct masks (only attending when query_pos >= key_pos in valid Sequence lengths)
        valid_score = m_mask[:, None] & (m_idx[:, None] >= n_idx[None, :])
        qk = tl.where(valid_score, qk, -float("inf"))

        # Block-wise online max and rescaling
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        safe_m_ij = tl.where(m_ij == -float("inf"), 0.0, m_ij)

        # Exponentiate difference utilizing standard EX2 hardware instructions
        alpha = tl.math.exp2(m_i - safe_m_ij)
        p = tl.math.exp2(qk - safe_m_ij[:, None])

        # Step tracking LSE norm divisor and current attention context vector sum 
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        
        m_i = m_ij

        # Advance block pointers
        k_ptrs += BLOCK_N * stride_ks
        v_ptrs += BLOCK_N * stride_vs

    # Finalize context normalization safely guarding fully masked queries against NaNs (0/0)
    safe_l_i = tl.where(l_i == 0.0, 1.0, l_i)
    out = acc / safe_l_i[:, None]

    # Revert max/l_i from base-2 back to natural logarithm base for returning conventional LSE 
    LN2: tl.constexpr = 0.6931471805599453
    lse_log2 = tl.where(l_i == 0.0, -float("inf"), m_i + tl.math.log2(safe_l_i))
    lse = lse_log2 * LN2

    # Store normalized context vector
    o_ptrs = o_ptrs + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(o_ptrs, out.to(tl.bfloat16), mask=q_mask)

    # Store finalized natural-logarithmic LSE
    lse_out_ptrs = lse_ptrs + offs_m * stride_lses
    tl.store(lse_out_ptrs, lse, mask=m_mask)

def run(Q, K, V, O, LSE):
    """
    Standard Triton causal multi-head attention forward computing Output and Natural-Log-Sum-Exp.
    Writes entirely in-place to preallocated 'O' and 'LSE'.
    """
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    sm_scale = 1.0 / (D ** 0.5)

    BLOCK_M = 128
    BLOCK_N = 64

    grid = (triton.cdiv(S, BLOCK_M), B, H)
    _fwd_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        S, sm_scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=128,
        num_warps=8, num_stages=3
    )