import torch
import triton
import triton.language as tl

def get_autotune_config():
    return [
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
    ]

@triton.autotune(
    configs=get_autotune_config(),
    key=['S']
)
@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale,
    O, LSE,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    S,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    start_m = tl.program_id(0) * BLOCK_M
    off_b = tl.program_id(1)
    off_h = tl.program_id(2)

    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    # Initialize pointers
    q_offset = off_b * stride_qb + off_h * stride_qh
    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd

    k_offset = off_b * stride_kb + off_h * stride_kh
    k_ptrs = K + k_offset + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd

    v_offset = off_b * stride_vb + off_h * stride_vh
    v_ptrs = V + v_offset + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd

    q_mask_1d = offs_m < S
    q_mask_2d = offs_m[:, None] < S

    # Load query block
    q = tl.load(q_ptrs, mask=q_mask_2d, other=0.0)

    # Initialize running reduction state
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # Determine limit for this query block (capping keys to min(S, start_m + BLOCK_M))
    end_m = start_m + BLOCK_M
    seq_limit = tl.minimum(S, end_m)
    num_steps = (seq_limit + BLOCK_N - 1) // BLOCK_N

    for start_n_idx in range(0, num_steps):
        start_n = start_n_idx * BLOCK_N
        
        # Load keys and values
        mask_n = (start_n + offs_n)[:, None] < S
        k = tl.load(k_ptrs, mask=mask_n, other=0.0)
        v = tl.load(v_ptrs, mask=mask_n, other=0.0)
        
        # Compute qk dot product
        qk = tl.dot(q, tl.trans(k)).to(tl.float32)
        qk = qk * sm_scale
        
        # Apply causal masking on block diagonal
        if start_n + BLOCK_N > start_m:
            causal_mask = (start_n + offs_n)[None, :] <= offs_m[:, None]
            qk = tl.where(causal_mask, qk, float("-inf"))
            
        # Mask out-of-bounds keys based on sequence length
        if start_n + BLOCK_N > S:
            valid_mask = (start_n + offs_n)[None, :] < S
            qk = tl.where(valid_mask, qk, float("-inf"))
            
        # Enforce queries strictly beyond boundary are kept clear from polluting LSE
        qk = tl.where(q_mask_2d, qk, float("-inf"))
        
        # Safe FlashAttention logic
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])
        
        # Rescale accumulator and perform RS-GEMM update
        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(q.dtype), v)
        
        # Finalize this block's stats
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        
        # Advance pointers
        k_ptrs += BLOCK_N * stride_ks
        v_ptrs += BLOCK_N * stride_vs

    # Epilogue: normalize with log-sum-exp
    acc = acc / l_i[:, None]
    lse = m_i + tl.log(l_i)

    # Store to allocated outputs
    o_offset = off_b * stride_ob + off_h * stride_oh
    o_ptrs = O + o_offset + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(q.dtype), mask=q_mask_2d)

    lse_offset = off_b * stride_lseb + off_h * stride_lseh
    lse_ptrs = LSE + lse_offset + offs_m * stride_lses
    tl.store(lse_ptrs, lse, mask=q_mask_1d)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    sm_scale = 1.0 / (D ** 0.5)

    grid = lambda META: (
        triton.cdiv(S, META['BLOCK_M']),
        B,
        H
    )

    _fwd_kernel[grid](
        Q, K, V, sm_scale,
        O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        S,
        BLOCK_D=D,
    )