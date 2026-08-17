import torch
import triton
import triton.language as tl
import math

def get_autotune_config():
    return [
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
    ]

@triton.autotune(
    configs=get_autotune_config(),
    key=['S'],
)
@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale,
    O, LSE,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lse_b, stride_lse_h, stride_lse_s,
    S,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    # Offset calculations for the current block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q_offset = pid_b * stride_qb + pid_h * stride_qh
    k_offset = pid_b * stride_kb + pid_h * stride_kh
    v_offset = pid_b * stride_vb + pid_h * stride_vh

    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    k_ptrs = K + k_offset + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = V + v_offset + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd

    # Load Q tile
    q_mask = offs_m[:, None] < S
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # Initialize running state
    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

    # Limit key tiles to respect the causal mask and sequence length
    max_kv_tiles = tl.cdiv((pid_m + 1) * BLOCK_M, BLOCK_N)
    total_kv_tiles = tl.cdiv(S, BLOCK_N)
    num_kv_tiles = max_kv_tiles
    if total_kv_tiles < max_kv_tiles:
        num_kv_tiles = total_kv_tiles

    for kv_tile in range(0, num_kv_tiles):
        kv_indices = kv_tile * BLOCK_N + offs_n
        valid_kv = kv_indices < S
        k_mask = valid_kv[:, None]
        
        # Load K and V tiles
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)
        v = tl.load(v_ptrs, mask=k_mask, other=0.0)

        # Compute Q @ K^T
        scores = tl.dot(q, k.T, out_dtype=tl.float32) * sm_scale
        
        # Apply causal mask and padding mask
        causal_mask = (offs_m[:, None] >= kv_indices[None, :]) & (offs_m[:, None] < S)
        # Convert to base-2 for fast exponentials
        scores = tl.where(causal_mask, scores * 1.4426950408889634, float("-inf"))

        # Online Softmax
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        safe_m_ij = tl.where(m_ij == float("-inf"), 0.0, m_ij)

        alpha = tl.exp2(m_i - safe_m_ij)
        p = tl.exp2(scores - safe_m_ij[:, None])

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        
        # Accumulate P @ V
        acc = tl.dot(p.to(tl.bfloat16), v, acc)

        m_i = m_ij

        # Advance pointers along the sequence dimension
        k_ptrs += BLOCK_N * stride_ks
        v_ptrs += BLOCK_N * stride_vs

    # Normalize output
    safe_l_i = tl.where(l_i == 0.0, 1.0, l_i)
    output = acc / safe_l_i[:, None]
    
    # Compute LSE (converting back to natural log)
    lse_log2 = tl.where(l_i == 0.0, float("-inf"), m_i + tl.log2(safe_l_i))
    lse = lse_log2 * 0.6931471805599453

    # Store Attention Output
    o_offset = pid_b * stride_ob + pid_h * stride_oh
    o_ptrs = O + o_offset + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(o_ptrs, output.to(tl.bfloat16), mask=q_mask)

    # Store LSE
    lse_offset = pid_b * stride_lse_b + pid_h * stride_lse_h
    lse_ptrs = LSE + lse_offset + offs_m * stride_lse_s
    lse_mask = offs_m < S
    tl.store(lse_ptrs, lse, mask=lse_mask)

def run(Q, K, V, O, LSE):
    """
    Computes Causal Multi-Head Attention forward pass returning O and LSE.
    """
    torch.cuda.set_device(Q.device)
    
    B = Q.shape[0]
    H = Q.shape[1]
    S = Q.shape[2]
    D = Q.shape[3]

    sm_scale = 1.0 / math.sqrt(D)
    
    # Launch configuration
    grid = lambda META: (triton.cdiv(S, META['BLOCK_M']), B, H)
    
    _fwd_kernel[grid](
        Q, K, V, sm_scale,
        O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        S,
        HEAD_DIM=D
    )