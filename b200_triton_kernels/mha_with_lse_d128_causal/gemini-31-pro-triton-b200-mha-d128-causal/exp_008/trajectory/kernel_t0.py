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
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    # Calculate pointers to the current batch and head
    q_offset = pid_b * stride_qb + pid_h * stride_qh
    k_offset = pid_b * stride_kb + pid_h * stride_kh
    v_offset = pid_b * stride_vb + pid_h * stride_vh
    o_offset = pid_b * stride_ob + pid_h * stride_oh
    lse_offset = pid_b * stride_lseb + pid_h * stride_lseh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    # Initialize data pointers
    Q_ptr = Q + q_offset + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    K_ptr = K + k_offset + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    V_ptr = V + v_offset + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd

    # Load query with masking
    q_mask = (offs_m[:, None] < S) & (offs_d[None, :] < D)
    q = tl.load(Q_ptr, mask=q_mask, other=0.0)

    # Initialize running softmax state and output accumulator
    m_i = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, D), tl.float32)

    RCP_LN2 = 1.4426950408889634

    # Optimization: cap the maximum sequence key index we need to visit to avoid causal padding
    limit = (pid_m + 1) * BLOCK_M
    max_n = S if S < limit else limit
    num_kv_tiles = (max_n + BLOCK_N - 1) // BLOCK_N

    for kv_tile in range(0, num_kv_tiles):
        n_start = kv_tile * BLOCK_N
        offs_n_curr = n_start + offs_n

        k_mask = (offs_n_curr[:, None] < S) & (offs_d[None, :] < D)
        v_mask = (offs_n_curr[:, None] < S) & (offs_d[None, :] < D)

        K_curr = K_ptr + n_start * stride_ks
        V_curr = V_ptr + n_start * stride_vs

        # Load keys and values
        k = tl.load(K_curr, mask=k_mask, other=0.0)
        v = tl.load(V_curr, mask=v_mask, other=0.0)

        # Matmul query and keys
        scores = tl.dot(q, k.T) * sm_scale
        
        # Causal and padding mask
        valid_mask = (offs_m[:, None] >= offs_n_curr[None, :]) & (offs_m[:, None] < S)
        scores_b2 = tl.where(valid_mask, scores * RCP_LN2, float("-inf"))

        # Row max update
        m_ij = tl.maximum(m_i, tl.max(scores_b2, axis=1))
        
        # Guard against -inf - (-inf) resulting in NaN when full rows are masked
        safe_m_ij = tl.where(m_ij == float("-inf"), 0.0, m_ij)

        alpha = tl.exp2(m_i - safe_m_ij)
        p = tl.exp2(scores_b2 - safe_m_ij[:, None])

        # State updates
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        
        m_i = m_ij

    # Epilogue normalization
    safe_l_i = tl.where(l_i == 0.0, 1.0, l_i)
    out = acc / safe_l_i[:, None]

    # Compute base-e LSE required by the contract
    LN2 = 0.6931471805599453
    lse_base2 = tl.where(l_i == 0.0, float("-inf"), m_i + tl.log2(safe_l_i))
    lse = lse_base2 * LN2

    # Writeback outputs
    O_ptr = O + o_offset + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(O_ptr, out.to(tl.bfloat16), mask=q_mask)

    LSE_ptr = LSE + lse_offset + offs_m * stride_lses
    tl.store(LSE_ptr, lse, mask=(offs_m < S))


def run(Q, K, V, O, LSE):
    """
    Computes causal multi-head attention forward.
    Receives all input tensors (Q, K, V) followed by output tensors (O, LSE).
    """
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    # Scale applied prior to exponential
    sm_scale = 1.0 / (D ** 0.5)
    
    # Tuned block dimensions for robust SM100 standard pointer access
    BLOCK_M = 64
    BLOCK_N = 64
    
    # Grid definition: mapping M sequence blocks, Batches, and Heads
    grid = (triton.cdiv(S, BLOCK_M), B, H)
    
    _attn_fwd_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        B, H, S,
        sm_scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        D=128,
        num_warps=4,
        num_stages=3
    )