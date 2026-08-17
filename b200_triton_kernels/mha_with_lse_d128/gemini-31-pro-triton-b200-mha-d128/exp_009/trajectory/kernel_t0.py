import torch
import triton
import triton.language as tl
import math

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
    ],
    key=["S"],
)
@triton.jit
def _fwd_kernel(
    Q, K, V, O, LSE,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    S,
    softmax_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
):
    # Map grid axes to query chunk, batch, and head
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    # Compute base offsets for this batch and head
    q_offset = pid_b * stride_qb + pid_h * stride_qh
    k_offset = pid_b * stride_kb + pid_h * stride_kh
    v_offset = pid_b * stride_vb + pid_h * stride_vh
    o_offset = pid_b * stride_ob + pid_h * stride_oh
    lse_offset = pid_b * stride_lseb + pid_h * stride_lseh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    # Cast to int64 for safe pointer arithmetic on large tensors
    offs_m_64 = offs_m.to(tl.int64)
    offs_d_64 = offs_d.to(tl.int64)

    # Load Q tile
    q_ptrs = Q + q_offset + offs_m_64[:, None] * stride_qs + offs_d_64[None, :] * stride_qd
    q_mask = (offs_m[:, None] < S) & (offs_d[None, :] < D)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # Math constants for base-2 exponentials
    RCP_LN2: tl.constexpr = 1.4426950408889634
    LN2: tl.constexpr = 0.6931471805599453
    neg_inf = float("-inf")
    
    # Initialize online softmax state
    m_i = tl.full((BLOCK_M,), neg_inf, tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, D), tl.float32)
    
    dtype = q.dtype

    n_tiles = tl.cdiv(S, BLOCK_N)
    for n in range(0, n_tiles):
        curr_offs_n = n * BLOCK_N + offs_n
        curr_offs_n_64 = curr_offs_n.to(tl.int64)
        
        # Load K and V tiles
        k_ptrs = K + k_offset + curr_offs_n_64[:, None] * stride_ks + offs_d_64[None, :] * stride_kd
        k_mask = (curr_offs_n[:, None] < S) & (offs_d[None, :] < D)
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)
        
        v_ptrs = V + v_offset + curr_offs_n_64[:, None] * stride_vs + offs_d_64[None, :] * stride_vd
        v_mask = (curr_offs_n[:, None] < S) & (offs_d[None, :] < D)
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)
        
        # Compute Q @ K^T
        scores = tl.dot(q, k.T, out_dtype=tl.float32) * softmax_scale
        
        # Apply causal/boundary sequence mask and convert to base-2 scale for exponentials
        valid_score = (offs_m[:, None] < S) & (curr_offs_n[None, :] < S)
        scores = tl.where(valid_score, scores * RCP_LN2, neg_inf)

        # Update running max
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        safe_m_ij = tl.where(m_ij == neg_inf, 0.0, m_ij)
        
        # Online softmax correction factor and score exponentials
        alpha = tl.exp2(m_i - safe_m_ij)
        p = tl.exp2(scores - safe_m_ij[:, None])
        
        # Update normalization sum
        l_i = l_i * alpha + tl.sum(p, axis=1)
        
        # Apply normalization to the output accumulator and compute P @ V
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(dtype), v.to(dtype), acc, out_dtype=tl.float32)
        
        m_i = m_ij

    # Safe division for completely masked rows
    safe_l_i = tl.where(l_i == 0.0, 1.0, l_i)
    output = acc / safe_l_i[:, None]
    
    # Reconstruct LSE in natural-log base consistent with PyTorch fallback
    lse_base2 = tl.where(l_i == 0.0, neg_inf, m_i + tl.log2(safe_l_i))
    lse_ln = lse_base2 * LN2

    # Store finalized output
    o_ptrs = O + o_offset + offs_m_64[:, None] * stride_os + offs_d_64[None, :] * stride_od
    tl.store(o_ptrs, output.to(dtype), mask=q_mask)

    # Store corresponding log-sum-exp
    lse_ptrs = LSE + lse_offset + offs_m_64 * stride_lses
    lse_mask = offs_m < S
    tl.store(lse_ptrs, lse_ln, mask=lse_mask)

def run(Q, K, V, O, LSE):
    """
    Computes standard Multi-Head Attention forward pass and returns LSE explicitly.
    All inputs and outputs are destination-passing semantics.
    """
    # Enforce current context logic matching
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    
    # Scale applied prior to softmax
    softmax_scale = 1.0 / math.sqrt(D)
    
    # Construct grid to cover chunks of Query sequence length
    grid = lambda META: (
        triton.cdiv(S, META["BLOCK_M"]),
        B,
        H,
    )
    
    _fwd_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        S,
        softmax_scale,
        D=D
    )