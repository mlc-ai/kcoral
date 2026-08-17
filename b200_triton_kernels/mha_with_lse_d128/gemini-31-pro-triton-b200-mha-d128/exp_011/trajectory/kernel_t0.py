import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=4),
    ],
    key=["S"],
)
@triton.jit
def _attention_kernel(
    Q, K, V, O, LSE,
    S,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Map program ids to coordinates
    batch_idx = tl.program_id(1)
    head_idx = tl.program_id(2)
    m_block_idx = tl.program_id(0)

    # Base offsets for this batch/head
    q_offset = batch_idx * stride_qb + head_idx * stride_qh
    k_offset = batch_idx * stride_kb + head_idx * stride_kh
    v_offset = batch_idx * stride_vb + head_idx * stride_vh
    o_offset = batch_idx * stride_ob + head_idx * stride_oh
    lse_offset = batch_idx * stride_lseb + head_idx * stride_lseh

    offs_m = m_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, 128)

    # Initialize pointers and load Q
    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    q_mask = offs_m[:, None] < S
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # Base pointers for K and V (we will add sequence offset in the loop)
    k_base_ptrs = K + k_offset + offs_d[None, :] * stride_kd
    v_base_ptrs = V + v_offset + offs_d[None, :] * stride_vd

    # Log constants
    RCP_LN2: tl.constexpr = 1.4426950408889634
    LN2: tl.constexpr = 0.6931471805599453

    # Initialize online softmax state
    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, 128), tl.float32)

    # Stream over K and V
    for n_block_idx in range(0, tl.cdiv(S, BLOCK_N)):
        offs_n = n_block_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        
        k_ptrs = k_base_ptrs + offs_n[:, None] * stride_ks
        v_ptrs = v_base_ptrs + offs_n[:, None] * stride_vs
        
        k_mask = offs_n[:, None] < S
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)
        v = tl.load(v_ptrs, mask=k_mask, other=0.0)
        
        # Dot product and scaling
        scores = tl.dot(q, k.T, out_dtype=tl.float32) * sm_scale
        
        # Apply padding mask
        valid_score = (offs_m[:, None] < S) & (offs_n[None, :] < S)
        scores = tl.where(valid_score, scores * RCP_LN2, -float("inf"))
        
        # Update row max
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        safe_m_ij = tl.where(m_ij == -float("inf"), 0.0, m_ij)
        
        # Calculate exponents
        alpha = tl.math.exp2(m_i - safe_m_ij)
        p = tl.math.exp2(scores - safe_m_ij[:, None])
        
        # Update sum and accumulator
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(q.dtype), v.to(q.dtype), acc)
        
        m_i = m_ij

    # Finalize softmax values
    safe_l_i = tl.where(l_i == 0.0, 1.0, l_i)
    output = acc / safe_l_i[:, None]
    
    # Calculate LogSumExp in base-2, then convert to base-e
    lse_log2 = tl.where(l_i == 0.0, -float("inf"), m_i + tl.math.log2(safe_l_i))
    lse_ln = lse_log2 * LN2

    # Store O
    o_ptrs = O + o_offset + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(o_ptrs, output.to(q.dtype), mask=offs_m[:, None] < S)

    # Store LSE
    lse_ptrs = LSE + lse_offset + offs_m * stride_lses
    tl.store(lse_ptrs, lse_ln, mask=offs_m < S)


def run(Q, K, V, O, LSE):
    """
    Computes a multi-head attention forward pass returning output and LogSumExp.
    """
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    # Scale by 1/sqrt(D)
    sm_scale = 1.0 / (D ** 0.5)
    
    grid = lambda META: (triton.cdiv(S, META["BLOCK_M"]), B, H)
    
    _attention_kernel[grid](
        Q, K, V, O, LSE,
        S,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        sm_scale
    )