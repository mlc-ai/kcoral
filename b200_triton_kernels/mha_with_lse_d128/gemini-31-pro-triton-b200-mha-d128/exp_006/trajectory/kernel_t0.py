import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=3),
    ],
    key=["S"],
)
@triton.jit
def _attention_kernel(
    Q, K, V, O, LSE,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lb, stride_lh, stride_ls,
    S,
    softmax_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    q_base = Q + pid_b * stride_qb + pid_h * stride_qh
    k_base = K + pid_b * stride_kb + pid_h * stride_kh
    v_base = V + pid_b * stride_vb + pid_h * stride_vh
    o_base = O + pid_b * stride_ob + pid_h * stride_oh
    lse_base = LSE + pid_b * stride_lb + pid_h * stride_lh

    # Mask for queries across sequence length S and head dimension D
    mask_q = (offs_m[:, None] < S) & (offs_d[None, :] < BLOCK_D)
    q_ptrs = q_base + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=mask_q, other=0.0)

    # Convert scale and natural log to base-2 log upfront to exploit tl.exp2
    RCP_LN2: tl.constexpr = 1.4426950408889634
    q_scaled = (q * (softmax_scale * RCP_LN2)).to(q.dtype)

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)

    for k_idx in range(0, tl.cdiv(S, BLOCK_N)):
        offs_n = k_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        
        mask_k = (offs_n[:, None] < S) & (offs_d[None, :] < BLOCK_D)
        k_ptrs = k_base + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        v_ptrs = v_base + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
        
        k = tl.load(k_ptrs, mask=mask_k, other=0.0)
        v = tl.load(v_ptrs, mask=mask_k, other=0.0)
        
        scores = tl.dot(q_scaled, k.T)
        
        # Mask out-of-bounds keys and queries
        valid_score = (offs_m[:, None] < S) & (offs_n[None, :] < S)
        scores = tl.where(valid_score, scores, float("-inf"))
        
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        safe_m_ij = tl.where(m_ij == float("-inf"), 0.0, m_ij)
        
        alpha = tl.exp2(m_i - safe_m_ij)
        p = tl.exp2(scores - safe_m_ij[:, None])
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        # Dot inputs must have compatible types, accumulate in fp32
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_ij

    safe_l_i = tl.where(l_i == 0.0, 1.0, l_i)
    out = acc / safe_l_i[:, None]
    
    # Convert base-2 state back to base-e natural logarithm for LSE
    LN2: tl.constexpr = 0.6931471805599453
    lse = (m_i + tl.log2(safe_l_i)) * LN2
    lse = tl.where(l_i == 0.0, float("-inf"), lse)

    # Store finalized output
    o_ptrs = o_base + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(o_ptrs, out.to(q.dtype), mask=mask_q)

    # Store Log-Sum-Exp statistics
    lse_ptrs = lse_base + offs_m * stride_ls
    tl.store(lse_ptrs, lse, mask=(offs_m < S))


def run(Q, K, V, O, LSE):
    """
    Computes a Multi-Head Attention forward pass returning output vectors and Log-Sum-Exp (LSE).
    Targeting NVIDIA Blackwell architecture.
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    softmax_scale = 1.0 / (D ** 0.5)

    # Define execution grid mapping to query tiles and (batch, head) loops
    grid = lambda META: (triton.cdiv(S, META["BLOCK_M"]), B, H)
    
    _attention_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        S,
        softmax_scale,
        BLOCK_D=128
    )