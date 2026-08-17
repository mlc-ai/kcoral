import torch
import triton
import triton.language as tl
import math

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
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
    B, H, S, sm_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    # Offsets and masks for queries
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    q_mask = offs_m < S

    # Load Q
    q_ptrs = Q + (pid_b * stride_qb + pid_h * stride_qh) + offs_m[:, None] * stride_qs + tl.arange(0, BLOCK_D)[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)

    # Initialize online softmax state
    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    k_base = K + pid_b * stride_kb + pid_h * stride_kh
    v_base = V + pid_b * stride_vb + pid_h * stride_vh

    kv_tiles = tl.cdiv(S, BLOCK_N)
    for kv_tile in range(0, kv_tiles):
        offs_n = kv_tile * BLOCK_N + tl.arange(0, BLOCK_N)
        kv_mask = offs_n < S

        # Load K (transposed logically via pointer arithmetic: shape [BLOCK_D, BLOCK_N])
        k_ptrs = k_base + tl.arange(0, BLOCK_D)[:, None] * stride_kd + offs_n[None, :] * stride_ks
        k = tl.load(k_ptrs, mask=kv_mask[None, :], other=0.0)

        # Load V (shape [BLOCK_N, BLOCK_D])
        v_ptrs = v_base + offs_n[:, None] * stride_vs + tl.arange(0, BLOCK_D)[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=kv_mask[:, None], other=0.0)

        # Compute Q @ K^T
        qk = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        qk = tl.dot(q, k, acc=qk)
        qk = qk * sm_scale
        
        # Apply masks to ignore padded keys and queries
        valid_mask = q_mask[:, None] & kv_mask[None, :]
        qk = tl.where(valid_mask, qk, -float("inf"))

        # Row-wise max and normalization
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        safe_m_ij = tl.where(m_ij == -float("inf"), 0.0, m_ij)
        
        alpha = tl.math.exp2(m_i - safe_m_ij)
        p = tl.math.exp2(qk - safe_m_ij[:, None])

        # Accumulate softmax sum and update running values
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc=acc)
        m_i = m_ij

    # Final normalization
    safe_l_i = tl.where(l_i == 0.0, 1.0, l_i)
    output = acc / safe_l_i[:, None]
    
    # Compute LSE in natural log (fp32)
    lse_log2 = tl.where(l_i == 0.0, -float("inf"), m_i + tl.math.log2(safe_l_i))
    LN2 = 0.6931471805599453
    lse = lse_log2 * LN2

    # Store output O
    o_ptrs = O + (pid_b * stride_ob + pid_h * stride_oh) + offs_m[:, None] * stride_os + tl.arange(0, BLOCK_D)[None, :] * stride_od
    tl.store(o_ptrs, output.to(tl.bfloat16), mask=q_mask[:, None])

    # Store LSE
    lse_ptrs = LSE + (pid_b * stride_lseb + pid_h * stride_lseh) + offs_m * stride_lses
    tl.store(lse_ptrs, lse, mask=q_mask)

def run(Q, K, V, O, LSE):
    """
    Computes standard non-causal multi-head attention forward mapping.
    Q, K, V: [B, H, S, D], bfloat16
    O:       [B, H, S, D], bfloat16
    LSE:     [B, H, S],    float32
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    
    # Apply standard FlashAttention log2(e) trick alongside sequence scaling
    RCP_LN2 = 1.4426950408889634
    sm_scale = (1.0 / math.sqrt(D)) * RCP_LN2

    grid = lambda META: (triton.cdiv(S, META["BLOCK_M"]), B, H)
    
    _fwd_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        B, H, S, sm_scale,
        BLOCK_D=D
    )