import torch
import triton
import triton.language as tl
import math

@triton.autotune(
    configs=[
        # Optimized configurations tailored for Tensor Cores and Blackwell's 228 KB shared memory
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64}, num_warps=8, num_stages=3),
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

    # Compute base pointers for this batch and head
    q_base = Q + pid_b * stride_qb + pid_h * stride_qh
    k_base = K + pid_b * stride_kb + pid_h * stride_kh
    v_base = V + pid_b * stride_vb + pid_h * stride_vh
    o_base = O + pid_b * stride_ob + pid_h * stride_oh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    
    q_mask = offs_m < S
    
    # Fully coalesced memory load of Q: [BLOCK_M, BLOCK_D]
    q_ptrs = q_base + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)

    # Initialize online softmax state accumulators globally in fp32
    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    kv_tiles = tl.cdiv(S, BLOCK_N)
    
    # Triton compiler optimally software-pipelines standard tl.load operations mapped linearly here
    for kv_tile in range(0, kv_tiles):
        offs_n = kv_tile * BLOCK_N + tl.arange(0, BLOCK_N)
        kv_mask = offs_n < S

        # Fully coalesced loads of K & V: inner physical strides align with thread dimension (stride_xd=1)
        k_ptrs = k_base + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=kv_mask[:, None], other=0.0)

        v_ptrs = v_base + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=kv_mask[:, None], other=0.0)

        # q is [BLOCK_M, BLOCK_D], k is [BLOCK_N, BLOCK_D]
        # Tensor Core handles transposes seamlessly on hardware registering logic (k.T = [BLOCK_D, BLOCK_N])
        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        qk = qk * sm_scale
        
        # Mask out padded K elements; padded Qs are structurally masked via storage mask `q_mask` exclusively
        qk = tl.where(kv_mask[None, :], qk, -float("inf"))

        # Row-wise max 
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        safe_m_ij = tl.where(m_ij == -float("inf"), 0.0, m_ij)
        
        # HW log2 base exponential integration
        alpha = tl.math.exp2(m_i - safe_m_ij)
        p = tl.math.exp2(qk - safe_m_ij[:, None])

        # Softmax denominators and dynamically re-weighted outputs
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc=acc)
        m_i = m_ij

    # Final softmax normalization division
    safe_l_i = tl.where(l_i == 0.0, 1.0, l_i)
    output = acc / safe_l_i[:, None]
    
    # Recover exact FP32 Natural-Log representation for LSE sequence alignment
    lse_log2 = tl.where(l_i == 0.0, -float("inf"), m_i + tl.math.log2(safe_l_i))
    LN2 = 0.6931471805599453
    lse = lse_log2 * LN2

    # Coalesced store of O: [BLOCK_M, BLOCK_D]
    o_ptrs = o_base + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(o_ptrs, output.to(tl.bfloat16), mask=q_mask[:, None])

    # Standard contiguous line store of Log-Sum-Exp statistics
    lse_ptrs = LSE + (pid_b * stride_lseb + pid_h * stride_lseh) + offs_m * stride_lses
    tl.store(lse_ptrs, lse, mask=q_mask)

def run(Q, K, V, O, LSE):
    """
    Computes standard non-causal multi-head attention forward mapping.
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    
    # Pre-fuse FlashAttention scale parameter incorporating Triton's exponential log2(e) hardware modifier
    RCP_LN2 = 1.4426950408889634
    sm_scale = (1.0 / math.sqrt(D)) * RCP_LN2

    # (M, B, H) grid grouping inherently swizzles multi-head cross-batches. Concurrent outer
    # programs (same B,H) implicitly cache-share all identically mapped K/V slices from L2
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