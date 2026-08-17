import torch
import triton
import triton.language as tl
import math

# Configure Triton's allocator to house JIT-compiled TMA descriptors in device memory infrastructure.
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

@triton.autotune(
    configs=[
        # Optimized configurations tailored specifically for Blackwell's 228 KB shared memory envelope.
        # Larger BLOCK_M sizes dramatically reduce K/V memory bandwidth demands.
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64}, num_warps=8, num_stages=4),
        
        # Standard balanced baselines heavily relying on deep pipelining
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        
        # High concurrency variants for robust latency hiding 
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=4),
        
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=3),
    ],
    key=["S"],
)
@triton.jit
def _fwd_kernel(
    Q, K, V, O, LSE,
    stride_qb, stride_qh, stride_qs,
    stride_kb, stride_kh, stride_ks,
    stride_vb, stride_vh, stride_vs,
    stride_ob, stride_oh, stride_os,
    stride_lseb, stride_lseh, stride_lses,
    B, H, S, sm_scale,
    EVEN_K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid = tl.program_id(0)
    
    # -------------------------------------------------------------------------
    # L2 Cache Swizzling Core: Intentionally groups multi-head queries globally 
    # to enforce cooperative L2 KV cache hits across SM hardware groups.
    # -------------------------------------------------------------------------
    NUM_PID_M = tl.cdiv(S, BLOCK_M)
    NUM_PID_BH = B * H
    GROUP_M = 8
    
    num_pid_in_group = GROUP_M * NUM_PID_BH
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(NUM_PID_M - first_pid_m, GROUP_M)
    
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_bh = (pid % num_pid_in_group) // group_size_m
    
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    # Evaluate exact memory offsets relative to isolated batch & head dimensions
    q_base = Q + pid_b * stride_qb + pid_h * stride_qh
    k_base = K + pid_b * stride_kb + pid_h * stride_kh
    v_base = V + pid_b * stride_vb + pid_h * stride_vh
    o_base = O + pid_b * stride_ob + pid_h * stride_oh

    # Hardware TMA standard layouts automatically handling 16-byte edge constraints 
    q_desc = tl.make_tensor_descriptor(
        q_base, shape=[S, BLOCK_D], strides=[stride_qs, 1],
        block_shape=[BLOCK_M, BLOCK_D], padding_option="zero"
    )
    k_desc = tl.make_tensor_descriptor(
        k_base, shape=[S, BLOCK_D], strides=[stride_ks, 1],
        block_shape=[BLOCK_N, BLOCK_D], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        v_base, shape=[S, BLOCK_D], strides=[stride_vs, 1],
        block_shape=[BLOCK_N, BLOCK_D], padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        o_base, shape=[S, BLOCK_D], strides=[stride_os, 1],
        block_shape=[BLOCK_M, BLOCK_D], padding_option="zero"
    )

    # Issue Initial Async Load
    offset_m = (pid_m * BLOCK_M).to(tl.int32)
    q = q_desc.load([offset_m, 0])
    
    # -----------------------------------------------------------------------
    # COMPUTE OPTIMIZATION: Pre-scale `Q` exclusively. This skips resolving 
    # FP32 scaling inside the KV loop, saving literally millions of arithmetic 
    # scaling operations throughout the sequence lifetime!
    # -----------------------------------------------------------------------
    q = (q * sm_scale).to(tl.bfloat16)

    # Establish strictly floating-point online softmax caches natively mapped 
    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    kv_tiles = tl.cdiv(S, BLOCK_N)
    
    for kv_tile in range(0, kv_tiles):
        offset_n = (kv_tile * BLOCK_N).to(tl.int32)
        k = k_desc.load([offset_n, 0])
        v = v_desc.load([offset_n, 0])

        # Fifth-Generation Native Tensor Cores perfectly matched (FP32 Accumulation)
        scores = tl.dot(q, k.T, out_dtype=tl.float32)
        
        # When `EVEN_K` natively satisfies dimensions, compiler strips everything inside this `if` block,
        # leaving an uninterrupted mathematically dense sequence directly bounding TC limits
        if not EVEN_K:
            offs_n_v = offset_n + tl.arange(0, BLOCK_N)
            scores = tl.where(offs_n_v[None, :] < S, scores, -float("inf"))

        # Fused Hardware Exponential Pipeline 
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        safe_m_ij = tl.where(m_ij == -float("inf"), 0.0, m_ij)
        
        alpha = tl.math.exp2(m_i - safe_m_ij)
        p = tl.math.exp2(scores - safe_m_ij[:, None])

        # Resolve & Aggregate 
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc=acc)
        m_i = m_ij

    safe_l_i = tl.where(l_i == 0.0, 1.0, l_i)
    output = acc / safe_l_i[:, None]
    
    # Reverse mapping matching PyTorch baseline API requirements (fp32 Natural-Log)
    lse_log2 = tl.where(l_i == 0.0, -float("inf"), m_i + tl.math.log2(safe_l_i))
    LN2 = 0.6931471805599453
    lse = lse_log2 * LN2

    # Descriptor automatically handles and ignores any padded rows safely mapping into output dimensions!
    o_desc.store([offset_m, 0], output.to(tl.bfloat16))

    # Classical store sequence for `LSE` pointers preserving dynamic outer masks
    offs_m_v = offset_m + tl.arange(0, BLOCK_M)
    q_valid = offs_m_v < S
    lse_ptrs = LSE + pid_b * stride_lseb + pid_h * stride_lseh + offs_m_v * stride_lses
    tl.store(lse_ptrs, lse, mask=q_valid)


def run(Q, K, V, O, LSE):
    """
    Computes natively bound standard scaled dot product attention. 
    Integrates L2 programmatic swizzling & TMA desc. mapping.
    """
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    # Algebraically consolidate the math constant log2(e) inherently directly into the initial scaling 
    RCP_LN2 = 1.4426950408889634
    sm_scale = (1.0 / math.sqrt(D)) * RCP_LN2

    # Boolean explicitly compiled down directly enabling zero-mask pipelined loops for aligned inputs (e.g S=4096)
    EVEN_K = (S % 64 == 0)

    # Total grid length matches our L2 programmatic distribution group 
    grid = lambda META: (triton.cdiv(S, META["BLOCK_M"]) * B * H, )
    
    _fwd_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        B, H, S, sm_scale,
        EVEN_K=EVEN_K,
        BLOCK_D=D
    )