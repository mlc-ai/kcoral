import math
import torch
import triton
import triton.language as tl


def alloc_fn(size: int, alignment: int, stream):
    # Infrastructure storage allocator for Hopper device-side descriptor creation
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=4),
    ],
    key=["S"],
)
@triton.jit
def _attn_fwd_kernel(
    Q, K, V, log2_scale,
    O, LSE,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    H, S,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    
    # Safely bypass fully out-of-bounds queries 
    if start_m * BLOCK_M >= S:
        return
        
    off_hz = tl.program_id(1)
    off_b = off_hz // H
    off_h = off_hz % H
    
    q_base = Q + off_b * stride_qb + off_h * stride_qh
    k_base = K + off_b * stride_kb + off_h * stride_kh
    v_base = V + off_b * stride_vb + off_h * stride_vh
    o_base = O + off_b * stride_ob + off_h * stride_oh
    lse_base = LSE + off_b * stride_lseb + off_h * stride_lseh
    
    # Generate TMA descriptors matching PyTorch's native contiguous layout 
    q_desc = tl.make_tensor_descriptor(
        q_base, shape=[S, D], strides=[stride_qs, 1],
        block_shape=[BLOCK_M, D], padding_option="zero"
    )
    k_desc = tl.make_tensor_descriptor(
        k_base, shape=[S, D], strides=[stride_ks, 1],
        block_shape=[BLOCK_N, D], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        v_base, shape=[S, D], strides=[stride_vs, 1],
        block_shape=[BLOCK_N, D], padding_option="zero"
    )
    # Target TMA handles O-bounds naturally skipping out-of-bounds rows
    o_desc = tl.make_tensor_descriptor(
        o_base, shape=[S, D], strides=[stride_os, 1],
        block_shape=[BLOCK_M, D]
    )
    
    q = q_desc.load([start_m * BLOCK_M, 0])
    
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    
    hi = (start_m + 1) * BLOCK_M
    if hi > S:
        hi = S
        
    num_total_steps = tl.cdiv(hi, BLOCK_N)
    
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    
    # A cleanly unified pipelinable inner loop ensuring TTGIR stability 
    # without duplication block explosions
    for step in range(0, num_total_steps):
        start_n = step * BLOCK_N
        k = k_desc.load([start_n, 0])
        v = v_desc.load([start_n, 0])
        
        # 1. Compute similarity & Scale directly on Fast FP32 Context
        qk = tl.dot(q, k.T)
        qk = qk * log2_scale
        
        # 2. Causality and Bounds Limits 
        # (This formulation safely masks both causal boundaries AND limits K to S-bounds implicitly)
        limit = offs_m[:, None] - start_n
        qk = tl.where(limit >= offs_n[None, :], qk, float("-inf"))
            
        # 3. Softmax State Tracing with single-cycle `tl.exp2` mapping
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        
        alpha = tl.exp2(m_i - m_ij)
        acc = acc * alpha[:, None]
        
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        
        # 4. Values Update mapped cleanly to WGMMA pipeline layout
        p = p.to(tl.bfloat16)
        acc = tl.dot(p, v, acc)
        
    inv_l = 1.0 / l_i
    acc = acc * inv_l[:, None]
    
    # Restore internal base-2 scaling LSE representation back to original Natural Log form `ln(2)`
    lse = (m_i + tl.log2(l_i)) * 0.6931471805599453
    
    # Safely commit boundaries via descriptor store logic
    acc_out = acc.to(tl.bfloat16)
    o_desc.store([start_m * BLOCK_M, 0], acc_out)
    
    # Vectorized 1D output limits block storage manually handled
    lse_ptrs = lse_base + offs_m * stride_lses
    q_mask = offs_m < S
    tl.store(lse_ptrs, lse, mask=q_mask)


def run(Q, K, V, O, LSE):
    """Causal multi-head attention ensuring destination-passing constraints natively."""
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    if S == 0:
        return
        
    # Scale embeds both normal PyTorch Attention 1/sqrt(D) and hardware optimal log2(e) precomputations
    log2_scale = float(1.0 / math.sqrt(D)) * 1.4426950408889634
    
    # Launch topology implicitly enforces ideal grouping mapping for the identical memory cache `start_hz` hits
    grid = lambda META: (triton.cdiv(S, META["BLOCK_M"]), B * H)
    
    _attn_fwd_kernel[grid](
        Q, K, V, log2_scale,
        O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        H, S,
        D=128,
    )