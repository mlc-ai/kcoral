import torch
import triton
import triton.language as tl

# Infrastructure storage for Hopper TMA descriptors.
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=4),
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
    B, H, S,
    qk_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(S, BLOCK_M)
    num_pid_bh = B * H

    # L2 Cache Swizzling: Group M blocks to maximize SM cache hit rates for K and V
    GROUP_M = 8
    num_pid_in_group = GROUP_M * num_pid_bh
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    
    pid_m = first_pid_m + (pid % group_size_m)
    off_bh = (pid % num_pid_in_group) // group_size_m
    
    start_m = pid_m
    b = off_bh // H
    h = off_bh % H
    
    # Base pointers for the current Batch and Head
    q_ptr = Q + b * stride_qb + h * stride_qh
    k_ptr = K + b * stride_kb + h * stride_kh
    v_ptr = V + b * stride_vb + h * stride_vh
    o_ptr = O + b * stride_ob + h * stride_oh
    
    # Native Hopper TMA Descriptors
    q_desc = tl.make_tensor_descriptor(
        q_ptr, shape=[S, BLOCK_D], strides=[stride_qs, stride_qd],
        block_shape=[BLOCK_M, BLOCK_D], padding_option="zero"
    )
    k_desc = tl.make_tensor_descriptor(
        k_ptr, shape=[S, BLOCK_D], strides=[stride_ks, stride_kd],
        block_shape=[BLOCK_N, BLOCK_D], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        v_ptr, shape=[S, BLOCK_D], strides=[stride_vs, stride_vd],
        block_shape=[BLOCK_N, BLOCK_D], padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        o_ptr, shape=[S, BLOCK_D], strides=[stride_os, stride_od],
        block_shape=[BLOCK_M, BLOCK_D]
    )
    
    # Pre-load and scale Query
    q = q_desc.load([start_m * BLOCK_M, 0])
    q = (q * qk_scale).to(tl.bfloat16)
    
    # Online Softmax (FlashAttention-2) statistics mapped securely to fast base-2 hardware functions
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    # Pre-compute static boundary threshold statically out of the hot loop
    limit = (S // BLOCK_N) * BLOCK_N
    
    for start_n in range(0, S, BLOCK_N):
        k = k_desc.load([start_n, 0])
        # WGMMA fused dot
        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        
        # Avoid masking logic in 99% of loop iterations dynamically 
        if start_n >= limit:
            offs_n = start_n + tl.arange(0, BLOCK_N)
            qk = tl.where(offs_n[None, :] < S, qk, float('-inf'))
            
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        
        alpha = tl.exp2(m_i - m_ij)
        acc = acc * alpha[:, None]
        
        v = v_desc.load([start_n, 0])
        acc += tl.dot(p.to(tl.bfloat16), v, out_dtype=tl.float32)
        
        m_i = m_ij
        l_i = l_i * alpha + l_ij
        
    # Finalize attention output
    acc = acc / l_i[:, None]
    
    # Re-normalize LSE into true natural-log: LSE = m_i * ln(2) + ln(l_i)
    lse = m_i * 0.6931471805599453 + tl.log(l_i)
    
    # TMA bounds check automatically handles boundary zeros gracefully
    o_desc.store([start_m * BLOCK_M, 0], acc.to(tl.bfloat16))
    
    # Store LSE linearly via mapped memory logic (fast pathing)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    lse_ptrs = LSE + b * stride_lseb + h * stride_lseh + offs_m * stride_lses
    if (start_m + 1) * BLOCK_M <= S:
        tl.store(lse_ptrs, lse)
    else:
        tl.store(lse_ptrs, lse, mask=offs_m < S)


def run(Q, K, V, O, LSE):
    """
    Executes a heavily optimized, memory-efficient non-causal Flash Attention 
    forward pass natively utilizing Hopper Tensor Memory Accelerator (TMA) WGMMA features.
    
    Inputs:
        Q, K, V: [B, H, S, 128] bfloat16 tensors.
    Outputs:
        O: Preallocated [B, H, S, 128] bfloat16 tensor.
        LSE: Preallocated [B, H, S] float32 tensor.
    """
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    sm_scale = 1.0 / (D ** 0.5)
    log2_e = 1.4426950408889634
    qk_scale = sm_scale * log2_e
    
    # 1D Grid enables precise Swizzling inside the kernel for optimal Hopper L2 Cache locality
    grid = lambda META: (triton.cdiv(S, META["BLOCK_M"]) * B * H, )
    
    _fwd_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        B, H, S,
        qk_scale,
        BLOCK_D=128
    )