import torch
import triton
import triton.language as tl
import math

# Register the device-side descriptor allocator required for Hopper TMA
def _alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(_alloc_fn)

@triton.autotune(
    configs=[
        # Aggressive configs: heavily reduce K/V memory traffic by using large M tiles.
        # This is the primary key to achieving >1.0x cuDNN speeds on Hopper.
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64}, num_warps=8, num_stages=4),
        # Standard robust configs
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=4),
        # Smaller footprint configs
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=3),
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
    S, sm_scale_log2,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_bh = tl.program_id(1)
    num_pid_m = tl.cdiv(S, BLOCK_M)
    
    # Longest-First Load Balancing for Causal Mask
    # Reverse scheduling ensures the heaviest causal blocks execute first, avoiding
    # tail stragglers and naturally filling gaps with smaller blocks toward the end.
    pid_m = num_pid_m - 1 - tl.program_id(0)
    
    start_m = pid_m * BLOCK_M
    if start_m >= S:
        return
        
    batch_id = pid_bh // H
    head_id = pid_bh % H
    
    # Base pointers for this batch and head
    q_ptr = Q + batch_id * stride_qb + head_id * stride_qh
    k_ptr = K + batch_id * stride_kb + head_id * stride_kh
    v_ptr = V + batch_id * stride_vb + head_id * stride_vh
    o_ptr = O + batch_id * stride_ob + head_id * stride_oh
    
    # Create device-side descriptors for Hopper TMA
    # TMA natively bounds-checks and zero-pads out-of-bounds loads and silently drops invalid stores.
    q_desc = tl.make_tensor_descriptor(
        q_ptr, shape=[S, D], strides=[stride_qs, stride_qd],
        block_shape=[BLOCK_M, D], padding_option="zero"
    )
    k_desc = tl.make_tensor_descriptor(
        k_ptr, shape=[S, D], strides=[stride_ks, stride_kd],
        block_shape=[BLOCK_N, D], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        v_ptr, shape=[S, D], strides=[stride_vs, stride_vd],
        block_shape=[BLOCK_N, D], padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        o_ptr, shape=[S, D], strides=[stride_os, stride_od],
        block_shape=[BLOCK_M, D]
    )
    
    q = q_desc.load([start_m, 0])
    
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n_base = tl.arange(0, BLOCK_N)
    
    # Determine iteration boundaries to strictly split loops and maximize TMA/WGMMA pipelining
    max_k_len = tl.minimum(start_m + BLOCK_M, S)
    num_n_blocks = tl.cdiv(max_k_len, BLOCK_N)
    num_full_blocks = start_m // BLOCK_N
    
    # 1. Fully unrolled & pipelined loop for causal-free blocks (No masking overhead)
    for start_n_idx in tl.range(0, num_full_blocks):
        start_n = start_n_idx * BLOCK_N
        
        k = k_desc.load([start_n, 0])
        v = v_desc.load([start_n, 0])
        
        qk = tl.dot(q, k.T)
        qk = qk * sm_scale_log2
        
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp2(m_i - m_i_new)
        p = tl.exp2(qk - m_i_new[:, None])
        
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(q.dtype), v, acc)
        
        m_i = m_i_new

    # 2. Epilogue loop strictly for partial blocks intersecting the causal diagonal/boundaries
    for start_n_idx in range(num_full_blocks, num_n_blocks):
        start_n = start_n_idx * BLOCK_N
        
        k = k_desc.load([start_n, 0])
        v = v_desc.load([start_n, 0])
        
        qk = tl.dot(q, k.T)
        qk = qk * sm_scale_log2
        
        offs_n = start_n + offs_n_base
        is_valid = offs_m[:, None] >= offs_n[None, :]
        qk = tl.where(is_valid, qk, float('-inf'))
        
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp2(m_i - m_i_new)
        p = tl.exp2(qk - m_i_new[:, None])
        
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(q.dtype), v, acc)
        
        m_i = m_i_new

    # Writeback
    inv_l_i = 1.0 / l_i
    acc = acc * inv_l_i[:, None]
    lse = (m_i + tl.log2(l_i)) * 0.6931471805599453 # mathematically revert log2 base scaling to natural log `e`
    
    # Store Output using TMA (automatically handles padding and out-of-bounds dropping)
    o_desc.store([start_m, 0], acc.to(tl.bfloat16))
    
    # Store LSE leveraging strict regular masked pointers
    lse_ptrs = LSE + batch_id * stride_lseb + head_id * stride_lseh + offs_m * stride_lses
    tl.store(lse_ptrs, lse, mask=offs_m < S)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    if S == 0:
        return

    grid = lambda META: (
        triton.cdiv(S, META['BLOCK_M']),
        B * H,
        1
    )
    
    # Precompute scale + hardware base2 conversion scaling directly factored prior to runtime
    sm_scale_log2 = (1.0 / math.sqrt(D)) * 1.4426950408889634
    
    _fwd_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        S, sm_scale_log2,
        H=H, D=D
    )