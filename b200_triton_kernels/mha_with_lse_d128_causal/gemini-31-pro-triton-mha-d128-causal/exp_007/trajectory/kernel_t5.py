import torch
import triton
import triton.language as tl
import math

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
    ],
    key=['S']
)
@triton.jit
def _attn_fwd_kernel(
    Q, K, V, sm_scale_log2,
    O, LSE,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    B, H, S, 
    BLOCK_D: tl.constexpr,
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    
    num_pid_m = tl.cdiv(S, BLOCK_M)
    num_pid_bh = B * H
    
    # Swizzle the launch grid to increase L2 cache hit rate for K and V
    group_id = pid // (GROUP_M * num_pid_bh)
    first_pid_m = group_id * GROUP_M
    group_size_m_adj = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    
    pid_in_group = pid - group_id * (GROUP_M * num_pid_bh)
    
    pid_m = first_pid_m + (pid_in_group % group_size_m_adj)
    batch_head = pid_in_group // group_size_m_adj
    
    offset_m = pid_m * BLOCK_M
    # Strict boundary exit (though grid swizzling guarantees we don't naturally exceed it)
    if offset_m >= S:
        return
        
    batch_idx = batch_head // H
    head_idx = batch_head % H
    
    # Offsets computing for coordinates in the flat memory pointers
    q_offset = batch_idx * stride_qb + head_idx * stride_qh
    k_offset = batch_idx * stride_kb + head_idx * stride_kh
    v_offset = batch_idx * stride_vb + head_idx * stride_vh
    o_offset = batch_idx * stride_ob + head_idx * stride_oh
    lse_offset = batch_idx * stride_lseb + head_idx * stride_lseh
    
    # Configure Hopper native TMA block data transfer descriptors mapping our contiguous dimensions optimally
    q_desc = tl.make_tensor_descriptor(
        Q + q_offset, shape=[S, BLOCK_D], strides=[stride_qs, stride_qd],
        block_shape=[BLOCK_M, BLOCK_D], padding_option="zero"
    )
    k_desc = tl.make_tensor_descriptor(
        K + k_offset, shape=[S, BLOCK_D], strides=[stride_ks, stride_kd],
        block_shape=[BLOCK_N, BLOCK_D], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        V + v_offset, shape=[S, BLOCK_D], strides=[stride_vs, stride_vd],
        block_shape=[BLOCK_N, BLOCK_D], padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        O + o_offset, shape=[S, BLOCK_D], strides=[stride_os, stride_od],
        block_shape=[BLOCK_M, BLOCK_D]
    )
    
    # Q remains mostly inside SM registers leveraging pipelined TMA. WGMMA uses Register-Shared mode
    q = q_desc.load([offset_m, 0])
    
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    # Calculate fully unmasked non-causal chunks guaranteeing offset_m >= offset_n
    num_full_blocks = offset_m // BLOCK_N
    
    # Stage 1: Evaluate chunks well within causality safely avoiding boundary conditions overheads
    for start_n in range(num_full_blocks):
        offset_n = start_n * BLOCK_N
        
        k = k_desc.load([offset_n, 0])
        v = v_desc.load([offset_n, 0])
        
        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        qk = qk * sm_scale_log2
        
        # Softmax math formulated effectively wrapping native MUFU hardware block capabilities (base 2)
        m_i_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp2(m_i - m_i_new)
        p = tl.exp2(qk - m_i_new[:, None])
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        p_bf16 = tl.cast(p, tl.bfloat16)
        
        acc = acc * alpha[:, None]
        # Distributing scaled values natively avoiding intermediate precision loss or store-restore penalties
        acc = tl.dot(p_bf16, v, acc=acc, out_dtype=tl.float32)
        
        m_i = m_i_new

    # Stage 2: Cross bounds evaluating causal boundaries
    n_start_max = offset_m + BLOCK_M
    if n_start_max > S:
        n_start_max = S
    
    num_n_blocks = tl.cdiv(n_start_max, BLOCK_N)
    offs_m = offset_m + tl.arange(0, BLOCK_M)
    
    for start_n in range(num_full_blocks, num_n_blocks):
        offset_n = start_n * BLOCK_N
        
        k = k_desc.load([offset_n, 0])
        v = v_desc.load([offset_n, 0])
        
        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        qk = qk * sm_scale_log2
        
        offs_n = offset_n + tl.arange(0, BLOCK_N)
        
        is_valid = offs_m[:, None] >= offs_n[None, :]
        qk = tl.where(is_valid, qk, float("-inf"))
        
        m_i_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp2(m_i - m_i_new)
        p = tl.exp2(qk - m_i_new[:, None])
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        p_bf16 = tl.cast(p, tl.bfloat16)
        
        acc = acc * alpha[:, None]
        acc = tl.dot(p_bf16, v, acc=acc, out_dtype=tl.float32)
        
        m_i = m_i_new

    acc = acc / l_i[:, None]
    
    # Reproject log2 domain outputs into strictly mathematically identical natural log equivalences properly 
    lse = m_i * 0.6931471805599453 + tl.log(l_i)
    
    o_desc.store([offset_m, 0], tl.cast(acc, tl.bfloat16))
    
    m_mask = offs_m < S
    lse_ptrs = LSE + lse_offset + offs_m * stride_lses
    tl.store(lse_ptrs, lse, mask=m_mask)


def run(Q, K, V, O, LSE):
    """
    Computes causal multi-head attention leveraging Triton hardware acceleration (WGMMA).
    Coordinates swizzle L2 hits utilizing async TMA operations perfectly overlapping.
    Destination tensors strictly filled in-place natively.
    """
    # Exposing generic allocator temporarily required by tl.make_tensor_descriptor for infrastructure storage.
    def alloc_fn(size: int, alignment: int, stream):
        return torch.empty(size, device="cuda", dtype=torch.int8)
    
    triton.set_allocator(alloc_fn)
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    
    # 1D flattened grid required explicitly enforcing optimal locality caching groups inside kernel
    grid = lambda META: (triton.cdiv(S, META['BLOCK_M']) * B * H, )
    
    # Base-2 Logarithmic scalar factoring mapping exactly log2(e) for hw unit utilization
    sm_scale_log2 = (1.0 / math.sqrt(D)) * 1.4426950408889634
    
    _attn_fwd_kernel[grid](
        Q, K, V, sm_scale_log2,
        O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        B, H, S, 
        BLOCK_D=D,
        GROUP_M=8,
    )