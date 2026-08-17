import torch
import triton
import triton.language as tl
import math

# Provides device-side backing infrastructure needed when initializing hardware TMA descriptors
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

@triton.autotune(
    configs=[
        # Max-sized tiles optimizing WGMMA throughput directly over sequence lengths
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "USE_TMA": 0}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "USE_TMA": 1}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "USE_TMA": 0}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "USE_TMA": 1}, num_warps=8, num_stages=3),
        
        # Pipelined asymmetric configurations tailored for shared memory occupancy capabilities
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "USE_TMA": 0}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "USE_TMA": 1}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "USE_TMA": 0}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "USE_TMA": 1}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "USE_TMA": 0}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "USE_TMA": 1}, num_warps=8, num_stages=4),
        
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "USE_TMA": 0}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "USE_TMA": 1}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "USE_TMA": 0}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "USE_TMA": 1}, num_warps=8, num_stages=3),
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
    B, H, S,
    softmax_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
    EXACT_M_N: tl.constexpr,
    USE_TMA: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(S, BLOCK_M)
    
    # Grid Swizzle scaling -> maps consecutive memory chunks inside L2 concurrently 
    GROUP_M: tl.constexpr = 16
    num_pid_in_group = GROUP_M * H
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(grid_m - first_pid_m, GROUP_M)
    
    pid_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_h = pid_in_group // group_size_m
    pid_b = tl.program_id(1)

    pid_b_64 = pid_b.to(tl.int64)
    pid_h_64 = pid_h.to(tl.int64)

    # Establish head base offsets
    q_offset = pid_b_64 * stride_qb + pid_h_64 * stride_qh
    k_offset = pid_b_64 * stride_kb + pid_h_64 * stride_kh
    v_offset = pid_b_64 * stride_vb + pid_h_64 * stride_vh
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    
    offs_m_64 = offs_m.to(tl.int64)
    offs_n_64 = offs_n.to(tl.int64)
    offs_d_64 = offs_d.to(tl.int64)

    if USE_TMA:
        # Note: we explicitly specify `1` for the last dim stride implicitly aligned with our contiguous PyTorch inputs
        q_desc = tl.make_tensor_descriptor(
            Q + q_offset, shape=[S, D], strides=[stride_qs, 1], block_shape=[BLOCK_M, D], padding_option="zero"
        )
        k_desc = tl.make_tensor_descriptor(
            K + k_offset, shape=[S, D], strides=[stride_ks, 1], block_shape=[BLOCK_N, D], padding_option="zero"
        )
        v_desc = tl.make_tensor_descriptor(
            V + v_offset, shape=[S, D], strides=[stride_vs, 1], block_shape=[BLOCK_N, D], padding_option="zero"
        )
        offset_m = (pid_m * BLOCK_M).to(tl.int32)
        q = q_desc.load([offset_m, 0])
    else:
        q_ptrs = Q + q_offset + offs_m_64[:, None] * stride_qs + offs_d_64[None, :]
        if EXACT_M_N:
            q = tl.load(q_ptrs)
        else:
            mask_m = offs_m < S
            q = tl.load(q_ptrs, mask=mask_m[:, None] & (offs_d < D)[None, :], other=0.0)

    # Convert the softmax scaling constraint to base-2 mathematically reducing in-loop FP operations
    RCP_LN2: tl.constexpr = 1.4426950408889634
    SCALE = softmax_scale * RCP_LN2
    dtype = q.dtype
    q = (q * SCALE).to(dtype)

    if not USE_TMA:
        k_ptrs = K + k_offset + offs_n_64[:, None] * stride_ks + offs_d_64[None, :]
        v_ptrs = V + v_offset + offs_n_64[:, None] * stride_vs + offs_d_64[None, :]

    if not EXACT_M_N:
        mask_m = offs_m < S

    neg_inf = float("-inf")
    m_i = tl.full((BLOCK_M,), neg_inf, tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, D), tl.float32)

    n_tiles = tl.cdiv(S, BLOCK_N)
    
    for kv_tile in range(0, n_tiles):
        if USE_TMA:
            offset_n = (kv_tile * BLOCK_N).to(tl.int32)
            k = k_desc.load([offset_n, 0])
            v = v_desc.load([offset_n, 0])
        else:
            if EXACT_M_N:
                k = tl.load(k_ptrs)
                v = tl.load(v_ptrs)
            else:
                curr_offs_n = kv_tile * BLOCK_N + offs_n
                mask_n = curr_offs_n < S
                kv_mask = mask_n[:, None] & (offs_d < D)[None, :]
                k = tl.load(k_ptrs, mask=kv_mask, other=0.0)
                v = tl.load(v_ptrs, mask=kv_mask, other=0.0)
        
        # Native WGMMA execution resolving `.T` transpositions inherently optimal in Blackwell
        scores = tl.dot(q, k.T, out_dtype=tl.float32)
        
        if not EXACT_M_N:
            if USE_TMA:
                curr_offs_n = kv_tile * BLOCK_N + offs_n
                mask_n = curr_offs_n < S
            valid_score = mask_m[:, None] & mask_n[None, :]
            scores = tl.where(valid_score, scores, neg_inf)
            
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        
        if not EXACT_M_N:
            safe_m_ij = tl.where(m_ij == neg_inf, 0.0, m_ij)
            alpha = tl.math.exp2(m_i - safe_m_ij)
            p = tl.math.exp2(scores - safe_m_ij[:, None])
        else:
            alpha = tl.math.exp2(m_i - m_ij)
            p = tl.math.exp2(scores - m_ij[:, None])
            
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(dtype), v, acc, out_dtype=tl.float32)
        
        m_i = m_ij
        
        if not USE_TMA:
            # Advance manual pointers independently avoiding interior array recalculation
            k_ptrs += BLOCK_N * stride_ks
            v_ptrs += BLOCK_N * stride_vs

    # Conform internal optimized properties back standard outputs formats safely
    LN2: tl.constexpr = 0.6931471805599453
    if not EXACT_M_N:
        safe_l_i = tl.where(l_i == 0.0, 1.0, l_i)
        output = acc / safe_l_i[:, None]
        lse_base2 = tl.where(l_i == 0.0, neg_inf, m_i + tl.math.log2(safe_l_i))
    else:
        output = acc / l_i[:, None]
        lse_base2 = m_i + tl.math.log2(l_i)
        
    lse_ln = lse_base2 * LN2

    o_offset = pid_b_64 * stride_ob + pid_h_64 * stride_oh
    o_ptrs = O + o_offset + offs_m_64[:, None] * stride_os + offs_d_64[None, :]
    
    lse_offset = pid_b_64 * stride_lseb + pid_h_64 * stride_lseh
    lse_ptrs = LSE + lse_offset + offs_m_64 * stride_lses

    if EXACT_M_N:
        tl.store(o_ptrs, output.to(dtype))
        tl.store(lse_ptrs, lse_ln)
    else:
        tl.store(o_ptrs, output.to(dtype), mask=mask_m[:, None] & (offs_d < D)[None, :])
        tl.store(lse_ptrs, lse_ln, mask=mask_m)

def run(Q, K, V, O, LSE):
    # Setup standard framework allocators enforcing TMA stability boundaries bounds checks
    triton.set_allocator(alloc_fn)
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    softmax_scale = 1.0 / math.sqrt(D)
    
    # Max bound available inside `configs` ensures exact scaling bounds check handles everything correctly
    EXACT_M_N = (S % 128 == 0)
    
    # 2D Grid structure supporting our underlying dynamic swizzle mechanism efficiently inside
    grid = lambda META: (
        triton.cdiv(S, META["BLOCK_M"]) * H,
        B,
    )
    
    _fwd_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), 
        K.stride(0), K.stride(1), K.stride(2), 
        V.stride(0), V.stride(1), V.stride(2), 
        O.stride(0), O.stride(1), O.stride(2), 
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        B, H, S,
        softmax_scale,
        D=D,
        EXACT_M_N=EXACT_M_N
    )