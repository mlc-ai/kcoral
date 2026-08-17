import torch
import triton
import triton.language as tl
import math

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=4),
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
    B, H, S, D: tl.constexpr,
    softmax_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(S, BLOCK_M)
    
    # Swizzle grouping to improve L2 cache utilization dramatically
    GROUP_M: tl.constexpr = 8
    num_pid_in_group = GROUP_M * H
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(grid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_h = (pid % num_pid_in_group) // group_size_m
    pid_b = tl.program_id(1)

    # Cast batch and head to int64 for safe base pointer math
    pid_b_64 = pid_b.to(tl.int64)
    pid_h_64 = pid_h.to(tl.int64)

    # Base pointers for this batch and head
    q_base = Q + pid_b_64 * stride_qb + pid_h_64 * stride_qh
    k_base = K + pid_b_64 * stride_kb + pid_h_64 * stride_kh
    v_base = V + pid_b_64 * stride_vb + pid_h_64 * stride_vh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    
    offs_m_64 = offs_m.to(tl.int64)
    offs_n_64 = offs_n.to(tl.int64)
    offs_d_64 = offs_d.to(tl.int64)

    mask_m = offs_m < S
    mask_d = offs_d < D

    # Load Q
    q_ptrs = q_base + offs_m_64[:, None] * stride_qs + offs_d_64[None, :] * stride_qd
    q_mask = mask_m[:, None] & mask_d[None, :]
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)
    
    # Pre-scale Q to save redundant operations in the inner loop
    RCP_LN2: tl.constexpr = 1.4426950408889634
    SCALE = softmax_scale * RCP_LN2
    dtype = q.dtype
    q = (q * SCALE).to(dtype)

    # Initialize pointers for K and V
    k_ptrs = k_base + offs_n_64[:, None] * stride_ks + offs_d_64[None, :] * stride_kd
    v_ptrs = v_base + offs_n_64[:, None] * stride_vs + offs_d_64[None, :] * stride_vd

    neg_inf = float("-inf")
    m_i = tl.full((BLOCK_M,), neg_inf, tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, D), tl.float32)

    n_tiles = tl.cdiv(S, BLOCK_N)
    
    for n in range(0, n_tiles):
        curr_offs_n = n * BLOCK_N + offs_n
        mask_n = curr_offs_n < S
        kv_mask = mask_n[:, None] & mask_d[None, :]
        
        k = tl.load(k_ptrs, mask=kv_mask, other=0.0)
        v = tl.load(v_ptrs, mask=kv_mask, other=0.0)
        
        scores = tl.dot(q, k.T, out_dtype=tl.float32)
        
        valid_score = mask_m[:, None] & mask_n[None, :]
        scores = tl.where(valid_score, scores, neg_inf)
        
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        safe_m_ij = tl.where(m_ij == neg_inf, 0.0, m_ij)
        
        alpha = tl.math.exp2(m_i - safe_m_ij)
        p = tl.math.exp2(scores - safe_m_ij[:, None])
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(dtype), v, acc, out_dtype=tl.float32)
        
        m_i = m_ij
        
        # Advance pointers explicitly
        k_ptrs += BLOCK_N * stride_ks
        v_ptrs += BLOCK_N * stride_vs

    safe_l_i = tl.where(l_i == 0.0, 1.0, l_i)
    output = acc / safe_l_i[:, None]
    
    LN2: tl.constexpr = 0.6931471805599453
    lse_base2 = tl.where(l_i == 0.0, neg_inf, m_i + tl.math.log2(safe_l_i))
    lse_ln = lse_base2 * LN2

    # Final masked stores
    o_base = O + pid_b_64 * stride_ob + pid_h_64 * stride_oh
    o_ptrs = o_base + offs_m_64[:, None] * stride_os + offs_d_64[None, :] * stride_od
    tl.store(o_ptrs, output.to(dtype), mask=mask_m[:, None] & mask_d[None, :])

    lse_base = LSE + pid_b_64 * stride_lseb + pid_h_64 * stride_lseh
    lse_ptrs = lse_base + offs_m_64 * stride_lses
    tl.store(lse_ptrs, lse_ln, mask=mask_m)

def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    softmax_scale = 1.0 / math.sqrt(D)
    
    grid = lambda META: (
        triton.cdiv(S, META["BLOCK_M"]) * H,
        B,
    )
    
    _fwd_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        B, H, S, D,
        softmax_scale
    )