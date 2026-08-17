import torch
import triton
import triton.language as tl
import math

# We configure an allocator infrastructure primarily required for device-side descriptor storage.
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "LOOP_STAGES": 2}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "LOOP_STAGES": 3}, num_warps=8, num_stages=3),
        
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "LOOP_STAGES": 3}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "LOOP_STAGES": 4}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "LOOP_STAGES": 5}, num_warps=8, num_stages=5),
        
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "LOOP_STAGES": 3}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "LOOP_STAGES": 4}, num_warps=8, num_stages=4),
        
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "LOOP_STAGES": 3}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "LOOP_STAGES": 4}, num_warps=4, num_stages=4),
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
    softmax_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    LOOP_STAGES: tl.constexpr,
    D: tl.constexpr,
    EXACT_M_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    # Cast batch and head to int64 for safe large tensor pointer math
    pid_b_64 = pid_b.to(tl.int64)
    pid_h_64 = pid_h.to(tl.int64)

    # Resolve base pointers logically bound to the current Batch & Head 
    q_base = Q + pid_b_64 * stride_qb + pid_h_64 * stride_qh
    k_base = K + pid_b_64 * stride_kb + pid_h_64 * stride_kh
    v_base = V + pid_b_64 * stride_vb + pid_h_64 * stride_vh
    
    # Device-side hardware TMA Descriptors providing seamless 2D block lowering
    q_desc = tl.make_tensor_descriptor(
        q_base, shape=[S, D], strides=[stride_qs, 1], block_shape=[BLOCK_M, D], padding_option="zero"
    )
    k_desc = tl.make_tensor_descriptor(
        k_base, shape=[S, D], strides=[stride_ks, 1], block_shape=[BLOCK_N, D], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        v_base, shape=[S, D], strides=[stride_vs, 1], block_shape=[BLOCK_N, D], padding_option="zero"
    )

    offset_m = (pid_m * BLOCK_M).to(tl.int32)
    q = q_desc.load([offset_m, 0])
    
    # Pre-scale Q to preserve tight-loop efficiency
    RCP_LN2: tl.constexpr = 1.4426950408889634
    SCALE = softmax_scale * RCP_LN2
    dtype = q.dtype
    q = (q * SCALE).to(dtype)

    neg_inf = float("-inf")
    m_i = tl.full((BLOCK_M,), neg_inf, tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, D), tl.float32)

    n_tiles = tl.cdiv(S, BLOCK_N)
    
    # Prune variables entirely when compiling for precisely divisible bounds
    if not EXACT_M_N:
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = (offs_m < S)[:, None]
        offs_n_template = tl.arange(0, BLOCK_N)
    
    for kv_tile in tl.range(0, n_tiles, num_stages=LOOP_STAGES):
        offset_n = (kv_tile * BLOCK_N).to(tl.int32)
        
        k = k_desc.load([offset_n, 0])
        v = v_desc.load([offset_n, 0])
        
        scores = tl.dot(q, k.T, out_dtype=tl.float32)
        
        if not EXACT_M_N:
            curr_offs_n = kv_tile * BLOCK_N + offs_n_template
            valid_score = mask_m & (curr_offs_n[None, :] < S)
            scores = tl.where(valid_score, scores, neg_inf)
        
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        
        # Dead-code eliminated at compile-time if S dictates perfectly aligned sequences
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

    # Reconstruct LSE via natural log mathematically conforming to PyTorch logic outputs
    LN2: tl.constexpr = 0.6931471805599453
    if not EXACT_M_N:
        safe_l_i = tl.where(l_i == 0.0, 1.0, l_i)
        output = acc / safe_l_i[:, None]
        lse_base2 = tl.where(l_i == 0.0, neg_inf, m_i + tl.math.log2(safe_l_i))
    else:
        output = acc / l_i[:, None]
        lse_base2 = m_i + tl.math.log2(l_i)
        
    lse_ln = lse_base2 * LN2

    # Deterministic unmasked store configuration avoiding redundant math branches
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_m_64 = offs_m.to(tl.int64)
    offs_d_64 = tl.arange(0, D).to(tl.int64)

    o_base = O + pid_b_64 * stride_ob + pid_h_64 * stride_oh
    o_ptrs = o_base + offs_m_64[:, None] * stride_os + offs_d_64[None, :] * stride_od
    
    lse_base = LSE + pid_b_64 * stride_lseb + pid_h_64 * stride_lseh
    lse_ptrs = lse_base + offs_m_64 * stride_lses

    if EXACT_M_N:
        tl.store(o_ptrs, output.to(dtype))
        tl.store(lse_ptrs, lse_ln)
    else:
        mask_m_store = offs_m < S
        tl.store(o_ptrs, output.to(dtype), mask=mask_m_store[:, None])
        tl.store(lse_ptrs, lse_ln, mask=mask_m_store)

def run(Q, K, V, O, LSE):
    # Prepare global execution state
    triton.set_allocator(alloc_fn)
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    softmax_scale = 1.0 / math.sqrt(D)
    
    # Check bounding capabilities logically to drop unnecessary heavy masking inside
    EXACT_M_N = (S % 128 == 0)
    
    # 3D grid layout organically dictates L2 swizzling properties prioritizing memory-reuse internally 
    grid = lambda META: (
        triton.cdiv(S, META["BLOCK_M"]),
        B,
        H,
    )
    
    _fwd_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        B, H, S,
        softmax_scale,
        D=D,
        EXACT_M_N=EXACT_M_N
    )