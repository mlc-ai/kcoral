import torch
import triton
import triton.language as tl
import math

def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "LOOP_STAGES": 3}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "LOOP_STAGES": 4}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "LOOP_STAGES": 2}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "LOOP_STAGES": 3}, num_warps=4, num_stages=3),
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
    S,
    softmax_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    LOOP_STAGES: tl.constexpr,
    D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1).to(tl.int64)
    pid_h = tl.program_id(2).to(tl.int64)

    # Base pointers offset for the current batch and head
    q_offset = pid_b * stride_qb + pid_h * stride_qh
    k_offset = pid_b * stride_kb + pid_h * stride_kh
    v_offset = pid_b * stride_vb + pid_h * stride_vh
    o_offset = pid_b * stride_ob + pid_h * stride_oh
    lse_offset = pid_b * stride_lseb + pid_h * stride_lseh

    # Hardware TMA Descriptors for Blackwell
    q_desc = tl.make_tensor_descriptor(
        Q + q_offset,
        shape=[S, D],
        strides=[stride_qs, 1],
        block_shape=[BLOCK_M, D],
        padding_option="zero"
    )
    k_desc = tl.make_tensor_descriptor(
        K + k_offset,
        shape=[S, D],
        strides=[stride_ks, 1],
        block_shape=[BLOCK_N, D],
        padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        V + v_offset,
        shape=[S, D],
        strides=[stride_vs, 1],
        block_shape=[BLOCK_N, D],
        padding_option="zero"
    )

    # Offsets for manually storing outputs safely
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)

    offs_m_64 = offs_m.to(tl.int64)
    offs_d_64 = offs_d.to(tl.int64)
    
    # Load query tile (TMA correctly pads out-of-bound sequences with 0)
    offset_m = (pid_m * BLOCK_M).to(tl.int32)
    q = q_desc.load([offset_m, 0])
    dtype = q.dtype

    RCP_LN2: tl.constexpr = 1.4426950408889634
    LN2: tl.constexpr = 0.6931471805599453
    neg_inf = float("-inf")
    
    # Trackers for online softmax
    m_i = tl.full((BLOCK_M,), neg_inf, tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, D), tl.float32)

    # Loop pipelining enabled using the explicitly separated loop_stages directive
    n_tiles = tl.cdiv(S, BLOCK_N)
    for kv_tile in tl.range(0, n_tiles, num_stages=LOOP_STAGES):
        offset_n = (kv_tile * BLOCK_N).to(tl.int32)
        
        # Descriptor-backed TMA loads
        k = k_desc.load([offset_n, 0])
        v = v_desc.load([offset_n, 0])
        
        # Mask calculation for bounds checking since padded TMA returns zeros which bias scores
        curr_offs_n = kv_tile * BLOCK_N + tl.arange(0, BLOCK_N)
        valid_score = (offs_m[:, None] < S) & (curr_offs_n[None, :] < S)
        
        # Matmul logic targeting FP32 accumulation out-of-the-box
        scores = tl.dot(q, k.T, out_dtype=tl.float32) * softmax_scale
        scores = tl.where(valid_score, scores * RCP_LN2, neg_inf)
        
        # Max reduction with NaN immunity
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        safe_m_ij = tl.where(m_ij == neg_inf, 0.0, m_ij)
        
        # Softmax exponentials calculation in base-2
        alpha = tl.math.exp2(m_i - safe_m_ij)
        p = tl.math.exp2(scores - safe_m_ij[:, None])
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(dtype), v.to(dtype), acc, out_dtype=tl.float32)
        
        m_i = m_ij

    # Finalize entirely masked rows cleanly without triggering NaN
    safe_l_i = tl.where(l_i == 0.0, 1.0, l_i)
    output = acc / safe_l_i[:, None]
    
    # Store standard multi-head attention outputs
    o_ptrs = O + o_offset + offs_m_64[:, None] * stride_os + offs_d_64[None, :] * stride_od
    tl.store(o_ptrs, output.to(dtype), mask=(offs_m[:, None] < S) & (offs_d[None, :] < D))
    
    # Formulate and store natural log LSE (as PyTorch outputs naturally)
    lse_base2 = tl.where(l_i == 0.0, neg_inf, m_i + tl.math.log2(safe_l_i))
    lse_ln = lse_base2 * LN2
    
    lse_ptrs = LSE + lse_offset + offs_m_64 * stride_lses
    lse_mask = offs_m < S
    tl.store(lse_ptrs, lse_ln, mask=lse_mask)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    
    # Setup allocator for device-side standard TMA descriptors
    triton.set_allocator(alloc_fn)
    
    B, H, S, D = Q.shape
    softmax_scale = 1.0 / math.sqrt(D)
    
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
        S,
        softmax_scale,
        D=D
    )