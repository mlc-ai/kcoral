import torch
import triton
import triton.language as tl
import math
from triton.tools.tensor_descriptor import TensorDescriptor

def desc_pre_hook(kwargs):
    """
    Hook to initialize host-side TensorDescriptors for the TMA. 
    Constructing them on the host avoids the significant overhead of per-CTA device-side TMA creation.
    """
    Q, K, V = kwargs["Q"], kwargs["K"], kwargs["V"]
    BM, BN, D = kwargs["BLOCK_M"], kwargs["BLOCK_N"], kwargs["D"]
    
    # Create 4D host descriptors. The block shape must match the tensor dimensionality.
    # We define 1x1 chunking along Batch and Head to easily slice exactly what we want in the kernel.
    kwargs["q_desc"] = TensorDescriptor.from_tensor(Q, [1, 1, BM, D])
    kwargs["k_desc"] = TensorDescriptor.from_tensor(K, [1, 1, BN, D])
    kwargs["v_desc"] = TensorDescriptor.from_tensor(V, [1, 1, BN, D])


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "LOOP_STAGES": 2}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "LOOP_STAGES": 3}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "LOOP_STAGES": 4}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "LOOP_STAGES": 3}, num_warps=4, num_stages=3),
    ],
    key=["S"],
    pre_hook=desc_pre_hook,
)
@triton.jit
def _fwd_kernel(
    q_desc, k_desc, v_desc,
    Q, K, V,  # Used by pre_hook; safely bypassed by TMA in the kernel
    O, LSE,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    S, SCALE,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, LOOP_STAGES: tl.constexpr, D: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    # Blackwell descriptors expect scalar int32 offsets
    pid_b_32 = pid_b.to(tl.int32)
    pid_h_32 = pid_h.to(tl.int32)
    offset_m = (pid_m * BLOCK_M).to(tl.int32)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    q_mask = offs_m < S
    
    # 4D TMA Load mapped directly to 2D
    q_4d = q_desc.load([pid_b_32, pid_h_32, offset_m, 0])
    q = tl.reshape(q_4d, [BLOCK_M, D])
    
    dtype = q.dtype
    neg_inf = float("-inf")
    
    m_i = tl.full((BLOCK_M,), neg_inf, tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, D), tl.float32)

    n_tiles = tl.cdiv(S, BLOCK_N)
    
    # Iterate with dedicated load loop-stages decoupled from dot pipelining
    for kv_tile in tl.range(0, n_tiles, num_stages=LOOP_STAGES):
        offset_n = (kv_tile * BLOCK_N).to(tl.int32)
        
        # TMA 4D Loads implicitly handle memory safety & zero-padding out-of-bounds automatically
        k_4d = k_desc.load([pid_b_32, pid_h_32, offset_n, 0])
        v_4d = v_desc.load([pid_b_32, pid_h_32, offset_n, 0])
        
        # Reshape isolated K, V 4D tiles logically down to 2D (zero-cost)
        k = tl.reshape(k_4d, [BLOCK_N, D])
        v = tl.reshape(v_4d, [BLOCK_N, D])
        
        # Q @ K^T Accumulation
        scores = tl.dot(q, k.T, out_dtype=tl.float32) * SCALE
        
        # Softmax safety sequence mask handling (only for preventing padded 0s artificially triggering calculations)
        curr_offs_n = kv_tile * BLOCK_N + tl.arange(0, BLOCK_N)
        valid_score = q_mask[:, None] & (curr_offs_n[None, :] < S)
        scores = tl.where(valid_score, scores, neg_inf)
        
        # Update running max
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        safe_m_ij = tl.where(m_ij == neg_inf, 0.0, m_ij)
        
        # Compute stabilized exponential scaling via natively supported base-2 limits
        alpha = tl.exp2(m_i - safe_m_ij)
        p = tl.exp2(scores - safe_m_ij[:, None])
        
        # Update normalization factors & standard attention dot product (P @ V)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(dtype), v, acc, out_dtype=tl.float32)
        
        m_i = m_ij

    # Safe division & log-sum-exp reconstruction to Natural Log (LSE consistency requirement)
    LN2: tl.constexpr = 0.6931471805599453
    safe_l_i = tl.where(l_i == 0.0, 1.0, l_i)
    output = acc / safe_l_i[:, None]
    
    lse_base2 = tl.where(l_i == 0.0, neg_inf, m_i + tl.math.log2(safe_l_i))
    lse_ln = lse_base2 * LN2
    
    # Store finalized computations accurately bypassing TMA stores to maintain deterministic simple writes
    offs_d = tl.arange(0, D)
    offs_m_64 = offs_m.to(tl.int64)
    offs_d_64 = offs_d.to(tl.int64)
    pid_b_64 = pid_b.to(tl.int64)
    pid_h_64 = pid_h.to(tl.int64)
    
    o_offset = pid_b_64 * stride_ob + pid_h_64 * stride_oh
    o_ptrs = O + o_offset + offs_m_64[:, None] * stride_os + offs_d_64[None, :] * stride_od
    tl.store(o_ptrs, output.to(dtype), mask=q_mask[:, None])
    
    lse_offset = pid_b_64 * stride_lseb + pid_h_64 * stride_lseh
    lse_ptrs = LSE + lse_offset + offs_m_64 * stride_lses
    tl.store(lse_ptrs, lse_ln, mask=q_mask)


def run(Q, K, V, O, LSE):
    """
    Executes standard Online-Softmax TMA block-pipelined FlashAttention on Blackwell hardware. 
    Uses destination passing; respects stream/device boundaries.
    """
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    # Premultiply by 1/ln(2) offline alongside scale to keep base-2 conversions unified globally
    softmax_scale = 1.0 / math.sqrt(D)
    RCP_LN2 = 1.4426950408889634
    SCALE = softmax_scale * RCP_LN2
    
    grid = lambda META: (
        triton.cdiv(S, META["BLOCK_M"]),
        B,
        H,
    )
    
    _fwd_kernel[grid](
        None, None, None,
        Q, K, V,
        O, LSE,
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        S, SCALE,
        D=D
    )