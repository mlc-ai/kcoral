import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

# Provide Triton's allocator for creating TMA descriptors dynamically on the device
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "LOOP_STAGES": 2}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "LOOP_STAGES": 4}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "LOOP_STAGES": 3}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "LOOP_STAGES": 3}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "LOOP_STAGES": 4}, num_warps=4, num_stages=4),
    ],
    key=["S"],
)
@triton.jit
def _mha_fwd_kernel(
    Q, K, V, O, LSE,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    S, H, scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
    LOOP_STAGES: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    b = pid_bh // H
    h = pid_bh % H
    
    start_m = pid_m * BLOCK_M
    
    # Compute base pointers for the specific batch and head
    q_base = Q + b * stride_qb + h * stride_qh
    k_base = K + b * stride_kb + h * stride_kh
    v_base = V + b * stride_vb + h * stride_vh
    o_base = O + b * stride_ob + h * stride_oh
    
    # Create tensor descriptors for robust TMA lowering
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
    o_desc = tl.make_tensor_descriptor(
        o_base, shape=[S, D], strides=[stride_os, 1],
        block_shape=[BLOCK_M, D], padding_option="zero"
    )
    
    # Load Q tile and pre-scale in base-2 units
    q = q_desc.load([start_m, 0])
    
    RCP_LN2: tl.constexpr = 1.4426950408889634
    scale_base2 = scale * RCP_LN2
    q = (q.to(tl.float32) * scale_base2).to(tl.bfloat16)
    
    # Initialize online-softmax state
    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)
    
    # Full block iterations for optimal pipelining
    limit = (S // BLOCK_N) * BLOCK_N
    
    for start_n in tl.range(0, limit, BLOCK_N, num_stages=LOOP_STAGES):
        k = k_desc.load([start_n, 0])
        v = v_desc.load([start_n, 0])
        
        scores = tl.dot(q, k.T)
        
        # Omit `-inf` recovery here as valid full block key vectors inherently guarantee max logic safety
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        
        alpha = tl.math.exp2(m_i - m_ij)
        p = tl.math.exp2(scores - m_ij[:, None])
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        m_i = m_ij

    # Specialized tail block iteration if sequence length is not purely divisible
    if S % BLOCK_N != 0:
        start_n = limit
        k = k_desc.load([start_n, 0])
        v = v_desc.load([start_n, 0])
        
        scores = tl.dot(q, k.T)
        
        offs_n = start_n + tl.arange(0, BLOCK_N)
        valid_score = offs_n[None, :] < S
        scores = tl.where(valid_score, scores, -float("inf"))
        
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        
        alpha = tl.math.exp2(m_i - m_ij)
        p = tl.math.exp2(scores - m_ij[:, None])
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        m_i = m_ij

    # Finalize safe normalization and compute O
    safe_l_i = tl.where(l_i == 0.0, 1.0, l_i)
    inv_l_i = 1.0 / safe_l_i
    output = acc * inv_l_i[:, None]
    
    # Evaluate and write outputs: TMA implicitly guards out-of-bounds storage mappings internally
    o_desc.store([start_m, 0], output.to(tl.bfloat16))
    
    # Compute LSE mapped to the natural logarithm scale (matches PyTorch native cuDNN outputs)
    LN2: tl.constexpr = 0.6931471805599453
    lse_log2 = tl.where(l_i == 0.0, -float("inf"), m_i + tl.math.log2(safe_l_i))
    lse = lse_log2 * LN2
    
    offs_m = start_m + tl.arange(0, BLOCK_M)
    m_mask = offs_m < S
    
    lse_ptrs = LSE + b * stride_lseb + h * stride_lseh + offs_m * stride_lses
    tl.store(lse_ptrs, lse, mask=m_mask)


def run(Q, K, V, O, LSE):
    """Destination-passing multi-head attention forward directly updating O and LSE storage vectors."""
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    scale = D ** -0.5
    
    # Schedule grid: map sequence blocks contiguously across `x` enabling localized KV buffer reuse caching
    grid = lambda META: (triton.cdiv(S, META["BLOCK_M"]), B * H)
    
    _mha_fwd_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        S, H, scale,
        D=128
    )