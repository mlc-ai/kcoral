import torch
import triton
import triton.language as tl
import math
from triton.tools.tensor_descriptor import TensorDescriptor

@triton.jit
def _fwd_kernel(
    q_desc, k_desc, v_desc, o_desc,
    LSE,
    stride_lb, stride_lh, stride_ls,
    S,
    SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    start_m = tl.program_id(0) * BLOCK_M
    b = tl.program_id(1)
    h = tl.program_id(2)

    # Early exit for entirely out-of-bounds query blocks
    if start_m >= S:
        return

    # Running statistics for Online Softmax
    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # Load Q block via native TMA descriptor mapping
    q = q_desc.load([b, h, start_m, 0])
    q = tl.reshape(q, [BLOCK_M, BLOCK_D])

    max_n = tl.minimum(S, start_m + BLOCK_M)
    num_steps = (max_n + BLOCK_N - 1) // BLOCK_N
    
    # Fast path: exclusively blocks safely underneath the causal diagonal 
    # (Removes heavy padding evaluation or diagonal mask computations)
    num_full_steps = tl.minimum(start_m // BLOCK_N, num_steps)

    for step in tl.range(0, num_full_steps, num_stages=3):
        start_n = step * BLOCK_N
        
        # Prefetch pipelines TMA natively on Blackwell hardware
        k = k_desc.load([b, h, start_n, 0])
        k = tl.reshape(k, [BLOCK_N, BLOCK_D])
        v = v_desc.load([b, h, start_n, 0])
        v = tl.reshape(v, [BLOCK_N, BLOCK_D])

        # Compute dot product dynamically mapping K transposed
        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        qk = qk * SCALE

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        
        # Softmax evaluation scaling mapped to base-2 hardware intrinsic bounds
        p = tl.math.exp2(qk - m_ij[:, None])
        alpha = tl.math.exp2(m_i - m_ij)
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        
        # V evaluation safely mapped natively to bfloat16 memory matrix multiply requirements
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        m_i = m_ij

    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    # Slow path: overlapping causal diagonal mask evaluations
    for step in range(num_full_steps, num_steps):
        start_n = step * BLOCK_N
        curr_offs_n = start_n + offs_n

        k = k_desc.load([b, h, start_n, 0])
        k = tl.reshape(k, [BLOCK_N, BLOCK_D])
        v = v_desc.load([b, h, start_n, 0])
        v = tl.reshape(v, [BLOCK_N, BLOCK_D])

        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        qk = qk * SCALE

        # Sequence-based bounds ensuring causality limit
        mask = (offs_m[:, None] >= curr_offs_n[None, :]) & (curr_offs_n[None, :] < S) & (offs_m[:, None] < S)
        qk = tl.where(mask, qk, -float("inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        safe_m_ij = tl.where(m_ij == -float("inf"), 0.0, m_ij)
        
        p = tl.math.exp2(qk - safe_m_ij[:, None])
        alpha = tl.math.exp2(m_i - safe_m_ij)
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        m_i = m_ij

    safe_l_i = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / safe_l_i[:, None]
    
    # Reproject to natural logarithmic boundaries
    LN2: tl.constexpr = 0.6931471805599453
    lse_log2 = tl.where(l_i == 0.0, -float("inf"), m_i + tl.math.log2(safe_l_i))
    lse = lse_log2 * LN2

    # Drop back via TMA
    acc_out = tl.reshape(acc.to(tl.bfloat16), [1, 1, BLOCK_M, BLOCK_D])
    o_desc.store([b, h, start_m, 0], acc_out)

    # Standard mask LSE fallback memory pointer store
    lse_ptrs = LSE + b * stride_lb + h * stride_lh + offs_m * stride_ls
    tl.store(lse_ptrs, lse, mask=offs_m < S)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    # Restrictively ensures standard memory mappings
    assert Q.stride(-1) == 1
    assert K.stride(-1) == 1
    assert V.stride(-1) == 1
    assert O.stride(-1) == 1
    
    RCP_LN2 = 1.4426950408889634
    scale = float((1.0 / math.sqrt(D)) * RCP_LN2)
    
    BLOCK_M = 128
    BLOCK_N = 128
    
    # Optimally grouped scheduling per SM
    grid = (triton.cdiv(S, BLOCK_M), B, H)
    
    # Establish fully decoupled hardware-specific tensor memory references in PyTorch to avoid descriptor creation bounds at Triton runtime execution
    q_desc = TensorDescriptor.from_tensor(Q, [1, 1, BLOCK_M, D])
    k_desc = TensorDescriptor.from_tensor(K, [1, 1, BLOCK_N, D])
    v_desc = TensorDescriptor.from_tensor(V, [1, 1, BLOCK_N, D])
    o_desc = TensorDescriptor.from_tensor(O, [1, 1, BLOCK_M, D])
    
    _fwd_kernel[grid](
        q_desc, k_desc, v_desc, o_desc,
        LSE,
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        S,
        SCALE=scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=D,
        num_warps=8,
        num_stages=3
    )