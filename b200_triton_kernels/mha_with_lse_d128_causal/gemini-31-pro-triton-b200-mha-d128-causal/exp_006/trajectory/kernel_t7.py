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

    # Running statistics for Online Softmax (guaranteed finite following step 0)
    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # Load Q block via native TMA descriptor mapping
    q = q_desc.load([b, h, start_m, 0])
    q = tl.reshape(q, [BLOCK_M, BLOCK_D])
    
    # Pre-scale Q to fold standard softmax evaluation and base-2 exp scaling
    q = (q.to(tl.float32) * SCALE).to(tl.bfloat16)

    max_n = tl.minimum(S, start_m + BLOCK_M)
    num_steps = (max_n + BLOCK_N - 1) // BLOCK_N
    
    # Fast path: blocks securely beneath the causal diagonal (no padding or masking evaluated)
    num_full_steps = tl.minimum(start_m // BLOCK_N, num_steps)

    for step in tl.range(0, num_full_steps, num_stages=3):
        start_n = step * BLOCK_N
        
        # TMA prefetch streaming optimally pipelined
        k = k_desc.load([b, h, start_n, 0])
        k = tl.reshape(k, [BLOCK_N, BLOCK_D])
        v = v_desc.load([b, h, start_n, 0])
        v = tl.reshape(v, [BLOCK_N, BLOCK_D])

        # Core dot product dynamically mapping K transposed implicitly handled by Blackwell
        qk = tl.dot(q, k.T, out_dtype=tl.float32)

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        
        # Evaluated safely strictly across bounding TF32 precision capacities 
        p = tl.math.exp2(qk - m_ij[:, None])
        alpha = tl.math.exp2(m_i - m_ij)
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        m_i = m_ij

    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    # Slow path: traversing causal diagonal elements
    for step in range(num_full_steps, num_steps):
        start_n = step * BLOCK_N
        curr_offs_n = start_n + offs_n

        k = k_desc.load([b, h, start_n, 0])
        k = tl.reshape(k, [BLOCK_N, BLOCK_D])
        v = v_desc.load([b, h, start_n, 0])
        v = tl.reshape(v, [BLOCK_N, BLOCK_D])

        qk = tl.dot(q, k.T, out_dtype=tl.float32)

        # Simplified mask omitting sequence length checks cleanly filtered safely on save
        mask = offs_m[:, None] >= curr_offs_n[None, :]
        qk = tl.where(mask, qk, -float("inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        
        p = tl.math.exp2(qk - m_ij[:, None])
        alpha = tl.math.exp2(m_i - m_ij)
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        m_i = m_ij

    # Safe bounds mathematically confirmed via step 0 minimum causality guarantees
    acc = acc / l_i[:, None]
    
    # Reproject logically to natural logarithmic boundaries expected by backward FA
    LN2: tl.constexpr = 0.6931471805599453
    lse = (m_i + tl.math.log2(l_i)) * LN2

    # Persist outputs safely truncating padded sequence out-of-bounds metrics natively
    acc_out = tl.reshape(acc.to(tl.bfloat16), [1, 1, BLOCK_M, BLOCK_D])
    o_desc.store([b, h, start_m, 0], acc_out)

    # Standard manual mask for fallback LSE pointer writes
    lse_ptrs = LSE + b * stride_lb + h * stride_lh + offs_m * stride_ls
    tl.store(lse_ptrs, lse, mask=offs_m < S)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    # Strictly validated mapping enforcing contiguous trailing dimension hardware standards
    assert Q.stride(-1) == 1
    assert K.stride(-1) == 1
    assert V.stride(-1) == 1
    assert O.stride(-1) == 1
    
    RCP_LN2 = 1.4426950408889634
    scale = float((1.0 / math.sqrt(D)) * RCP_LN2)
    
    BLOCK_M = 128
    BLOCK_N = 128
    
    # Unified consecutive memory access scheduling
    grid = (triton.cdiv(S, BLOCK_M), B, H)
    
    # TMA runtime tensor descriptor generation via explicit CPU bindings optimizing away dynamic allocation
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