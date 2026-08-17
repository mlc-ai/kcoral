import torch
import triton
import triton.language as tl

# Host-level infrastructure allocator for Triton device-side descriptor storage
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
    ],
    key=['seq_len'],
)
@triton.jit
def _attn_fwd_kernel(
    Q, K, V, sm_scale,
    O, LSE,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    B, H, seq_len, D,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    start_m = pid_m * BLOCK_M
    if start_m >= seq_len:
        return
        
    b = pid_bh // H
    h = pid_bh % H
    
    # Formulate strictly 4D TMA descriptors allowing hardware bound-checking and L2 prefetching
    # without redundant manual element offset computations.
    q_desc = tl.make_tensor_descriptor(
        Q, shape=[B, H, seq_len, D], strides=[stride_qb, stride_qh, stride_qs, stride_qd],
        block_shape=[1, 1, BLOCK_M, BLOCK_D], padding_option="zero"
    )
    k_desc = tl.make_tensor_descriptor(
        K, shape=[B, H, seq_len, D], strides=[stride_kb, stride_kh, stride_ks, stride_kd],
        block_shape=[1, 1, BLOCK_N, BLOCK_D], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        V, shape=[B, H, seq_len, D], strides=[stride_vb, stride_vh, stride_vs, stride_vd],
        block_shape=[1, 1, BLOCK_N, BLOCK_D], padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        O, shape=[B, H, seq_len, D], strides=[stride_ob, stride_oh, stride_os, stride_od],
        block_shape=[1, 1, BLOCK_M, BLOCK_D], padding_option="zero"
    )
    
    q = q_desc.load([b, h, start_m, 0])
    
    # Ensure rigorous FP32 accumulators retaining safe normalization constants
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    limit_n = tl.minimum(start_m, seq_len)
    limit_n_aligned = (limit_n // BLOCK_N) * BLOCK_N
    
    # Phase 1: Fully unmasked blocks (highly pipelined hardware native math sequence)
    for start_n in tl.range(0, limit_n_aligned, BLOCK_N, num_stages=3):
        k = k_desc.load([b, h, start_n, 0])
        v = v_desc.load([b, h, start_n, 0])
        
        # Scaling directly in FP32 guarantees precision equivalency to PyTorch's native SDPA
        qk = tl.dot(q, k.T, out_dtype=tl.float32) * sm_scale
        
        m_ij = tl.max(qk, 1)
        m_new = tl.maximum(m_i, m_ij)
        
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(qk - m_new[:, None])
        
        l_i = l_i * alpha + tl.sum(beta, 1)
        
        acc = acc * alpha[:, None]
        acc = tl.dot(beta.to(tl.bfloat16), v, acc)
        
        m_i = m_new

    # Phase 2: Causal blocks physically intersecting the sequence boundary
    causal_end = tl.minimum(start_m + BLOCK_M, seq_len)
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    
    for start_n in range(limit_n_aligned, causal_end, BLOCK_N):
        k = k_desc.load([b, h, start_n, 0])
        v = v_desc.load([b, h, start_n, 0])
        
        qk = tl.dot(q, k.T, out_dtype=tl.float32) * sm_scale
        
        # Causal triangular isolation
        offs_n_curr = start_n + offs_n
        causal_mask = offs_m[:, None] >= offs_n_curr[None, :]
        qk = tl.where(causal_mask, qk, float("-inf"))
        
        m_ij = tl.max(qk, 1)
        m_new = tl.maximum(m_i, m_ij)
        
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(qk - m_new[:, None])
        
        l_i = l_i * alpha + tl.sum(beta, 1)
        
        acc = acc * alpha[:, None]
        acc = tl.dot(beta.to(tl.bfloat16), v, acc)
        
        m_i = m_new

    # Final scaling & log-sum-exp derivation
    inv_l_i = 1.0 / l_i
    acc = acc * inv_l_i[:, None]
    lse = m_i + tl.log(l_i)
    
    # Store matrix output bounds via localized TMA natively resolving limits
    o_desc.store([b, h, start_m, 0], acc.to(tl.bfloat16))
    
    # Store LSE linearly mapping sequence bounds explicitly
    lse_offset = b * stride_lseb + h * stride_lseh
    lse_ptrs = LSE + lse_offset + offs_m * stride_lses
    tl.store(lse_ptrs, lse, mask=offs_m < seq_len)

def run(Q, K, V, O, LSE):
    """
    Computes rigorous causal Multi-Head Attention forward pass on NVIDIA Blackwell.
    Accelerated with native Tensor Memory TMA bounded via localized 4D shape caching, 
    L2 broadcast alignments minimizing cross-streaming, and FP32 precise pipelined SDPA logic.
    """
    torch.cuda.set_device(Q.device)
    
    # TMA descriptors must be registered via framework allocator logic securely
    triton.set_allocator(alloc_fn)
    
    B, H, S, D = Q.shape
    if S == 0:
        return
        
    sm_scale = 1.0 / (D ** 0.5)
    
    # Structured to iterate Sequence elements internally, ensuring L2 shared broadcast hits
    # concurrently across aligned SM hardware processing matching batch and heads targets
    grid = lambda META: (
        triton.cdiv(S, META['BLOCK_M']),
        B * H,
        1
    )
    
    _attn_fwd_kernel[grid](
        Q, K, V, sm_scale,
        O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        B, H, S, D,
        BLOCK_D=128
    )