import math
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _attn_fwd_kernel(
    Q_desc, K_desc, V_desc, O_desc,
    LSE,
    scale, ln_2,
    stride_lsez, stride_lseh, stride_lses,
    B, H, S, D_HEAD: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    EVEN_M: tl.constexpr, EVEN_N: tl.constexpr
):
    # Process blocks across sequence length grouped naturally by batch and head.
    # The 2D grid ensures all M chunks for a given Head execute temporally close, 
    # practically guaranteeing optimal L2 cache residency for loaded K and V tiles.
    start_m = tl.program_id(0)
    batch_head = tl.program_id(1)
    
    batch_idx = batch_head // H
    head_idx = batch_head % H
    
    # Standard 4D TMA descriptors inherently resolve indexing limits with zero overhead
    q_4d = Q_desc.load([batch_idx, head_idx, start_m * BLOCK_M, 0])
    q = tl.reshape(q_4d, [BLOCK_M, D_HEAD])
    
    # Pre-scale Q completely outside the hotloop and convert back to bfloat16.
    # This mathematically avoids millions of elementwise fp32 scaling operations inside the TMA pipeline.
    q = (q.to(tl.float32) * scale).to(tl.bfloat16)
    
    # Initialize running softmax sequence tracking
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D_HEAD], dtype=tl.float32)
    
    num_full_blocks = S // BLOCK_N
    
    # Pipelined sequence loop tightly integrated with 3-stage hardware execution
    for block_idx in tl.range(0, num_full_blocks, num_stages=3):
        start_n = block_idx * BLOCK_N
        
        k_4d = K_desc.load([batch_idx, head_idx, start_n, 0])
        v_4d = V_desc.load([batch_idx, head_idx, start_n, 0])
        
        k = tl.reshape(k_4d, [BLOCK_N, D_HEAD])
        v = tl.reshape(v_4d, [BLOCK_N, D_HEAD])
        
        # Blackwell Native Base-2 Accelerated WGMMA Dot Product
        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        
        # Mathematically equivalent Base-2 softmax statistics
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        
        alpha = tl.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc=acc, out_dtype=tl.float32)
        
        m_i = m_ij

    # Resolves tail sequences, dynamically stripped completely by compiler if purely divisible
    if not EVEN_N:
        start_n = num_full_blocks * BLOCK_N
        if start_n < S:
            k_4d = K_desc.load([batch_idx, head_idx, start_n, 0])
            v_4d = V_desc.load([batch_idx, head_idx, start_n, 0])
            
            k = tl.reshape(k_4d, [BLOCK_N, D_HEAD])
            v = tl.reshape(v_4d, [BLOCK_N, D_HEAD])
            
            qk = tl.dot(q, k.T, out_dtype=tl.float32)
            
            offs_n_base = tl.arange(0, BLOCK_N)
            mask_k = (start_n + offs_n_base) < S
            qk = tl.where(mask_k[None, :], qk, float("-inf"))
            
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            p = tl.exp2(qk - m_ij[:, None])
            l_ij = tl.sum(p, 1)
            
            alpha = tl.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            
            acc = acc * alpha[:, None]
            acc = tl.dot(p.to(tl.bfloat16), v, acc=acc, out_dtype=tl.float32)
            
            m_i = m_ij

    # Fast reciprocals strictly circumventing division latency costs
    inv_l_i = 1.0 / l_i
    acc = acc * inv_l_i[:, None]
    
    # Backscale strictly from Base-2 LSE back into Base-e PyTorch Natural Log Space format
    lse = (m_i + tl.log2(l_i)) * ln_2
    
    # Outbound storage using natively bounds-checked 4D TMA properties implicitly
    acc_4d = tl.reshape(acc, [1, 1, BLOCK_M, D_HEAD])
    O_desc.store([batch_idx, head_idx, start_m * BLOCK_M, 0], acc_4d.to(tl.bfloat16))
    
    # Point-based LSE storage incorporating mask bounds inherently avoiding invalid writes
    lse_offset = batch_idx * stride_lsez + head_idx * stride_lseh
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    lse_ptrs = LSE + lse_offset + offs_m * stride_lses
    
    if EVEN_M:
        tl.store(lse_ptrs, lse)
    else:
        mask_m = offs_m < S
        tl.store(lse_ptrs, lse, mask=mask_m)


def run(Q, K, V, O, LSE):
    """
    Computes Non-causal Multi-Head Attention mapping exactly to optimized NVIDIA Blackwell WGMMA limits.
    Utilizes pipelined Host TMA Descriptors and intrinsic FP32 Base-2 calculations targeting SM100 natively.
    Outputs strictly pass directly to their definition tensors with no intervening allocations.
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    sm_scale = 1.0 / math.sqrt(D)
    
    log2_e = 1.4426950408889634
    ln_2 = 0.6931471805599453
    # Merges model scaling and exponent base conversion to minimize elementwise instructions
    scale = sm_scale * log2_e
    
    # 128x128 blocking strategy fits the precise SMEM limits implicitly mapped to the 3-stage TMAs cleanly 
    # without blowing past the 228KiB limit on Blackwell (32K Q + 192K K/V pipelining == 224KB).
    BLOCK_M = 128
    BLOCK_N = 128
    
    # 4D descriptor alignment allows the exact bounding characteristics to function without arbitrary slices.
    Q_desc = TensorDescriptor.from_tensor(Q, [1, 1, BLOCK_M, D])
    K_desc = TensorDescriptor.from_tensor(K, [1, 1, BLOCK_N, D])
    V_desc = TensorDescriptor.from_tensor(V, [1, 1, BLOCK_N, D])
    O_desc = TensorDescriptor.from_tensor(O, [1, 1, BLOCK_M, D])

    # Compiler evaluates explicitly pruning away conditional blocks completely if lengths are purely divisible.
    EVEN_M = (S % BLOCK_M == 0)
    EVEN_N = (S % BLOCK_N == 0)

    # 2D Gridding guarantees iterative sequencing over Head subsets matching temporal blocks resolving L2 structurally natively.
    grid = (triton.cdiv(S, BLOCK_M), B * H)
    
    _attn_fwd_kernel[grid](
        Q_desc, K_desc, V_desc, O_desc, LSE,
        scale, ln_2,
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        B, H, S, D_HEAD=D,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        EVEN_M=EVEN_M, EVEN_N=EVEN_N,
        num_warps=8,
        num_stages=3
    )