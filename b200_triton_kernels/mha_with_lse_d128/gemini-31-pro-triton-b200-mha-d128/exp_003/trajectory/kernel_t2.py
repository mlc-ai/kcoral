import math
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _attn_fwd_kernel(
    Q_desc, K_desc, V_desc, O_desc, LSE,
    sm_scale,
    stride_lsez, stride_lseh, stride_lses,
    B, H, S, D_HEAD: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    EVEN_M: tl.constexpr, EVEN_N: tl.constexpr
):
    start_m = tl.program_id(0)
    batch_head = tl.program_id(1)
    
    batch_idx = batch_head // H
    head_idx = batch_head % H
    
    # Load Q tile using 4D TMA descriptor for zero-overhead stride handling
    q_4d = Q_desc.load([batch_idx, head_idx, start_m * BLOCK_M, 0])
    q = tl.reshape(q_4d, [BLOCK_M, D_HEAD])
    
    # Initialize softmax states
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D_HEAD], dtype=tl.float32)
    
    offs_n_base = tl.arange(0, BLOCK_N)
    num_full_blocks = S // BLOCK_N
    
    # Process full blocks without boundary checks
    for start_n in range(0, num_full_blocks * BLOCK_N, BLOCK_N):
        # TMA descriptor loads for K and V
        k_4d = K_desc.load([batch_idx, head_idx, start_n, 0])
        v_4d = V_desc.load([batch_idx, head_idx, start_n, 0])
        
        k = tl.reshape(k_4d, [BLOCK_N, D_HEAD])
        v = tl.reshape(v_4d, [BLOCK_N, D_HEAD])
        
        # Q @ K^T
        qk = tl.dot(q, k.T) * sm_scale
        
        # Softmax computation
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        
        # Accumulate P @ V
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc=acc)
        
        m_i = m_ij

    # Process the tail block if sequence length is not fully divisible
    if not EVEN_N:
        start_n = num_full_blocks * BLOCK_N
        if start_n < S:
            k_4d = K_desc.load([batch_idx, head_idx, start_n, 0])
            v_4d = V_desc.load([batch_idx, head_idx, start_n, 0])
            
            k = tl.reshape(k_4d, [BLOCK_N, D_HEAD])
            v = tl.reshape(v_4d, [BLOCK_N, D_HEAD])
            
            qk = tl.dot(q, k.T) * sm_scale
            
            # Mask out-of-bounds padded keys to -inf
            mask_k = (start_n + offs_n_base) < S
            qk = tl.where(mask_k[None, :], qk, float("-inf"))
            
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            p = tl.exp(qk - m_ij[:, None])
            l_ij = tl.sum(p, 1)
            
            alpha = tl.exp(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            
            acc = acc * alpha[:, None]
            acc = tl.dot(p.to(tl.bfloat16), v, acc=acc)
            
            m_i = m_ij

    # Finalize Probabilities and LSE
    acc = acc / l_i[:, None]
    lse = m_i + tl.log(l_i)
    
    # Store O using TMA descriptor
    acc_4d = tl.reshape(acc, [1, 1, BLOCK_M, D_HEAD])
    O_desc.store([batch_idx, head_idx, start_m * BLOCK_M, 0], acc_4d.to(tl.bfloat16))
    
    # Store LSE using masked pointers
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
    Highly optimized multi-head non-causal attention forward pass targeting NVIDIA Blackwell.
    Uses 4D host TMA descriptors to perfectly match the memory layout, entirely bypassing the need 
    to slice or view arbitrary tensor strides, ensuring zero overhead natively mapped TMA logic.
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    sm_scale = 1.0 / math.sqrt(D)

    # Blackwell optimal execution parameters. 
    # BLOCK_M=128 and BLOCK_N=128 provide massive MAC utilization while exactly
    # fitting the 3 pipelined stages strictly within the 228KiB SMEM limit of the SM100.
    BLOCK_M = 128
    BLOCK_N = 128
    num_warps = 8
    num_stages = 3

    # Dynamic Host TMA descriptor allocation using structural physical shapes
    Q_desc = TensorDescriptor.from_tensor(Q, [1, 1, BLOCK_M, D])
    K_desc = TensorDescriptor.from_tensor(K, [1, 1, BLOCK_N, D])
    V_desc = TensorDescriptor.from_tensor(V, [1, 1, BLOCK_N, D])
    O_desc = TensorDescriptor.from_tensor(O, [1, 1, BLOCK_M, D])

    # Compile-time evaluation facts to explicitly prune bounds-checking out of the hotloop
    EVEN_M = (S % BLOCK_M == 0)
    EVEN_N = (S % BLOCK_N == 0)

    # Grid unravels M independently as contiguous block mapping promoting L2 caching characteristics for K/V loads
    grid = (triton.cdiv(S, BLOCK_M), B * H)
    
    _attn_fwd_kernel[grid](
        Q_desc, K_desc, V_desc, O_desc, LSE,
        sm_scale,
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        B, H, S, D_HEAD=D,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        EVEN_M=EVEN_M, EVEN_N=EVEN_N,
        num_warps=num_warps, num_stages=num_stages
    )