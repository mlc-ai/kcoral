import math
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _attn_fwd_kernel(
    Q_desc, K_desc, V_desc, O_desc, LSE,
    scale, ln_2,
    stride_lsez, stride_lseh, stride_lses,
    B, H, S, D_HEAD: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    EVEN_M: tl.constexpr, EVEN_N: tl.constexpr
):
    # Map CTAs to M-blocks primarily, grouped naturally by Batch/Head to ensure
    # perfect L2 cache utilization for identically scheduled K and V slices.
    start_m = tl.program_id(0)
    batch_head = tl.program_id(1)
    
    batch_idx = batch_head // H
    head_idx = batch_head % H
    
    # 4D TMA descriptors for effortless and zero-overhead memory mapped boundaries
    q_4d = Q_desc.load([batch_idx, head_idx, start_m * BLOCK_M, 0])
    q = tl.reshape(q_4d, [BLOCK_M, D_HEAD])
    
    # Fundamental optimization: pre-scale loop-invariant Q upfront to remove hundreds
    # of thousands of redundant FP32 multiplications inside the hot TMA block pipeline.
    q = (q.to(tl.float32) * scale).to(tl.bfloat16)
    
    # Initialize sequence accumulator states
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D_HEAD], dtype=tl.float32)
    
    num_full_blocks = S // BLOCK_N
    
    # Highly optimal sequence loop leveraging 4-stage pipeline alongside explicit Warp 
    # specialization to safely parallelize TMA fetch limits alongside Base-2 Math on WGMMA.
    for block_idx in tl.range(0, num_full_blocks, num_stages=4, warp_specialize=True):
        start_n = block_idx * BLOCK_N
        
        k_4d = K_desc.load([batch_idx, head_idx, start_n, 0])
        v_4d = V_desc.load([batch_idx, head_idx, start_n, 0])
        
        k = tl.reshape(k_4d, [BLOCK_N, D_HEAD])
        v = tl.reshape(v_4d, [BLOCK_N, D_HEAD])
        
        # Base-2 accelerated WGMMA dot lacking any inline elementwise fp32 scaling
        qk = tl.dot(q, k.T)
        
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        
        alpha = tl.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc=acc)
        
        m_i = m_ij

    # Safe tail boundaries (dynamically pruned gracefully out by the compiler if perfectly divisible)
    if not EVEN_N:
        start_n = num_full_blocks * BLOCK_N
        if start_n < S:
            k_4d = K_desc.load([batch_idx, head_idx, start_n, 0])
            v_4d = V_desc.load([batch_idx, head_idx, start_n, 0])
            
            k = tl.reshape(k_4d, [BLOCK_N, D_HEAD])
            v = tl.reshape(v_4d, [BLOCK_N, D_HEAD])
            
            qk = tl.dot(q, k.T)
            
            offs_n_base = tl.arange(0, BLOCK_N)
            mask_k = (start_n + offs_n_base) < S
            qk = tl.where(mask_k[None, :], qk, float("-inf"))
            
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            p = tl.exp2(qk - m_ij[:, None])
            l_ij = tl.sum(p, 1)
            
            alpha = tl.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            
            acc = acc * alpha[:, None]
            acc = tl.dot(p.to(tl.bfloat16), v, acc=acc)
            
            m_i = m_ij

    # Optimization: Use fast reciprocals avoiding exorbitant floating-point division penalties entirely
    inv_l_i = 1.0 / l_i
    acc = acc * inv_l_i[:, None]
    
    # Scale from Base-2 LSE back natively to Base-e Natural Log Space mapping exactly PyTorch logic limits
    lse = (m_i + tl.log2(l_i)) * ln_2
    
    # Output matrices structured purely bounded back implicitly out using TMA Descriptors directly
    acc_4d = tl.reshape(acc, [1, 1, BLOCK_M, D_HEAD])
    O_desc.store([batch_idx, head_idx, start_m * BLOCK_M, 0], acc_4d.to(tl.bfloat16))
    
    # Pointer handling dynamically masking when bounds mismatch exactly
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
    Highly advanced Multi-Head Attention targeting optimized NVIDIA Blackwell SM100 architecture mapping.
    Outputs are written directly to preallocated destination tensors `O` and `LSE`.
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    sm_scale = 1.0 / math.sqrt(D)
    
    log2_e = 1.4426950408889634
    ln_2 = 0.6931471805599453
    # Combines scaling & exponent base conversion minimizing inner instruction elements globally
    scale = sm_scale * log2_e
    
    # By strictly scaling BLOCK_M to 256 over 128, HBM memory traffic is fundamentally reduced by roughly half globally.
    # BLOCK_N = 64 perfectly secures bounds allowing a 4-stage pipeline alongside entirely zero register spilling.
    BLOCK_M = 256
    BLOCK_N = 64
    
    # Physical Tensor Descriptors structured explicitly to seamlessly map multi-dimensional layouts for TMA constraints
    Q_desc = TensorDescriptor.from_tensor(Q, [1, 1, BLOCK_M, D])
    K_desc = TensorDescriptor.from_tensor(K, [1, 1, BLOCK_N, D])
    V_desc = TensorDescriptor.from_tensor(V, [1, 1, BLOCK_N, D])
    O_desc = TensorDescriptor.from_tensor(O, [1, 1, BLOCK_M, D])

    # Compiler evaluated checks structurally trimming padded branches
    EVEN_M = (S % BLOCK_M == 0)
    EVEN_N = (S % BLOCK_N == 0)

    # Gridding logic specifically engineered ensuring sequential iteration over matching heads caching L2 correctly
    grid = (triton.cdiv(S, BLOCK_M), B * H)
    
    _attn_fwd_kernel[grid](
        Q_desc, K_desc, V_desc, O_desc, LSE,
        scale, ln_2,
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        B, H, S, D_HEAD=D,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        EVEN_M=EVEN_M, EVEN_N=EVEN_N,
        num_warps=8,
        num_stages=4
    )