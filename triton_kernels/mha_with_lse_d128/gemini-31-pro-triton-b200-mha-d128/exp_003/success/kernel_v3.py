import math
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.autotune(
    configs=[
        # Optimized Blackwell warp-specialized configurations
        triton.Config({"WARP_SPECIALIZE": True, "NUM_STAGES_LOOP": 3}, num_warps=8, num_stages=3),
        triton.Config({"WARP_SPECIALIZE": True, "NUM_STAGES_LOOP": 4}, num_warps=8, num_stages=4),
        triton.Config({"WARP_SPECIALIZE": True, "NUM_STAGES_LOOP": 3}, num_warps=4, num_stages=3),
        triton.Config({"WARP_SPECIALIZE": True, "NUM_STAGES_LOOP": 4}, num_warps=4, num_stages=4),
        
        # Fallbacks without warp specialization to ensure robustness and correctness baseline
        triton.Config({"WARP_SPECIALIZE": False, "NUM_STAGES_LOOP": 3}, num_warps=8, num_stages=3),
        triton.Config({"WARP_SPECIALIZE": False, "NUM_STAGES_LOOP": 4}, num_warps=8, num_stages=4),
        triton.Config({"WARP_SPECIALIZE": False, "NUM_STAGES_LOOP": 3}, num_warps=4, num_stages=3),
    ],
    key=["S"],
)
@triton.jit
def _attn_fwd_kernel(
    Q_desc, K_desc, V_desc, O_desc, LSE,
    scale, log2_e,
    stride_lsez, stride_lseh, stride_lses,
    B, H, S, D_HEAD: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    EVEN_M: tl.constexpr, EVEN_N: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr, NUM_STAGES_LOOP: tl.constexpr
):
    # Using start_m as the fast-changing axis naturally groups sequence-level CTAs 
    # to perfectly strike L2 cache residency for loaded K and V chunks.
    start_m = tl.program_id(0)
    batch_head = tl.program_id(1)
    
    batch_idx = batch_head // H
    head_idx = batch_head % H
    
    # Standard Blackwell Host TMA descriptor loads avoiding indexing bounds overhead
    q_4d = Q_desc.load([batch_idx, head_idx, start_m * BLOCK_M, 0])
    q = tl.reshape(q_4d, [BLOCK_M, D_HEAD])
    
    # Initialize softmax sequence variables
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D_HEAD], dtype=tl.float32)
    
    num_full_blocks = S // BLOCK_N
    
    # Highly optimal loop: warp-specialized natively on SM100 limits latency by detaching TMA requests and Math
    for block_idx in tl.range(0, num_full_blocks, num_stages=NUM_STAGES_LOOP, warp_specialize=WARP_SPECIALIZE):
        start_n = block_idx * BLOCK_N
        
        k_4d = K_desc.load([batch_idx, head_idx, start_n, 0])
        v_4d = V_desc.load([batch_idx, head_idx, start_n, 0])
        
        k = tl.reshape(k_4d, [BLOCK_N, D_HEAD])
        v = tl.reshape(v_4d, [BLOCK_N, D_HEAD])
        
        # Scaling dot inline using base-2 math leverages optimal pipeline acceleration
        qk = tl.dot(q, k.T) * scale
        
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        
        alpha = tl.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc=acc)
        
        m_i = m_ij

    # Residual sequence chunk avoiding boundary penalties in the core loop
    if not EVEN_N:
        start_n = num_full_blocks * BLOCK_N
        if start_n < S:
            k_4d = K_desc.load([batch_idx, head_idx, start_n, 0])
            v_4d = V_desc.load([batch_idx, head_idx, start_n, 0])
            
            k = tl.reshape(k_4d, [BLOCK_N, D_HEAD])
            v = tl.reshape(v_4d, [BLOCK_N, D_HEAD])
            
            qk = tl.dot(q, k.T) * scale
            
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

    # Convert tracked maximums mapping strictly back to natural log domain (LSE) for PyTorch precision limits
    acc = acc / l_i[:, None]
    lse = (m_i + tl.log2(l_i)) / log2_e
    
    # Output stores utilizing precise shape mapping back strictly through 4D TMA descriptor bounds
    acc_4d = tl.reshape(acc, [1, 1, BLOCK_M, D_HEAD])
    O_desc.store([batch_idx, head_idx, start_m * BLOCK_M, 0], acc_4d.to(tl.bfloat16))
    
    # Store output LSE with pointers using masked limits where natively needed
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
    Computes Non-causal Multi-Head Attention explicitly targeting highly optimized NVIDIA Blackwell WGMMA properties.
    Utilizes Host TMA Descriptors exclusively across static layouts. Inner loop exploits unified Math Base2
    transformations mapping to hardware natively. Outputs are written strictly into preallocated destination tensors.
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    sm_scale = 1.0 / math.sqrt(D)
    log2_e = 1.4426950408889634
    
    # Combines scaling and base conversion ensuring internal fp32 math operates purely with exp2
    scale = sm_scale * log2_e

    # Locked shape optimizations fitting SM100 TMEM / SMEM balance seamlessly avoiding autotune descriptor re-allocations
    BLOCK_M = 128
    BLOCK_N = 128
    
    # Create 4D host views natively translating PyTorch contiguous memory stride layers right into TMAs hardware unit
    Q_desc = TensorDescriptor.from_tensor(Q, [1, 1, BLOCK_M, D])
    K_desc = TensorDescriptor.from_tensor(K, [1, 1, BLOCK_N, D])
    V_desc = TensorDescriptor.from_tensor(V, [1, 1, BLOCK_N, D])
    O_desc = TensorDescriptor.from_tensor(O, [1, 1, BLOCK_M, D])

    # Compiler constant assertions statically removing bounds loops implicitly across divisible lengths
    EVEN_M = (S % BLOCK_M == 0)
    EVEN_N = (S % BLOCK_N == 0)

    # Structuring thread grid guaranteeing L2 memory hits are maximized horizontally across shared K/V sequences natively
    grid = (triton.cdiv(S, BLOCK_M), B * H)
    
    _attn_fwd_kernel[grid](
        Q_desc, K_desc, V_desc, O_desc, LSE,
        scale, log2_e,
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        B, H, S, D_HEAD=D,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        EVEN_M=EVEN_M, EVEN_N=EVEN_N
    )