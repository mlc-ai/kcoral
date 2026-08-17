import torch
import triton
import triton.language as tl
import math

# Configure Triton's device descriptor allocator for Blackwell TMA
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

def pre_hook(kwargs):
    # Determine at compile time whether the sequence divides cleanly into blocks
    kwargs['EVEN_M'] = (kwargs['N_CTX'] % kwargs['BLOCK_M'] == 0)
    kwargs['EVEN_N'] = (kwargs['N_CTX'] % kwargs['BLOCK_N'] == 0)

@triton.autotune(
    configs=[
        # Optimized configuration space avoiding SRAM overflow limits (228KB per SM on B200)
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'NUM_STAGES': 3}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'NUM_STAGES': 3}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'NUM_STAGES': 3}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'NUM_STAGES': 3}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'NUM_STAGES': 4}, num_stages=4, num_warps=4),
    ],
    key=['N_CTX'],
    pre_hook=pre_hook,
)
@triton.jit
def _attn_fwd_kernel(
    Q, K, V, sm_scale, O, LSE,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_on,
    stride_lsez, stride_lseh, stride_lsem,
    H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    EVEN_M: tl.constexpr, EVEN_N: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(N_CTX, BLOCK_M)
    
    # 1D Grid Swizzling: Group all M blocks sequentially for the same batch and head.
    # Dispatched CTAs will completely share K and V blocks simultaneously in the L2 cache!
    start_m = pid % grid_m
    off_hz = pid // grid_m
    
    off_z = off_hz // H
    off_h = off_hz % H
    
    # Base pointers matching this exact head sequence
    q_ptr = Q + off_z * stride_qz + off_h * stride_qh
    k_ptr = K + off_z * stride_kz + off_h * stride_kh
    v_ptr = V + off_z * stride_vz + off_h * stride_vh
    o_ptr = O + off_z * stride_oz + off_h * stride_oh
    
    # Instantiate Device Tensor Descriptors (natively activates Blackwell TMA and `tcgen05` Tensor Cores)
    q_desc = tl.make_tensor_descriptor(
        q_ptr, shape=[N_CTX, BLOCK_DMODEL], strides=[stride_qm, stride_qk],
        block_shape=[BLOCK_M, BLOCK_DMODEL], padding_option="zero"
    )
    k_desc = tl.make_tensor_descriptor(
        k_ptr, shape=[N_CTX, BLOCK_DMODEL], strides=[stride_kn, stride_kk],
        block_shape=[BLOCK_N, BLOCK_DMODEL], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        v_ptr, shape=[N_CTX, BLOCK_DMODEL], strides=[stride_vn, stride_vk],
        block_shape=[BLOCK_N, BLOCK_DMODEL], padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        o_ptr, shape=[N_CTX, BLOCK_DMODEL], strides=[stride_om, stride_on],
        block_shape=[BLOCK_M, BLOCK_DMODEL], padding_option="zero"
    )
    
    # Load entire queries segment natively utilizing TMA offsets
    q = q_desc.load([start_m * BLOCK_M, 0])
    
    # Mathematically exact FlashAttention V2 accumulators 
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    
    offs_n = tl.arange(0, BLOCK_N)
    
    # Context loop pipelined seamlessly via TMA asynchronous dispatches
    for start_n_idx in tl.range(0, tl.cdiv(N_CTX, BLOCK_N), num_stages=NUM_STAGES):
        n_offset = start_n_idx * BLOCK_N
        
        k = k_desc.load([n_offset, 0])
        v = v_desc.load([n_offset, 0])
        
        # Native tcgen05 dot(Q, K.T) mapping automatically managed via TMEM registers
        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        qk = qk * sm_scale
        
        # Free out-of-bounds masking when completely divisible cleanly at compile-time
        if not EVEN_N:
            offs_n_curr = n_offset + offs_n
            mask_n = offs_n_curr < N_CTX
            qk = tl.where(mask_n[None, :], qk, float("-inf"))
            
        # Numerical scaling bound protections
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        
        # Reductive accumulation mapping mapped inside FMA Tensor Cores
        acc = acc * alpha[:, None]
        p_bf16 = p.to(tl.bfloat16)
        acc = tl.dot(p_bf16, v, acc, out_dtype=tl.float32)
        
        m_i = m_ij

    # Value corrections
    acc = acc / l_i[:, None]
    lse = m_i + tl.log(l_i)
    
    # Final matrix output vectors stored uniformly utilizing TMA
    o_desc.store([start_m * BLOCK_M, 0], acc.to(tl.bfloat16))
    
    # Resolve traditional element indexing strictly for calculating exact 1D natural log-sum-exp 
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    lse_ptrs = LSE + off_z * stride_lsez + off_h * stride_lseh + offs_m * stride_lsem
    
    if EVEN_M:
        tl.store(lse_ptrs, lse)
    else:
        mask_m = offs_m < N_CTX
        tl.store(lse_ptrs, lse, mask=mask_m)

def run(Q, K, V, O, LSE):
    """
    Computes Non-Causal Multi-Head Attention explicitly allocating results into defined output tensors.
    Utilizes Blackwell's TMA mechanism natively driven directly from device descriptors.
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    if S == 0:
        return
    
    sm_scale = 1.0 / math.sqrt(D)
    
    # Execute as a mapped 1D swizzled vector natively organizing CTAs onto L2 grouping sequences
    grid = lambda META: (
        triton.cdiv(S, META['BLOCK_M']) * B * H,
    )
    
    _attn_fwd_kernel[grid](
        Q, K, V, sm_scale, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        H, S,
        BLOCK_DMODEL=D,
        EVEN_M=False, EVEN_N=False,  
    )