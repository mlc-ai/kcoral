import torch
import triton
import triton.language as tl
import math

# Configure Triton's descriptor allocator to support device-side `tl.make_tensor_descriptor`
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

def pre_hook(kwargs):
    # Determines if the sequence dimension is perfectly divisible by the block size,
    # enabling a compile-time branch to remove bounds-checking during execution.
    kwargs['EVEN_N'] = (kwargs['N_CTX'] % kwargs['BLOCK_N'] == 0)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'WARP_SPECIALIZE': True}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'WARP_SPECIALIZE': True}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'WARP_SPECIALIZE': True}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'WARP_SPECIALIZE': True}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'WARP_SPECIALIZE': False}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'WARP_SPECIALIZE': False}, num_stages=3, num_warps=4),
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
    WARP_SPECIALIZE: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    # Calculate batch and head mappings
    off_z = off_hz // H
    off_h = off_hz % H
    
    # Compute base offsets for the current batch & head
    q_ptr = Q + off_z * stride_qz + off_h * stride_qh
    k_ptr = K + off_z * stride_kz + off_h * stride_kh
    v_ptr = V + off_z * stride_vz + off_h * stride_vh
    o_ptr = O + off_z * stride_oz + off_h * stride_oh
    
    # Create TMA device descriptors. The shapes perfectly enclose a single Head's inner context.
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
    
    # Pre-load queries segment via TMA
    q = q_desc.load([start_m * BLOCK_M, 0])
    
    # Initialize mathematically stable FlashAttention accumulators
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    
    # TMA hardware pipelined loop (potentially utilizing Blackwell Automatic Warp Specialization)
    for start_n_idx in tl.range(0, tl.cdiv(N_CTX, BLOCK_N), warp_specialize=WARP_SPECIALIZE):
        n_offset = start_n_idx * BLOCK_N
        
        # Asynchronously load Key and Value blocks mapped via the hardware
        k = k_desc.load([n_offset, 0])
        v = v_desc.load([n_offset, 0])
        
        # Calculate scores dot(Q, K.T)
        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        qk = qk * sm_scale
        
        # Out-of-bounds keys are masked correctly to -inf to ensure their attention contribution remains strictly 0. 
        # By enforcing `EVEN_N`, this branch resolves completely uniformly in compilation if safe!
        if not EVEN_N:
            offs_n = n_offset + tl.arange(0, BLOCK_N)
            mask_n = offs_n < N_CTX
            qk = tl.where(mask_n[None, :], qk, float("-inf"))
        
        # Increment max running bounds
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        
        # Softmax scaling state corrections
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        
        # Output summation and normalization weighting dot(P, V)
        acc = acc * alpha[:, None]
        p_bf16 = p.to(tl.bfloat16)
        acc = tl.dot(p_bf16, v, acc, out_dtype=tl.float32)
        
        m_i = m_ij

    # Resolve finalized softmax values 
    acc = acc / l_i[:, None]
    lse = m_i + tl.log(l_i)
    
    # Standardize result vectors efficiently; Out-of-bounds M stores are fully safely ignored by TMA natively
    o_desc.store([start_m * BLOCK_M, 0], acc.to(tl.bfloat16))
    
    # Resolve Natural Log-Sum-Exp mapping bounds
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    lse_ptrs = LSE + off_z * stride_lsez + off_h * stride_lseh + offs_m * stride_lsem
    mask_m = offs_m < N_CTX
    tl.store(lse_ptrs, lse, mask=mask_m)


def run(Q, K, V, O, LSE):
    """
    Computes Non-Causal Multi-Head Attention forward pass returning Outputs (O) and natural LogSumExp (LSE).
    Allocates output results dynamically natively mapped inside explicitly constrained PyTorch tensors.
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    if S == 0:
        return
    
    sm_scale = 1.0 / math.sqrt(D)
    
    # Maps directly matching blocks over Heads/Batches linearly (L2 optimizing)
    grid = lambda META: (
        triton.cdiv(S, META['BLOCK_M']),
        B * H,
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
        EVEN_N=False,  # Replaced actively by triton pre_hook mapped exactly to dimensions mappings
    )