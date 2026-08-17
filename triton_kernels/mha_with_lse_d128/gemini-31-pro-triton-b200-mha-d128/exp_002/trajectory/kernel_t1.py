import torch
import triton
import triton.language as tl
import math

def pre_hook(kwargs):
    from triton.tools.tensor_descriptor import TensorDescriptor
    BLOCK_M = kwargs["BLOCK_M"]
    BLOCK_N = kwargs["BLOCK_N"]
    D = kwargs["D"]
    N_CTX = kwargs["N_CTX"]
    
    # Compute if the sequence length perfectly divides the block sizes
    kwargs["EVEN_M"] = (N_CTX % BLOCK_M == 0)
    kwargs["EVEN_N"] = (N_CTX % BLOCK_N == 0)
    
    # Only wrap if it's a tensor (has data_ptr), which handles repeated calls by autotuner safely
    if hasattr(kwargs["Q_desc"], 'data_ptr'):
        # Create TMA 4D host tensor descriptors mapped perfectly to PyTorch multi-dim layouts
        kwargs["Q_desc"] = TensorDescriptor.from_tensor(kwargs["Q_desc"], [1, 1, BLOCK_M, D])
        kwargs["K_desc"] = TensorDescriptor.from_tensor(kwargs["K_desc"], [1, 1, BLOCK_N, D])
        kwargs["V_desc"] = TensorDescriptor.from_tensor(kwargs["V_desc"], [1, 1, BLOCK_N, D])
        kwargs["O_desc"] = TensorDescriptor.from_tensor(kwargs["O_desc"], [1, 1, BLOCK_M, D])

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'WARP_SPECIALIZE': False, 'NUM_STAGES': 3}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'WARP_SPECIALIZE': True, 'NUM_STAGES': 3}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'WARP_SPECIALIZE': False, 'NUM_STAGES': 3}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'WARP_SPECIALIZE': True, 'NUM_STAGES': 3}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'WARP_SPECIALIZE': False, 'NUM_STAGES': 4}, num_stages=4, num_warps=4),
    ],
    key=['N_CTX'],
    pre_hook=pre_hook,
)
@triton.jit
def _attn_fwd_kernel(
    Q_desc, K_desc, V_desc, sm_scale,
    O_desc, LSE,
    stride_lsez, stride_lseh, stride_lsem,
    H, N_CTX, D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    EVEN_M: tl.constexpr, EVEN_N: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    # Extract batch (Z) and head (H) coordinates
    off_z = off_hz // H
    off_h = off_hz % H
    
    # Load Q tile via TMA
    q = Q_desc.load([off_z, off_h, start_m * BLOCK_M, 0])
    q = tl.reshape(q, [BLOCK_M, D])
    
    # Initialize running FlashAttention variables 
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < N_CTX
    
    # Process K and V blocks over the sequence length, with pipelining & optional warp specialization
    for start_n_idx in tl.range(0, tl.cdiv(N_CTX, BLOCK_N), num_stages=NUM_STAGES, warp_specialize=WARP_SPECIALIZE):
        n_offset = start_n_idx * BLOCK_N
        
        # Load K and V tiles via TMA
        k = K_desc.load([off_z, off_h, n_offset, 0])
        v = V_desc.load([off_z, off_h, n_offset, 0])
        
        k = tl.reshape(k, [BLOCK_N, D])
        v = tl.reshape(v, [BLOCK_N, D])
        
        # Compute QK^T scaled dot product
        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        qk = qk * sm_scale
        
        # Branch-free masking for keys if sequence length leaves partial tile
        if not EVEN_N:
            offs_n = n_offset + tl.arange(0, BLOCK_N)
            mask_n = offs_n < N_CTX
            qk = tl.where(mask_n[None, :], qk, float("-inf"))
            
        # Branch-free masking for queries if sequence length leaves partial tile
        if not EVEN_M:
            qk = tl.where(mask_m[:, None], qk, float("-inf"))
            
        # Update running max and scaling factor (FlashAttention formulation)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        
        # Update accumulator
        acc = acc * alpha[:, None]
        p = p.to(tl.bfloat16)
        acc = tl.dot(p, v, acc, out_dtype=tl.float32)
        
        m_i = m_ij
        
    # Finalize normalization for valid rows
    acc = acc / l_i[:, None]
    
    # Compute natural log-sum-exp
    lse = m_i + tl.log(l_i)
    
    # Store O tile via TMA - TMA natively ignores writes outside bounding box
    acc_4d = tl.reshape(acc.to(tl.bfloat16), [1, 1, BLOCK_M, D])
    O_desc.store([off_z, off_h, start_m * BLOCK_M, 0], acc_4d)
    
    # Store LSE output
    lse_ptrs = LSE + off_z * stride_lsez + off_h * stride_lseh + offs_m * stride_lsem
    tl.store(lse_ptrs, lse, mask=mask_m)

def run(Q, K, V, O, LSE):
    """
    Computes Non-Causal Multi-Head Attention forward.
    Inputs and O shape: (B, H, S, D) in bfloat16.
    LSE shape: (B, H, S) in float32.
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    if S == 0:
        return
    
    # 1/sqrt(D) scaling factor
    sm_scale = 1.0 / math.sqrt(D)
    
    # Configure grid layout over seq length and (Batch * Head) space
    grid = lambda META: (
        triton.cdiv(S, META['BLOCK_M']),
        B * H,
    )
    
    # Kernel Launch. Note: EVEN_M and EVEN_N are supplied with dummy values here 
    # and correctly overridden by the autotune pre_hook using precise shapes.
    _attn_fwd_kernel[grid](
        Q_desc=Q, K_desc=K, V_desc=V, sm_scale=sm_scale,
        O_desc=O, LSE=LSE,
        stride_lsez=LSE.stride(0), stride_lseh=LSE.stride(1), stride_lsem=LSE.stride(2),
        H=H, N_CTX=S, D=D,
        EVEN_M=False, EVEN_N=False,
    )