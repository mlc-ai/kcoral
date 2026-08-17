import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
import math

def pre_hook(kwargs):
    BLOCK_M = kwargs["BLOCK_M"]
    BLOCK_N = kwargs["BLOCK_N"]
    
    # Safely overwrite the tensor arguments with TMA descriptors configured for the current config block shapes.
    # The Triton autotuner provides a fresh copy of the initial kwargs dictionary for each trial, 
    # so we can mutate these keys directly without state bleeding between configurations.
    kwargs["Q_desc"] = TensorDescriptor.from_tensor(kwargs["Q_desc"], [1, 1, BLOCK_M, kwargs["D"]])
    kwargs["K_desc"] = TensorDescriptor.from_tensor(kwargs["K_desc"], [1, 1, BLOCK_N, kwargs["D"]])
    kwargs["V_desc"] = TensorDescriptor.from_tensor(kwargs["V_desc"], [1, 1, BLOCK_N, kwargs["D"]])
    kwargs["O_desc"] = TensorDescriptor.from_tensor(kwargs["O_desc"], [1, 1, BLOCK_M, kwargs["D"]])
    
    kwargs["EVEN_M"] = (kwargs["N_CTX"] % BLOCK_M == 0)
    kwargs["EVEN_N"] = (kwargs["N_CTX"] % BLOCK_N == 0)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=4, num_warps=4),
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
    NUM_STAGES: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    # Extract Batch (z) and Head (h) coordinates
    off_z = off_hz // H
    off_h = off_hz % H
    
    # Load queries chunk natively via TMA
    q = Q_desc.load([off_z, off_h, start_m * BLOCK_M, 0])
    q = tl.reshape(q, [BLOCK_M, D])
    
    # Initialize FlashAttention V2 accumulators 
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    
    offs_n = tl.arange(0, BLOCK_N)
    
    # Pipelined loop over sequence context
    for start_n_idx in tl.range(0, tl.cdiv(N_CTX, BLOCK_N), num_stages=NUM_STAGES):
        n_offset = start_n_idx * BLOCK_N
        
        # Hardware pipelined TMA loads for K and V
        k = K_desc.load([off_z, off_h, n_offset, 0])
        v = V_desc.load([off_z, off_h, n_offset, 0])
        
        k = tl.reshape(k, [BLOCK_N, D])
        v = tl.reshape(v, [BLOCK_N, D])
        
        # Native Dot Product: Blackwell optimizes dot(Q, K.T) efficiently using TMEM accumulators
        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        qk = qk * sm_scale
        
        # Branch-free out-of-bounds key masking isolated to sequence edges 
        if not EVEN_N:
            offs_n_curr = n_offset + offs_n
            mask_n = offs_n_curr < N_CTX
            qk = tl.where(mask_n[None, :], qk, float("-inf"))
            
        # Running scaling & numerically-stable Softmax
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        
        acc = acc * alpha[:, None]
        
        # Bfloat16 reduction accumulation mapping
        p_bf16 = p.to(tl.bfloat16)
        acc = tl.dot(p_bf16, v, acc, out_dtype=tl.float32)
        
        m_i = m_ij
        
    # Scale finalized outputs
    acc = acc / l_i[:, None]
    lse = m_i + tl.log(l_i)
    
    # Store matrix multiplications output via TMA. Hardware will automatically clamp stores correctly!
    acc_4d = tl.reshape(acc.to(tl.bfloat16), [1, 1, BLOCK_M, D])
    O_desc.store([off_z, off_h, start_m * BLOCK_M, 0], acc_4d)
    
    # Calculate regular pointer arithmetic to safely store natural log-sum-exp
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    lse_ptrs = LSE + off_z * stride_lsez + off_h * stride_lseh + offs_m * stride_lsem
    
    if EVEN_M:
        tl.store(lse_ptrs, lse)
    else:
        tl.store(lse_ptrs, lse, mask=offs_m < N_CTX)

def run(Q, K, V, O, LSE):
    """
    Computes Non-Causal Multi-Head Attention forward pass over preallocated Tensors.
    Inputs/O shapes: (B, H, S, D) in bfloat16.
    LSE shape: (B, H, S) in float32.
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    if S == 0:
        return
    
    sm_scale = 1.0 / math.sqrt(D)
    
    grid = lambda META: (
        triton.cdiv(S, META['BLOCK_M']),
        B * H,
    )
    
    # Launch kernels passing PyTorch tensors into desc keys, which are overwritten intelligently inside pre_hook
    _attn_fwd_kernel[grid](
        Q_desc=Q, K_desc=K, V_desc=V, sm_scale=sm_scale,
        O_desc=O, LSE=LSE,
        stride_lsez=LSE.stride(0), stride_lseh=LSE.stride(1), stride_lsem=LSE.stride(2),
        H=H, N_CTX=S, D=D,
        EVEN_M=False, EVEN_N=False,  
    )