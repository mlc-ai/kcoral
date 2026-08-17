import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
import math

def pre_hook(kwargs):
    BLOCK_M = kwargs["BLOCK_M"]
    BLOCK_N = kwargs["BLOCK_N"]
    D = kwargs["D"]
    
    B_times_H = kwargs["Q_ptr"].shape[0] * kwargs["Q_ptr"].shape[1]
    S = kwargs["N_CTX"]
    
    # Reshape to 3D on the host to map gracefully to a 3-dimensional TMA TensorDescriptor.
    # This avoids the rank limits and dimension edge cases of 4D descriptors.
    Q_3d = kwargs["Q_ptr"].view(B_times_H, S, D)
    K_3d = kwargs["K_ptr"].view(B_times_H, S, D)
    V_3d = kwargs["V_ptr"].view(B_times_H, S, D)
    O_3d = kwargs["O_ptr"].view(B_times_H, S, D)
    
    kwargs["Q_desc"] = TensorDescriptor.from_tensor(Q_3d, [1, BLOCK_M, D])
    kwargs["K_desc"] = TensorDescriptor.from_tensor(K_3d, [1, BLOCK_N, D])
    kwargs["V_desc"] = TensorDescriptor.from_tensor(V_3d, [1, BLOCK_N, D])
    kwargs["O_desc"] = TensorDescriptor.from_tensor(O_3d, [1, BLOCK_M, D])

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'NUM_STAGES': 3, 'WARP_SPECIALIZE': True}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'NUM_STAGES': 4, 'WARP_SPECIALIZE': True}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'NUM_STAGES': 3, 'WARP_SPECIALIZE': True}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'NUM_STAGES': 4, 'WARP_SPECIALIZE': True}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'NUM_STAGES': 3, 'WARP_SPECIALIZE': False}, num_stages=3, num_warps=8),
    ],
    key=['N_CTX'],
    pre_hook=pre_hook,
)
@triton.jit
def _attn_fwd_kernel(
    Q_desc, K_desc, V_desc, sm_scale, O_desc, LSE,
    stride_lsez, stride_lseh, stride_lsem,
    H, N_CTX, D: tl.constexpr,
    Q_ptr, K_ptr, V_ptr, O_ptr, 
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    NUM_STAGES: tl.constexpr, WARP_SPECIALIZE: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    off_z = off_hz // H
    off_h = off_hz % H
    
    # Load query tile from 3D TMA descriptor
    q = Q_desc.load([off_hz, start_m * BLOCK_M, 0])
    q = tl.reshape(q, [BLOCK_M, D])
    
    # Initialize FlashAttention variables
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    
    offs_n = tl.arange(0, BLOCK_N)
    
    # Main pipelined dot loop. Uses warp_specialize purely branch-free to comply with the transformation.
    for start_n_idx in tl.range(0, tl.cdiv(N_CTX, BLOCK_N), num_stages=NUM_STAGES, warp_specialize=WARP_SPECIALIZE):
        n_offset = start_n_idx * BLOCK_N
        
        # TMA Hardware Loads (zero-pads out-of-bounds implicitly)
        k = K_desc.load([off_hz, n_offset, 0])
        v = V_desc.load([off_hz, n_offset, 0])
        
        k = tl.reshape(k, [BLOCK_N, D])
        v = tl.reshape(v, [BLOCK_N, D])
        
        # Blackwell Native tcgen05 dot(Q, K.T)
        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        qk = qk * sm_scale
        
        # Mask out-of-bound keys perfectly with element-wise ALU operations natively
        offs_n_curr = n_offset + offs_n
        qk = tl.where(offs_n_curr[None, :] < N_CTX, qk, float("-inf"))
            
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        
        acc = acc * alpha[:, None]
        
        # Bfloat16 cast for TMEM value accumulation dot(P, V)
        p_bf16 = p.to(tl.bfloat16)
        acc = tl.dot(p_bf16, v, acc, out_dtype=tl.float32)
        
        m_i = m_ij
        
    # Scale finalized outputs correctly
    acc = acc / l_i[:, None]
    lse = m_i + tl.log(l_i)
    
    # Store matrix multiplications output via TMA 3D descriptor. (Hardware naturally drops out-of-bounds rows.)
    acc_3d = tl.reshape(acc.to(tl.bfloat16), [1, BLOCK_M, D])
    O_desc.store([off_hz, start_m * BLOCK_M, 0], acc_3d)
    
    # Save natural log-sum-exp via traditional pointer indexing
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    lse_ptrs = LSE + off_z * stride_lsez + off_h * stride_lseh + offs_m * stride_lsem
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
    
    # Pass baseline valid descriptors which satisfy Triton's type-checker structurally. 
    # The pre_hook runs immediately after and substitutes exactly matching descriptors for the benchmarked autotune blocks sizes.
    Q_3d = Q.view(B * H, S, D)
    K_3d = K.view(B * H, S, D)
    V_3d = V.view(B * H, S, D)
    O_3d = O.view(B * H, S, D)
    
    Q_desc = TensorDescriptor.from_tensor(Q_3d, [1, 128, D])
    K_desc = TensorDescriptor.from_tensor(K_3d, [1, 128, D])
    V_desc = TensorDescriptor.from_tensor(V_3d, [1, 128, D])
    O_desc = TensorDescriptor.from_tensor(O_3d, [1, 128, D])
    
    grid = lambda META: (
        triton.cdiv(S, META['BLOCK_M']),
        B * H,
    )
    
    _attn_fwd_kernel[grid](
        Q_desc=Q_desc, K_desc=K_desc, V_desc=V_desc, sm_scale=sm_scale, O_desc=O_desc, LSE=LSE,
        stride_lsez=LSE.stride(0), stride_lseh=LSE.stride(1), stride_lsem=LSE.stride(2),
        H=H, N_CTX=S, D=D,
        Q_ptr=Q, K_ptr=K, V_ptr=V, O_ptr=O,
    )