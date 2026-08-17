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
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    # grid = (cdiv(S, BLOCK_M), B * H)
    start_m = tl.program_id(0)
    batch_head = tl.program_id(1)
    
    batch_idx = batch_head // H
    head_idx = batch_head % H

    # LSE is standard pointer-based, compute its base offset
    lse_offset = batch_idx * stride_lsez + head_idx * stride_lseh
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    
    # Load Q using TMA descriptor
    # Descriptor is 3D [B*H, S, D_HEAD], so coordinates are [batch_head, start_m * BLOCK_M, 0]
    q = Q_desc.load([batch_head, start_m * BLOCK_M, 0])
    q = tl.reshape(q, [BLOCK_M, D_HEAD])
    
    # Initialize running softmax statistics and accumulator
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D_HEAD], dtype=tl.float32)
    
    offs_n_base = tl.arange(0, BLOCK_N)
    
    for start_n in range(0, S, BLOCK_N):
        # Load K and V tiles using TMA
        k = K_desc.load([batch_head, start_n, 0])
        v = V_desc.load([batch_head, start_n, 0])
        
        k = tl.reshape(k, [BLOCK_N, D_HEAD])
        v = tl.reshape(v, [BLOCK_N, D_HEAD])
        
        # Compute Q @ K.T
        qk = tl.dot(q, k.T) * sm_scale
        
        # Mask out-of-bounds keys (descriptor zero-pads, which we overwrite to -inf)
        mask_k = (start_n + offs_n_base) < S
        qk = tl.where(mask_k[None, :], qk, float("-inf"))
        
        # Numerically stable softmax steps
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        
        acc = acc * alpha[:, None]
        
        # Compute P @ V
        acc = tl.dot(p.to(tl.bfloat16), v, acc=acc)
        
        m_i = m_ij

    # Finalize probabilities and LSE
    acc = acc / l_i[:, None]
    lse = m_i + tl.log(l_i)
    
    # Store O using TMA descriptor
    acc_3d = tl.reshape(acc, [1, BLOCK_M, D_HEAD])
    O_desc.store([batch_head, start_m * BLOCK_M, 0], acc_3d.to(tl.bfloat16))
    
    # Store LSE using masked pointers
    lse_ptrs = LSE + lse_offset + offs_m * stride_lses
    mask_m = offs_m < S
    tl.store(lse_ptrs, lse, mask=mask_m)


def run(Q, K, V, O, LSE):
    """
    Computes Non-causal Multi-Head Attention leveraging Blackwell TMA.
    Outputs are written directly to the preallocated destination tensors O and LSE.
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    
    # Optimized Blackwell/Hopper tile shapes
    BLOCK_M = 128
    BLOCK_N = 128
    num_warps = 8
    num_stages = 3
    
    # Create 3D host views to match descriptor layout [B*H, S, D]
    Q_3d = Q.view(B * H, S, D)
    K_3d = K.view(B * H, S, D)
    V_3d = V.view(B * H, S, D)
    O_3d = O.view(B * H, S, D)
    
    # Create Host TMA Descriptors dynamically to take advantage of hardware padding logic natively
    Q_desc = TensorDescriptor.from_tensor(Q_3d, [1, BLOCK_M, D])
    K_desc = TensorDescriptor.from_tensor(K_3d, [1, BLOCK_N, D])
    V_desc = TensorDescriptor.from_tensor(V_3d, [1, BLOCK_N, D])
    O_desc = TensorDescriptor.from_tensor(O_3d, [1, BLOCK_M, D])
    
    sm_scale = 1.0 / math.sqrt(D)
    
    # Use 2D grid: unravel start_m as contiguous fastest axis to promote L2 hits for keys and values
    grid = (triton.cdiv(S, BLOCK_M), B * H)
    
    _attn_fwd_kernel[grid](
        Q_desc, K_desc, V_desc, O_desc, LSE,
        sm_scale,
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        B, H, S, D_HEAD=D,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=num_warps, 
        num_stages=num_stages
    )