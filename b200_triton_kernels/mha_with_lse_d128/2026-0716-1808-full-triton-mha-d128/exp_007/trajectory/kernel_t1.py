import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
import math


@triton.jit
def _attention_kernel(
    Q, K, V, O, LSE,
    S,
    stride_bh, stride_s, stride_d,
    scale,
    Br: tl.constexpr,
    Bc: tl.constexpr,
):
    """
    Hopper-Optimized FlashAttention Kernel
    
    Utilizes Tensor Descriptors for efficient TMA loads of a unified D=128 block. 
    Accumulates natively in FP32 utilizing WGMMA-style layouts.
    """
    pid_q = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    # Load Q block using optimized TMA descriptor (Layout [Br, 128])
    Q = tl.load(Q, [pid_bh * S + pid_q * Br, 0])
    
    # Accumulators
    acc = tl.zeros((Br, 128), tl.float32)
    m = tl.full((Br,), -1e38, tl.float32)
    l = tl.full((Br,), 0.0, tl.float32)
    
    q_offsets = pid_q * Br + tl.arange(0, Br)
    q_mask = q_offsets < S
    
    num_kv_blocks = tl.cdiv(S, Bc)
    
    for i in range(num_kv_blocks):
        # Load chunked K and V slices dynamically mapped over D=128 
        K = tl.load(K, [pid_bh * S + i * Bc, 0])
        V = tl.load(V, [pid_bh * S + i * Bc, 0])
        
        # QK^T Calculation utilizing accumulated dot products over dim 128
        s = tl.dot(Q, K.T)
        s *= scale # Apply scaling factor 1/sqrt(D)
        
        m_prev = m
        m = tl.maximum(m, tl.max(s, axis=1))
        P = tl.exp(s - m[:, None])
        
        k_offsets = i * Bc + tl.arange(0, Bc)
        k_mask = k_offsets < S
        P = tl.where(k_mask[None, :], P, 0.0)
        
        acc *= tl.exp(m_prev - m)[:, None]
        
        # P @ V
        acc = tl.dot(P, V, acc)
        
        l = l * tl.exp(m_prev - m) + tl.sum(P, axis=1)

    # Final Output Normalization
    acc /= l[:, None]
    
    ptr = O + pid_bh * stride_bh + q_offsets[:, None] * stride_s + tl.arange(0, 128)[None, :] * stride_d
    # Standard FP32 -> BF16 conversion
    tl.store(ptr, acc.to(tl.bfloat16), mask=q_mask[:, None])
    
    lse = m + tl.log(l)
    tl.store(LSE + pid_bh * S + q_offsets, lse, mask=q_mask)


def run(Q, K, V, O, LSE):
    """Compute ``O = softmax(Q@K^T/sqrt(D))@V`` and natural log LSE into preallocated CUDA tensors."""
    torch.cuda.set_device(Q.device)
    B, H, S, D_value = Q.shape
    
    # Fixed block sizes leveraging powers of 2 
    Br = 128
    Bc = 128
    
    scale = 1.0 / math.sqrt(D_value)
    stride_bh = S * D_value
    stride_s = D_value
    stride_d = 1
    
    # Tensor descriptors mapping native contiguous PyTorch layout
    q_desc = TensorDescriptor.from_tensor(Q, [Br, D_value])
    k_desc = TensorDescriptor.from_tensor(K, [Bc, D_value])
    v_desc = TensorDescriptor.from_tensor(V, [Bc, D_value])
    
    num_q_blocks = triton.cdiv(S, Br)
    grid = (num_q_blocks, B * H)
    
    _attention_kernel[grid](
        q_desc, k_desc, v_desc, O, LSE, S,
        stride_bh, stride_s, stride_d,
        scale,
        Br=Br, Bc=Bc,
        num_warps=8,
        num_stages=4
    )