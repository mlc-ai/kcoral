import torch
import triton
import triton.language as tl
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
    Optimized FlashAttention Kernel for Hopper GPUs.
    
    Iterates sequentially over chunks of the Key/Value sequence to compute 
    the attention outputs and the log-sum-exp.
    """
    pid_q = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    q_base = Q + pid_bh * stride_bh
    k_base = K + pid_bh * stride_bh
    v_base = V + pid_bh * stride_bh
    o_base = O + pid_bh * stride_bh
    
    q_offsets = pid_q * Br + tl.arange(0, Br)
    q_mask = q_offsets < S
    
    d_offsets0 = tl.arange(0, 64)
    d_offsets1 = 64 + tl.arange(0, 64)
    
    # Load the Q block utilizing two independent GEMM operations along the contiguous dimension
    q0 = tl.load(q_base + q_offsets[:, None] * stride_s + d_offsets0[None, :] * stride_d,
                 mask=q_mask[:, None], other=0.0)
    q1 = tl.load(q_base + q_offsets[:, None] * stride_s + d_offsets1[None, :] * stride_d,
                 mask=q_mask[:, None], other=0.0)
    
    acc0 = tl.zeros((Br, 64), tl.float32)
    acc1 = tl.zeros((Br, 64), tl.float32)
    
    m = tl.full((Br,), -1e38, tl.float32)
    l = tl.full((Br,), 0.0, tl.float32)
    
    num_kv_blocks = tl.cdiv(S, Bc)
    
    for i in range(num_kv_blocks):
        k_offsets = i * Bc + tl.arange(0, Bc)
        k_mask = k_offsets < S
        
        # Gather chunked slices of K and V mapped identically to Q's physical layout
        k0 = tl.load(k_base + k_offsets[:, None] * stride_s + d_offsets0[None, :] * stride_d, mask=k_mask[:, None], other=0.0)
        k1 = tl.load(k_base + k_offsets[:, None] * stride_s + d_offsets1[None, :] * stride_d, mask=k_mask[:, None], other=0.0)
        v0 = tl.load(v_base + k_offsets[:, None] * stride_s + d_offsets0[None, :] * stride_d, mask=k_mask[:, None], other=0.0)
        v1 = tl.load(v_base + k_offsets[:, None] * stride_s + d_offsets1[None, :] * stride_d, mask=k_mask[:, None], other=0.0)
        
        s = tl.dot(q0, k0.T)
        s = tl.dot(q1, k1.T, s)
        
        s *= scale
        
        # Mask out-of-bounds items effectively sending them to negative infinity before the exponential
        s = tl.where(k_mask[None, :], s, -1e38)
        
        m_prev = m
        m = tl.maximum(m, tl.max(s, axis=1))
        P = tl.exp(s - m[:, None])
        
        # Zero out masked values post-exponential
        P = tl.where(k_mask[None, :], P, 0.0)
        
        # Scale previous accumulation to account for updated numerical limits
        acc0 *= tl.exp(m_prev - m)[:, None]
        acc1 *= tl.exp(m_prev - m)[:, None]
        
        acc0 = tl.dot(P, v0, acc0)
        acc1 = tl.dot(P, v1, acc1)
        
        l = l * tl.exp(m_prev - m) + tl.sum(P, axis=1)

    # Final Output Normalization scaling
    acc_scale = 1.0 / l[:, None]
    acc0 *= acc_scale
    acc1 *= acc_scale
    
    ptr0 = o_base + q_offsets[:, None] * stride_s + d_offsets0[None, :] * stride_d
    ptr1 = o_base + q_offsets[:, None] * stride_s + d_offsets1[None, :] * stride_d
    
    stored0 = acc0.to(tl.bfloat16)
    stored1 = acc1.to(tl.bfloat16)
    
    tl.store(ptr0, stored0, mask=q_mask[:, None])
    tl.store(ptr1, stored1, mask=q_mask[:, None])
    
    lse = m + tl.log(l)
    tl.store(LSE + pid_bh * S + q_offsets, lse, mask=q_mask)


def run(Q, K, V, O, LSE):
    """Compute ``O = softmax(Q@K^T/sqrt(D))@V`` and natural log LSE into preallocated CUDA tensors."""
    torch.cuda.set_device(Q.device)
    B, H, S, D_value = Q.shape
    
    Br = 128
    Bc = 64
    
    scale = 1.0 / math.sqrt(D_value)
    stride_bh = S * D_value
    stride_s = D_value
    stride_d = 1
    
    num_q_blocks = triton.cdiv(S, Br)
    grid = (num_q_blocks, B * H)
    
    _attention_kernel[grid](
        Q, K, V, O, LSE, S,
        stride_bh, stride_s, stride_d,
        scale,
        Br=Br, Bc=Bc,
        num_warps=4,
        num_stages=2
    )