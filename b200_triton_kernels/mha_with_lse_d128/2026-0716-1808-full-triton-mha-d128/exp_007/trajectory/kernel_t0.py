import torch
import triton
import triton.language as tl
import math


@triton.jit
def _attention_kernel(
    Q, K, V, O, LSE,
    S,
    stride_bh_q, stride_s_q, stride_d_q,
    stride_bh_k, stride_s_k, stride_d_k,
    stride_bh_v, stride_s_v, stride_d_v,
    stride_bh_o, stride_s_o, stride_d_o,
    stride_bh_lse, stride_s_lse,
    scale,
    Br: tl.constexpr,
    Bc: tl.constexpr,
):
    """
    Standard FlashAttention-Style Kernel (Non-Causal)
    
    Computes the attention output O and Log-Sum-Exp (LSE) for a specific 
    query block and batch/head combination.
    """
    # 1. Determine location in the GEMMs
    pid_q = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    q_offsets = pid_q * Br + tl.arange(0, Br)
    q_mask = q_offsets < S
    
    # 2. Setup pointers to relevant rows of Q, K, V, O, LSE
    q_base = Q + pid_bh * stride_bh_q
    k_base = K + pid_bh * stride_bh_k
    v_base = V + pid_bh * stride_bh_v
    o_base = O + pid_bh * stride_bh_o
    lse_base = LSE + pid_bh * stride_bh_lse
    
    # 3. Load Q blocks (D=128 is divided into two independent halves of 64)
    q0 = tl.load(q_base + q_offsets[:, None] * stride_s_q + tl.arange(0, 64)[None, :] * stride_d_q,
                 mask=q_mask[:, None], other=0.0)
    q1 = tl.load(q_base + q_offsets[:, None] * stride_s_q + (64 + tl.arange(0, 64))[None, :] * stride_d_q,
                 mask=q_mask[:, None], other=0.0)
    
    # 4. Initialize online-softmax state accumulators
    out0 = tl.zeros((Br, 64), tl.float32)
    out1 = tl.zeros((Br, 64), tl.float32)
    m = tl.full((Br,), -1e38, tl.float32)
    l = tl.full((Br,), 0.0, tl.float32)
    
    num_kv_blocks = tl.cdiv(S, Bc)
    
    # 5. Main KV Iteration Loop
    for i in range(num_kv_blocks):
        k_offsets = i * Bc + tl.arange(0, Bc)
        k_mask = k_offsets < S
        
        # Load chunked K and V slices
        k0 = tl.load(k_base + k_offsets[:, None] * stride_s_k + tl.arange(0, 64)[None, :] * stride_d_k, mask=k_mask[:, None], other=0.0)
        k1 = tl.load(k_base + k_offsets[:, None] * stride_s_k + (64 + tl.arange(0, 64))[None, :] * stride_d_k, mask=k_mask[:, None], other=0.0)
        v0 = tl.load(v_base + k_offsets[:, None] * stride_s_v + tl.arange(0, 64)[None, :] * stride_d_v, mask=k_mask[:, None], other=0.0)
        v1 = tl.load(v_base + k_offsets[:, None] * stride_s_v + (64 + tl.arange(0, 64))[None, :] * stride_d_v, mask=k_mask[:, None], other=0.0)
        
        # QK^T Calculation utilizing accumulated dot products over dim 64
        s = tl.dot(q0, k0.T)
        s = tl.dot(q1, k1.T, s)
        s *= scale # Apply scaling factor 1/sqrt(D)
        
        # Mask out-of-bounds sequence items effectively sending them to -inf
        s = tl.where(k_mask[None, :], s, -1e38)
        
        # Standard FlashAttention updates using previous iteration max/sums
        m_prev = m
        m = tl.maximum(m, tl.reduce(s, axis=1, op=tl.max))
        p = tl.exp(s - m)
        p = tl.where(k_mask[None, :], p, 0.0)
        
        out0 *= tl.exp(m_prev - m)[:, None]
        out1 *= tl.exp(m_prev - m)[:, None]
        
        out0 = tl.dot(p, v0, out0)
        out1 = tl.dot(p, v1, out1)
        
        l = l * tl.exp(m_prev - m) + tl.reduce(p, axis=1, op=tl.sum)

    # 6. Final Output Normalization and Storage
    
    # Divide outputs by total iteration sequence sum to yield expected attention outputs
    out0 /= l[:, None]
    out1 /= l[:, None]
    
    # Utilize bit-level casting to convert accumulator `float32` results to tensor `bfloat16` 
    stored0 = (out0.to_bits() >> 16).to(tl.bfloat16, bitcast=True)
    stored1 = (out1.to_bits() >> 16).to(tl.bfloat16, bitcast=True)
    
    ptr0 = o_base + q_offsets[:, None] * stride_s_o + tl.arange(0, 64)[None, :] * stride_d_o
    ptr1 = o_base + q_offsets[:, None] * stride_s_o + (64 + tl.arange(0, 64))[None, :] * stride_d_o
    
    tl.store(ptr0, stored0, mask=q_mask[:, None])
    tl.store(ptr1, stored1, mask=q_mask[:, None])
    
    lse = m + tl.log(l)
    tl.store(lse_base + q_offsets, lse, mask=q_mask)


def run(Q, K, V, O, LSE):
    """Compute ``O = softmax(Q@K^T/sqrt(D))@V`` and natural log LSE into preallocated CUDA tensors."""
    torch.cuda.set_device(Q.device)
    B, H, S, D_value = Q.shape
    
    # Hardcoded block heuristics leveraging power-of-two limits 
    # (Br=128, Bc=128) optimally mapped across 4 independent warp groups.
    Br = 128
    Bc = 128
    
    scale = 1.0 / math.sqrt(D_value)
    stride_bh = S * D_value
    stride_s = D_value
    stride_d = 1
    
    num_q_blocks = triton.cdiv(S, Br)
    grid = (num_q_blocks, B * H)
    
    _attention_kernel[grid](
        Q, K, V, O, LSE, S,
        stride_bh, stride_s, stride_d,
        stride_bh, stride_s, stride_d,
        stride_bh, stride_s, stride_d,
        stride_bh, stride_s, stride_d,
        S, 1,
        scale,
        Br=Br, Bc=Bc,
        num_warps=4,
        num_stages=2
    )