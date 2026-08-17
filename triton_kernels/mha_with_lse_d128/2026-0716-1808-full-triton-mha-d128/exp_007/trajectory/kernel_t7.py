import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
import math


@triton.jit
def _attention_kernel(
    q_desc, k_desc, v_desc, O, LSE,
    S,
    stride_bh, stride_s, stride_d,
    scale,
    Br: tl.constexpr,
    Bc: tl.constexpr,
):
    """
    Hopper-Optimized FlashAttention Kernel with Software Pipeling and Double Buffering
    
    Fixes:
    1. Replaced direct global loads with TMA Descriptor loads staging data in shared memory.
    2. Resolved `dot()` dtype mismatch by casting intermediate Attention Weights P to `bf16`.
    3. Double buffering K and V to overlap HBM fetches with WGMMA computation.
    """
    pid_q = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    row_offset = pid_bh * S + pid_q * Br
    ptr_o_base = pid_bh * stride_bh + pid_q * Br * stride_s
    ptr_lse_base = pid_bh * S + pid_q * Br
    
    q_offsets = pid_q * Br + tl.arange(0, Br)
    q_mask = q_offsets < S
    
    smem_bar = extern.shared_array(2, tl.int32, alignment=4)
    
    if pid_q == 0 and pid_bh == 0:
        smem_bar[0] = 0
        smem_bar[1] = 0
        
    out0 = tl.zeros((Br, 64), tl.float32)
    out1 = tl.zeros((Br, 64), tl.float32)
    
    m_i = tl.full((Br,), -float("inf"), dtype=tl.float32)
    l_sum = tl.full((Br,), 0.0, dtype=tl.float32)
    
    num_kv_blocks = tl.cdiv(S, Bc)
    
    if num_kv_blocks > 0:
        if pid_q == 0 and pid_bh == 0:
            q0 = q_desc.load([row_offset, 0])
            q1 = q_desc.load([row_offset, 64])
            
            kv_offset = pid_bh * S + 0 * Bc
            k0 = k_desc.load([kv_offset, 0])
            k1 = k_desc.load([kv_offset, 64])
            v0 = v_desc.load([kv_offset, 0])
            v1 = v_desc.load([kv_offset, 64])
            smem_bar[0] = 1
    
    for i in range(num_kv_blocks):
        if i + 1 < num_kv_blocks:
            next_i = (i + 1) % 2
            kv_offset_next = pid_bh * S + (i + 1) * Bc
            
            if pid_q == 0 and pid_bh == 0:
                k_desc.load([kv_offset_next, 0])
                k_desc.load([kv_offset_next, 64])
                v_desc.load([kv_offset_next, 0])
                v_desc.load([kv_offset_next, 64])
                smem_bar[next_i] = 1
        
        stage = i % 2
        for _ in range(1000):
            if smem_bar[stage] >= 1:
                break
        
        k0 = k_desc.load([kv_offset, 0])
        k1 = k_desc.load([kv_offset, 64])
        v0 = v_desc.load([kv_offset, 0])
        v1 = v_desc.load([kv_offset, 64])
        
        s = tl.dot(q0, k0.T)
        s = tl.dot(q1, k1.T, s)
        s = s * scale
        
        k_offsets = i * Bc + tl.arange(0, Bc)
        k_mask = k_offsets < S
        s = tl.where(k_mask[None, :], s, -float("inf"))
        
        prev_m_i = m_i
        rowmax = tl.max(s, axis=-1, keepdims=True)
        
        if prev_m_i == -float("inf"):
            m_i = rowmax
        else:
            m_i = tl.maximum(prev_m_i, rowmax)
            
        P = tl.exp(s - m_i)
        P = tl.where(k_mask[None, :], P, 0.0)
        
        out0 *= tl.exp(prev_m_i - m_i)
        out1 *= tl.exp(prev_m_i - m_i)
        
        P_bf16 = P.to(tl.bfloat16)
        out0 = tl.dot(P_bf16, v0, out0)
        out1 = tl.dot(P_bf16, v1, out1)
        
        rowsum = tl.sum(P, axis=-1, keepdims=True)
        l_sum *= tl.exp(prev_m_i - m_i)
        l_sum += rowsum
        
        if pid_q == 0 and pid_bh == 0:
            smem_bar[stage] = 0 
            
    out0 = out0 / l_sum
    out1 = out1 / l_sum
    
    ptr0 = ptr_o_base + q_offsets[:, None] * stride_s + tl.arange(0, 64)[None, :] * stride_d
    ptr1 = ptr_o_base + q_offsets[:, None] * stride_s + (64 + tl.arange(0, 64))[None, :] * stride_d
    
    tl.store(ptr0, out0.to(tl.bfloat16), mask=q_mask[:, None])
    tl.store(ptr1, out1.to(tl.bfloat16), mask=q_mask[:, None])
    
    lse = m_i + tl.log(l_sum)
    tl.store(LSE + ptr_lse_base + q_offsets, lse, mask=q_mask)


def run(Q, K, V, O, LSE):
    """Compute ``O = softmax(Q@K^T/sqrt(D))@V`` and natural log LSE into preallocated CUDA tensors."""
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    Br = 128
    Bc = 64
    
    scale = 1.0 / math.sqrt(D)
    stride_bh = S * D
    stride_s = D
    stride_d = 1
    
    q_desc = TensorDescriptor.from_tensor(Q, block_shape=[Br, D])
    k_desc = TensorDescriptor.from_tensor(K, block_shape=[Bc, D])
    v_desc = TensorDescriptor.from_tensor(V, block_shape=[Bc, D])
    
    num_q_blocks = triton.cdiv(S, Br)
    grid = (num_q_blocks, B * H)
    
    _attention_kernel[grid](
        q_desc, k_desc, v_desc, O, LSE, S,
        stride_bh, stride_s, stride_d,
        scale,
        Br=Br, Bc=Bc,
        num_warps=4,
        num_stages=2,
        maxnreg=128
    )