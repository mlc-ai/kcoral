import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _attention_kernel(
    q_desc,
    k_desc,
    v_desc,
    o_desc,
    lse_ptr,
    S_seq,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Optimized FlashAttention kernel for bf16 inputs targeting Hopper SM90.
    
    Computes Output O and Log-Sum-Exp (LSE) for non-causal multi-head attention.
    Uses a single-pass approach (QK^T and PV) pipelined through TMA descriptors.
    """
    row_start = tl.program_id(0) * BLOCK_M
    bh = tl.program_id(1)
    
    m = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    l = tl.zeros((BLOCK_M,), dtype=tl.float32)
    out0 = tl.zeros((BLOCK_M, 64), dtype=tl.float32)
    out1 = tl.zeros((BLOCK_M, 64), dtype=tl.float32)
    
    scale = 1.0 / tl.sqrt(D)
    
    num_kv_blocks = S_seq // BLOCK_N
    
    row_idx = tl.arange(0, BLOCK_M)
    valid_rows = row_start + row_idx < S_seq
    
    col_idx = tl.arange(0, BLOCK_N)
    
    if num_kv_blocks > 0:
        row_offset = bh * S_seq + row_start
        q0 = q_desc.load([row_offset, 0])
        q1 = q_desc.load([row_offset, 64])
        
        kv_offset = bh * S_seq + 0
        k0_cur = k_desc.load([kv_offset, 0])
        k1_cur = k_desc.load([kv_offset, 64])
        
        v0_cur = v_desc.load([kv_offset, 0])
        v1_cur = v_desc.load([kv_offset, 64])
        
        for j in range(num_kv_blocks):
            kv_start = j * BLOCK_N
            
            if j < num_kv_blocks - 1:
                next_kv_start = (j + 1) * BLOCK_N
                next_kv_offset = bh * S_seq + next_kv_start
                
                k0_next = k_desc.load([next_kv_offset, 0])
                k1_next = k_desc.load([next_kv_offset, 64])
                
                v0_next = v_desc.load([next_kv_offset, 0])
                v1_next = v_desc.load([next_kv_offset, 64])
            
            acc_S = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            acc_S = tl.dot(q0, k0_cur.T, acc_S)
            acc_S = tl.dot(q1, k1_cur.T, acc_S)
            
            S = acc_S * scale
            
            mask = (kv_start + col_idx[None, :]) < S_seq
            S = tl.where(mask, S, -float('inf'))
            
            m_prev = m
            row_max = tl.max(S, axis=1)
            m = tl.maximum(m_prev, row_max)
            
            P = tl.exp(S - m[None, :])
            P = P.to(tl.bfloat16)
            P = tl.where(mask, P, 0.0)
            
            row_sum = tl.sum(P, axis=1)
            
            exp_scale = tl.exp(m_prev - m)
            
            l = l * exp_scale + row_sum
            
            out0 *= exp_scale[:, None]
            out1 *= exp_scale[:, None]
            
            out0 = tl.dot(P, v0_cur, out0)
            out1 = tl.dot(P, v1_cur, out1)
            
            if j < num_kv_blocks - 1:
                k0_cur = k0_next
                k1_cur = k1_next
                v0_cur = v0_next
                v1_cur = v1_next
                
            else:
                break 
    
    if num_kv_blocks > 0:
        if l > 0.0:
            inv_l = 1.0 / l
            out0 *= inv_l[:, None]
            out1 *= inv_l[:, None]
    
    out0 = out0.to(tl.bfloat16)
    out1 = out1.to(tl.bfloat16)
    
    row_offset = bh * S_seq + row_start
    o_desc.store([row_offset, 0], out0)
    o_desc.store([row_offset, 64], out1)
    
    if num_kv_blocks > 0:
        lse = m + tl.log(l)
    else:
        lse = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
        
    lse = tl.where(valid_rows, lse, float('inf'))
    
    tl.store(lse_ptr + (bh * S_seq + row_start) + row_idx, lse, mask=valid_rows)


def run(Q, K, V, O, LSE):
    """Compute Non-causal Multi-Head Attention O and LSE into preallocated CUDA tensors."""
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    BLOCK_M = 256
    BLOCK_N = 128 
    
    Q_2D = Q.view(-1, D)
    K_2D = K.view(-1, D)
    V_2D = V.view(-1, D)
    O_2D = O.view(-1, D)
    
    q_desc = TensorDescriptor.from_tensor(Q_2D, block_shape=[BLOCK_M, 64])
    k_desc = TensorDescriptor.from_tensor(K_2D, block_shape=[BLOCK_N, 64])
    v_desc = TensorDescriptor.from_tensor(V_2D, block_shape=[BLOCK_N, 64])
    o_desc = TensorDescriptor.from_tensor(O_2D, block_shape=[BLOCK_M, 64], padding_option="zero")
    
    grid = (triton.cdiv(S, BLOCK_M), B * H)
    _attention_kernel[grid](
        q_desc, k_desc, v_desc, o_desc, LSE, S,
        D=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=8, num_stages=3
    )