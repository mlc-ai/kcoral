import torch
import triton
import triton.language as tl


@triton.jit
def _attention_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    S_seq, D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """
    Optimized FlashAttention kernel for bf16 inputs targeting Hopper SM90.
    
    Computes Output O and Log-Sum-Exp (LSE) for non-causal multi-head attention.
    Uses a single-pass approach with inline QK^T and PV GEMMs.
    """
    row_start = tl.program_id(0) * BLOCK_M
    bh = tl.program_id(1)
    
    m = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    l = tl.zeros((BLOCK_M,), dtype=tl.float32)
    out0 = tl.zeros((BLOCK_M, 64), dtype=tl.float32)
    out1 = tl.zeros((BLOCK_M, 64), dtype=tl.float32)
    
    scale = 1.0 / tl.sqrt(D)
    
    row_idx = tl.arange(0, BLOCK_M)
    col_idx = tl.arange(0, BLOCK_N)
    c_idx_0 = tl.arange(0, 64)
    c_idx_1 = tl.arange(0, 64)
    
    row_mask = (row_start + row_idx) < S_seq
    
    q0 = tl.load(Q_ptr + (bh * S_seq + row_start + row_idx[:, None]) * D + c_idx_0[None, :],
                 mask=row_mask[:, None], other=0.0)
    q1 = tl.load(Q_ptr + (bh * S_seq + row_start + row_idx[:, None]) * D + 64 + c_idx_1[None, :],
                 mask=row_mask[:, None], other=0.0)
    
    num_kv_blocks = (S_seq + BLOCK_N - 1) // BLOCK_N
    
    if num_kv_blocks > 0:
        k0_cur = tl.load(K_ptr + (bh * S_seq + 0 + col_idx[:, None]) * D + c_idx_0[None, :],
                         mask=((0 + col_idx) < S_seq)[:, None], other=0.0)
        k1_cur = tl.load(K_ptr + (bh * S_seq + 0 + col_idx[:, None]) * D + 64 + c_idx_1[None, :],
                         mask=((0 + col_idx) < S_seq)[:, None], other=0.0)
                         
        v0_cur = tl.load(V_ptr + (bh * S_seq + 0 + col_idx[:, None]) * D + c_idx_0[None, :],
                         mask=((0 + col_idx) < S_seq)[:, None], other=0.0)
        v1_cur = tl.load(V_ptr + (bh * S_seq + 0 + col_idx[:, None]) * D + 64 + c_idx_1[None, :],
                         mask=((0 + col_idx) < S_seq)[:, None], other=0.0)
        
        for j in range(num_kv_blocks):
            kv_start = j * BLOCK_N
            kv_mask = (kv_start + col_idx) < S_seq
            
            if j < num_kv_blocks - 1:
                next_kv_start = (j + 1) * BLOCK_N
                
                k0_next = tl.load(K_ptr + (bh * S_seq + next_kv_start + col_idx[:, None]) * D + c_idx_0[None, :],
                                  mask=(kv_mask)[:, None], other=0.0)
                k1_next = tl.load(K_ptr + (bh * S_seq + next_kv_start + col_idx[:, None]) * D + 64 + c_idx_1[None, :],
                                  mask=(kv_mask)[:, None], other=0.0)
                                  
                v0_next = tl.load(V_ptr + (bh * S_seq + next_kv_start + col_idx[:, None]) * D + c_idx_0[None, :],
                                  mask=(kv_mask)[:, None], other=0.0)
                v1_next = tl.load(V_ptr + (bh * S_seq + next_kv_start + col_idx[:, None]) * D + 64 + c_idx_1[None, :],
                                  mask=(kv_mask)[:, None], other=0.0)
            
            acc_S = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            acc_S = tl.dot(q0, k0_cur.T, acc_S)
            acc_S = tl.dot(q1, k1_cur.T, acc_S)
            
            S = acc_S * scale
            
            mask = (kv_start + col_idx[None, :]) < S_seq
            S = tl.where(mask, S, -float('inf'))
            
            m_prev = m
            m = tl.maximum(m_prev, tl.max(S, axis=1))
            
            P = tl.exp(S - m[None, :])
            
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
                
    inv_l = 1.0 / l
    out0 *= inv_l[:, None]
    out1 *= inv_l[:, None]
    
    out0 = out0.to(tl.bfloat16)
    out1 = out1.to(tl.bfloat16)
    
    o_mask = row_mask[:, None]
    
    tl.store(O_ptr + (bh * S_seq + row_start + row_idx[:, None]) * D + c_idx_0[None, :], out0, mask=o_mask)
    tl.store(O_ptr + (bh * S_seq + row_start + row_idx[:, None]) * D + 64 + c_idx_1[None, :], out1, mask=o_mask)
    
    if num_kv_blocks > 0:
        lse = m + tl.log(l)
    else:
        lse = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    
    tl.store(LSE_ptr + bh * S_seq + row_start + row_idx, lse, mask=row_mask)


def run(Q, K, V, O, LSE):
    """Compute Non-causal Multi-Head Attention O and LSE into preallocated CUDA tensors."""
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    BLOCK_M = 128
    BLOCK_N = 128 
    
    Q_2D = Q.view(-1, D)
    K_2D = K.view(-1, D)
    V_2D = V.view(-1, D)
    O_2D = O.view(-1, D)
    
    grid = (triton.cdiv(S, BLOCK_M), B * H)
    _attention_kernel[grid](
        Q_2D, K_2D, V_2D, O_2D, LSE, S,
        D=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=8, num_stages=3
    )