import torch
import triton
import triton.language as tl
import math


@triton.jit
def _attention_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    S, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_d = tl.program_id(1)
    bh = tl.program_id(2)
    
    q_start = pid_m * BLOCK_M
    d_start = pid_d * BLOCK_D
    
    row_mask = (q_start + tl.arange(0, BLOCK_M)) < S
    
    stride_row = 128
    q_base = Q_ptr + bh * S * stride_row
    k_base = K_ptr + bh * S * stride_row
    v_base = V_ptr + bh * S * stride_row
    
    Q_0 = tl.load(q_base + (q_start + tl.arange(0, BLOCK_M))[:, None] * stride_row + tl.arange(0, BLOCK_D)[None, :],
                  mask=row_mask[:, None], other=0.0)
    Q_1 = tl.load(q_base + (q_start + tl.arange(0, BLOCK_M))[:, None] * stride_row + BLOCK_D + tl.arange(0, BLOCK_D)[None, :],
                  mask=row_mask[:, None], other=0.0)
    
    O_acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    local_max = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    local_sum = tl.full((BLOCK_M,), 0.0, dtype=tl.float32)
    
    for j in range(0, S, BLOCK_N):
        col_mask = (j + tl.arange(0, BLOCK_N)) < S
        
        K_0 = tl.load(k_base + (j + tl.arange(0, BLOCK_N))[:, None] * stride_row + tl.arange(0, BLOCK_D)[None, :],
                      mask=col_mask[:, None], other=0.0)
        K_1 = tl.load(k_base + (j + tl.arange(0, BLOCK_N))[:, None] * stride_row + BLOCK_D + tl.arange(0, BLOCK_D)[None, :],
                      mask=col_mask[:, None], other=0.0)
        V = tl.load(v_base + (j + tl.arange(0, BLOCK_N))[:, None] * stride_row + d_start + tl.arange(0, BLOCK_D)[None, :],
                    mask=col_mask[:, None], other=0.0)
        
        P = tl.dot(Q_0.to(tl.bfloat16), K_0.T.to(tl.bfloat16))
        P += tl.dot(Q_1.to(tl.bfloat16), K_1.T.to(tl.bfloat16))
        
        P = tl.where(row_mask[:, None] & col_mask[None, :], P * scale, -float('inf'))
        
        m_prev = local_max
        local_max = tl.maximum(local_max, tl.max(P, axis=1))
        
        exp_scale = tl.exp(m_prev - local_max)
        O_acc *= exp_scale[:, None]
        
        P = P - local_max[:, None]
        P_exp = tl.exp(P)
        local_sum = local_sum * exp_scale + tl.sum(P_exp, axis=1)
        
        O_acc += tl.dot(P_exp.to(tl.bfloat16), V.to(tl.bfloat16))
    
    O = O_acc / local_sum[:, None]
    
    o_base = O_ptr + bh * S * stride_row
    tl.store(o_base + (q_start + tl.arange(0, BLOCK_M))[:, None] * stride_row + d_start + tl.arange(0, BLOCK_D)[None, :],
             O.to(tl.bfloat16), mask=row_mask[:, None])
    
    if pid_d == 0:
        lse = tl.where(row_mask, local_max + tl.math.log(local_sum), 0.0)
        tl.store(LSE_ptr + bh * S + q_start + tl.arange(0, BLOCK_M), lse, mask=row_mask)


def run(Q, K, V, O, LSE):
    """Compute Attention O and LSE into preallocated CUDA tensors."""
    torch.cuda.set_device(Q.device)
    
    B = Q.shape[0]
    H = Q.shape[1]
    S = Q.shape[2]
    
    scale = 1.0 / math.sqrt(128)
    
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_D = 64
    
    grid = (triton.cdiv(S, BLOCK_M), 2, B * H)
    _attention_kernel[grid](Q, K, V, O, LSE, S, scale, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D, num_warps=8, num_stages=3)