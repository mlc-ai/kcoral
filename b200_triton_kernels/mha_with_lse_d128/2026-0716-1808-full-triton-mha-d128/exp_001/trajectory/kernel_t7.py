import torch
import triton
import triton.language as tl
import math


@triton.jit
def _attention_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    S, scale, num_iters,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_d = tl.program_id(1)
    bh = tl.program_id(2)
    
    q_start = pid_m * BLOCK_M
    d_start = pid_d * BLOCK_D
    
    arange_M = tl.arange(0, BLOCK_M)
    arange_N = tl.arange(0, BLOCK_N)
    arange_D = tl.arange(0, BLOCK_D)
    
    valid_rows = (q_start + arange_M) < S
    
    Q_0 = tl.load(Q_ptr + bh * S * 128 + (q_start + arange_M)[:, None] * 128 + arange_D[None, :],
                  mask=valid_rows[:, None], other=0.0)
    Q_1 = tl.load(Q_ptr + bh * S * 128 + (q_start + arange_M)[:, None] * 128 + BLOCK_D + arange_D[None, :],
                  mask=valid_rows[:, None], other=0.0)
    
    o_acc_0 = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    o_acc_1 = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    local_max = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    local_sum = tl.full((BLOCK_M,), 0.0, dtype=tl.float32)
    
    for k_tile in range(num_iters):
        j = k_tile * BLOCK_N
        valid_k = (j + arange_N) < S
        
        K_0 = tl.load(K_ptr + bh * S * 128 + (j + arange_N)[:, None] * 128 + arange_D[None, :],
                       mask=valid_k[:, None], other=0.0)
        K_1 = tl.load(K_ptr + bh * S * 128 + (j + arange_N)[:, None] * 128 + BLOCK_D + arange_D[None, :],
                       mask=valid_k[:, None], other=0.0)
        V_0 = tl.load(V_ptr + bh * S * 128 + (j + arange_N)[:, None] * 128 + d_start + arange_D[None, :],
                       mask=valid_k[:, None], other=0.0)
        V_1 = tl.load(V_ptr + bh * S * 128 + (j + arange_N)[:, None] * 128 + BLOCK_D + d_start + arange_D[None, :],
                       mask=valid_k[:, None], other=0.0)
        
        p = tl.dot(Q_0, K_0.T)
        p += tl.dot(Q_1, K_1.T)
        
        valid_q = (q_start + arange_M) < S
        
        p = tl.where(valid_q[:, None] & valid_k[None, :], p * scale, -float('inf'))
        
        m_prev = local_max
        local_max = tl.maximum(m_prev, tl.max(p, axis=1))
        
        exp_scale = tl.exp(m_prev - local_max)
        o_acc_0 = o_acc_0 * exp_scale[:, None]
        o_acc_1 = o_acc_1 * exp_scale[:, None]
        
        p = p - local_max[:, None]
        p_exp = tl.exp(p)
        local_sum = local_sum * exp_scale + tl.sum(p_exp, axis=1)
        
        p_exp = p_exp.to(tl.bfloat16)
        
        o_acc_0 += tl.dot(p_exp, V_0)
        o_acc_1 += tl.dot(p_exp, V_1)
        
    o_0 = o_acc_0 / local_sum[:, None]
    o_1 = o_acc_1 / local_sum[:, None]
    
    if pid_d == 0:
        out_ptr = O_ptr + bh * S * 128 + (q_start + arange_M)[:, None] * 128 + arange_D[None, :]
        tl.store(out_ptr, o_0.to(tl.bfloat16), mask=valid_rows[:, None])
    else:
        out_ptr = O_ptr + bh * S * 128 + (q_start + arange_M)[:, None] * 128 + BLOCK_D + arange_D[None, :]
        tl.store(out_ptr, o_1.to(tl.bfloat16), mask=valid_rows[:, None])
    
    if pid_d == 0:
        lse = tl.where(valid_rows, local_max + tl.math.log(local_sum), 0.0)
        lse_ptr = LSE_ptr + bh * S + q_start + arange_M
        tl.store(lse_ptr, lse, mask=valid_rows)


def run(Q, K, V, O, LSE):
    """Compute Attention O and LSE into preallocated CUDA tensors."""
    torch.cuda.set_device(Q.device)
    
    B = Q.shape[0]
    H = Q.shape[1]
    S = Q.shape[2]
    D = Q.shape[3]
    
    scale = 1.0 / math.sqrt(D)
    
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_D = 64
    
    num_iters = (S + BLOCK_N - 1) // BLOCK_N
    num_m_tiles = triton.cdiv(S, BLOCK_M)
    grid = (num_m_tiles, 2, B * H)
    
    _attention_kernel[grid](Q, K, V, O, LSE, S, scale, num_iters,
                           BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
                           num_warps=8, num_stages=4)