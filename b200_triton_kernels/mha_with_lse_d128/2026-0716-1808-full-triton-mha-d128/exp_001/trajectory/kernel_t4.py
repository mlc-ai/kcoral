import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
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
    
    s_Q = tl.empty((2, BLOCK_M, 128), dtype=tl.bfloat16, name="s_Q")
    s_K = tl.empty((2, BLOCK_N, 128), dtype=tl.bfloat16, name="s_K")
    s_V = tl.empty((2, BLOCK_N, BLOCK_D), dtype=tl.bfloat16, name="s_V")
    
    for i in range(0, 128, 32):
        ptr = Q_ptr + bh * S * 128 + (q_start + tl.arange(0, BLOCK_M))[:, None] * 128 + i + tl.arange(0, 32)[None, :]
        mask = (q_start + tl.arange(0, BLOCK_M))[:, None] < S
        Q_chunk = tl.load(ptr, mask=mask, other=0.0)
        s_Q[0, (q_start + tl.arange(0, BLOCK_M))[:, None], i + tl.arange(0, 32)[None, :]] = Q_chunk
        
    o_acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    local_max = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    local_sum = tl.full((BLOCK_M,), 0.0, dtype=tl.float32)
    
    num_iters = triton.cdiv(S, BLOCK_N)
    
    if num_iters > 0:
        # Prologue load for K and V
        ptr_k0 = K_ptr + bh * S * 128 + (0 + tl.arange(0, BLOCK_N))[:, None] * 128 + tl.arange(0, 64)[None, :]
        ptr_k1 = K_ptr + bh * S * 128 + (0 + tl.arange(0, BLOCK_N))[:, None] * 128 + 64 + tl.arange(0, 64)[None, :]
        mask_k = (0 + tl.arange(0, BLOCK_N))[:, None] < S
        K_0 = tl.load(ptr_k0, mask=mask_k, other=0.0)
        K_1 = tl.load(ptr_k1, mask=mask_k, other=0.0)
        s_K[0, (0 + tl.arange(0, BLOCK_N))[:, None], tl.arange(0, 64)[None, :]] = K_0
        s_K[0, (0 + tl.arange(0, BLOCK_N))[:, None], 64 + tl.arange(0, 64)[None, :]] = K_1
        
        ptr_v0 = V_ptr + bh * S * 128 + (0 + tl.arange(0, BLOCK_N))[:, None] * 128 + d_start + tl.arange(0, BLOCK_D)[None, :]
        V_0 = tl.load(ptr_v0, mask=mask_k, other=0.0)
        s_V[0, (0 + tl.arange(0, BLOCK_N))[:, None], tl.arange(0, BLOCK_D)[None, :]] = V_0

    for iteration in range(num_iters):
        j = iteration * BLOCK_N
        buf_idx = iteration % 2
        next_buf_idx = (iteration + 1) % 2
        
        if iteration + 1 < num_iters:
            next_j = (iteration + 1) * BLOCK_N
            ptr_k0 = K_ptr + bh * S * 128 + (next_j + tl.arange(0, BLOCK_N))[:, None] * 128 + tl.arange(0, 64)[None, :]
            ptr_k1 = K_ptr + bh * S * 128 + (next_j + tl.arange(0, BLOCK_N))[:, None] * 128 + 64 + tl.arange(0, 64)[None, :]
            mask_k = (next_j + tl.arange(0, BLOCK_N))[:, None] < S
            K_0 = tl.load(ptr_k0, mask=mask_k, other=0.0)
            K_1 = tl.load(ptr_k1, mask=mask_k, other=0.0)
            s_K[next_buf_idx, (next_j + tl.arange(0, BLOCK_N))[:, None], tl.arange(0, 64)[None, :]] = K_0
            s_K[next_buf_idx, (next_j + tl.arange(0, BLOCK_N))[:, None], 64 + tl.arange(0, 64)[None, :]] = K_1
            
            ptr_v0 = V_ptr + bh * S * 128 + (next_j + tl.arange(0, BLOCK_N))[:, None] * 128 + d_start + tl.arange(0, BLOCK_D)[None, :]
            V_0 = tl.load(ptr_v0, mask=mask_k, other=0.0)
            s_V[next_buf_idx, (next_j + tl.arange(0, BLOCK_N))[:, None], tl.arange(0, BLOCK_D)[None, :]] = V_0

        Q_0 = s_Q[0, :, 0:64]
        Q_1 = s_Q[0, :, 64:128]
        
        K_0 = s_K[buf_idx, :, 0:64]
        K_1 = s_K[buf_idx, :, 64:128]
        
        p = tl.dot(Q_0, K_0.T)
        p += tl.dot(Q_1, K_1.T)
        
        valid_q = (q_start + tl.arange(0, BLOCK_M)) < S
        valid_k = (j + tl.arange(0, BLOCK_N)) < S
        p = tl.where(valid_q[:, None] & valid_k[None, :], p * scale, -float('inf'))
        
        m_prev = local_max
        local_max = tl.maximum(m_prev, tl.max(p, axis=1))
        
        exp_scale = tl.exp(m_prev - local_max)
        o_acc = o_acc * exp_scale[:, None]
        
        p = p - local_max[:, None]
        p_exp = tl.exp(p)
        local_sum = local_sum * exp_scale + tl.sum(p_exp, axis=1)
        
        V_0 = s_V[buf_idx, :, :]
        o_acc += tl.dot(p_exp.to(tl.bfloat16), V_0.to(tl.bfloat16))
        
    o_ptr = O_ptr + bh * S * 128 + (q_start + tl.arange(0, BLOCK_M))[:, None] * 128 + d_start + tl.arange(0, BLOCK_D)[None, :]
    o_val = o_acc / local_sum[:, None]
    tl.store(o_ptr, o_val.to(tl.bfloat16), mask=(q_start + tl.arange(0, BLOCK_M))[:, None] < S)
    
    if pid_d == 0:
        lse = tl.where((q_start + tl.arange(0, BLOCK_M)) < S, local_max + tl.math.log(local_sum), 0.0)
        lse_ptr = LSE_ptr + bh * S + q_start + tl.arange(0, BLOCK_M)
        tl.store(lse_ptr, lse, mask=(q_start + tl.arange(0, BLOCK_M)) < S)


def run(Q, K, V, O, LSE):
    """Compute Attention O and LSE into preallocated CUDA tensors."""
    torch.cuda.set_device(Q.device)
    
    B = Q.shape[0]
    H = Q.shape[1]
    S = Q.shape[2]
    D = Q.shape[3]
    
    scale = 1.0 / math.sqrt(D)
    
    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_D = 32
    
    grid = (triton.cdiv(S, BLOCK_M), 4, B * H)
    
    _attention_kernel[grid](Q, K, V, O, LSE, S, scale, 
                           BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
                           num_warps=4, num_stages=2)