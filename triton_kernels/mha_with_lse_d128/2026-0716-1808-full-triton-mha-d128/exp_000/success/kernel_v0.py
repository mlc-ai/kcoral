import torch
import triton
import triton.language as tl
import math
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _attention_kernel(
    q_desc, k_desc, v_desc, o_desc,
    lse_ptr, S_length, lse_stride,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    bh = pid_bh
    S = S_length
    row_offset = bh * S + pid_m * BLOCK_M
    
    q0 = q_desc.load([row_offset, 0])
    q1 = q_desc.load([row_offset, BLOCK_N])
    
    acc_o0 = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    acc_o1 = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    acc_m = tl.full((BLOCK_M,), float('-inf'), tl.float32)
    acc_sum = tl.zeros((BLOCK_M,), tl.float32)
    
    scale = 1.0 / math.sqrt(128.0)
    
    for k_tile in range(tl.cdiv(S_length, BLOCK_N)):
        col_offset = bh * S + k_tile * BLOCK_N
        
        k0 = k_desc.load([col_offset, 0])
        k1 = k_desc.load([col_offset, BLOCK_N])
        v0 = v_desc.load([col_offset, 0])
        v1 = v_desc.load([col_offset, BLOCK_N])
        
        s = tl.dot(q0, k0.T) + tl.dot(q1, k1.T)
        s *= scale
        
        cols = tl.arange(0, BLOCK_N)
        col_valid = k_tile * BLOCK_N + cols < S_length
        s = tl.where(col_valid[None, :], s, float('-inf'))
        
        row_max = tl.max(s, axis=1)
        m_new = tl.maximum(acc_m, row_max)
        alpha = tl.exp(acc_m - m_new)
        p = tl.exp(s - m_new[:, None])
        
        acc_sum = acc_sum * alpha + tl.sum(p, axis=1)
        acc_m = m_new
        
        p = p.to(tl.bfloat16)
        
        acc_o0 = acc_o0 * alpha[:, None] + tl.dot(p, v0)
        acc_o1 = acc_o1 * alpha[:, None] + tl.dot(p, v1)
        
    denom = acc_sum
    
    acc_o0 = acc_o0 / denom[:, None]
    acc_o1 = acc_o1 / denom[:, None]
    
    o_desc.store([row_offset, 0], acc_o0.to(tl.bfloat16))
    o_desc.store([row_offset, BLOCK_N], acc_o1.to(tl.bfloat16))
    
    rows = tl.arange(0, BLOCK_M)
    lse = acc_m + tl.log(denom)
    
    lse_ptr_curr = lse_ptr + (row_offset + rows) * lse_stride
    row_valid = row_offset + rows < (bh + 1) * S
    tl.store(lse_ptr_curr, lse, mask=row_valid)


def run(Q, K, V, O, LSE):
    """Compute Non-causal multi-head attention forward returning O and LSE."""
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    if S == 0:
        return
    
    Q = Q.reshape(B * H * S, D)
    K = K.reshape(B * H * S, D)
    V = V.reshape(B * H * S, D)
    O = O.reshape(B * H * S, D)
    LSE = LSE.reshape(B * H * S, 1)
    
    BLOCK_M, BLOCK_N = 128, 64
    
    q_desc = TensorDescriptor.from_tensor(Q, [BLOCK_M, BLOCK_N])
    k_desc = TensorDescriptor.from_tensor(K, [BLOCK_N, BLOCK_N])
    v_desc = TensorDescriptor.from_tensor(V, [BLOCK_N, BLOCK_N])
    o_desc = TensorDescriptor.from_tensor(O, [BLOCK_M, BLOCK_N])
    
    grid = (triton.cdiv(S, BLOCK_M), B * H)
    
    _attention_kernel[grid](
        q_desc, k_desc, v_desc, o_desc, LSE, S, LSE.stride(0),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=8, num_stages=3
    )