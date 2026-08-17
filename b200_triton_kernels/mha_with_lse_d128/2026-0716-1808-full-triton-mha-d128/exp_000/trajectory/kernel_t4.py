import torch
import triton
import triton.language as tl
import math
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _attention_kernel(
    q_desc, k_desc, v_desc, o_desc,
    lse_ptr, S_length,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    bh = pid_bh
    S = S_length
    row_offset = bh * S + pid_m * BLOCK_M
    
    q = q_desc.load([row_offset, 0])
    
    acc_o = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    acc_m = tl.full((BLOCK_M,), float('-inf'), tl.float32)
    acc_sum = tl.zeros((BLOCK_M,), tl.float32)
    
    scale = 1.0 / math.sqrt(128.0)
    
    for k_tile in range(tl.cdiv(S_length, BLOCK_N)):
        col_offset = bh * S + k_tile * BLOCK_N
        
        k = k_desc.load([col_offset, 0])
        v = v_desc.load([col_offset, 0])
        
        s = tl.dot(q, k.T) * scale
        
        cols = tl.arange(0, BLOCK_N)
        col_valid = k_tile * BLOCK_N + cols < S_length
        s = tl.where(col_valid[None, :], s, float('-inf'))
        
        row_max = tl.max(s, axis=1)
        m_new = tl.maximum(acc_m, row_max)
        alpha = tl.exp(acc_m - m_new)
        p = tl.exp(s - m_new[:, None])
        
        acc_sum = acc_sum * alpha + tl.sum(p, axis=1)
        acc_m = m_new
        
        acc_o = acc_o * alpha[:, None] + tl.dot(p, v)
        
    denom = acc_sum
    
    if denom > 0.0:
        acc_o = acc_o / denom[:, None]
    
    o_desc.store([row_offset, 0], acc_o.to(tl.bfloat16))
    
    rows = tl.arange(0, BLOCK_M)
    lse = acc_m + tl.log(denom)
    
    lse_ptr_curr = lse_ptr + row_offset + rows
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
    
    BLOCK_M, BLOCK_N = 128, 128
    
    q_desc = TensorDescriptor.from_tensor(Q, [BLOCK_M, BLOCK_N])
    k_desc = TensorDescriptor.from_tensor(K, [BLOCK_N, BLOCK_N])
    v_desc = TensorDescriptor.from_tensor(V, [BLOCK_N, BLOCK_N])
    o_desc = TensorDescriptor.from_tensor(O, [BLOCK_M, BLOCK_N])
    
    grid = (triton.cdiv(S, BLOCK_M), B * H)
    
    _attention_kernel[grid](
        q_desc, k_desc, v_desc, o_desc, LSE, S,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=8, num_stages=3
    )