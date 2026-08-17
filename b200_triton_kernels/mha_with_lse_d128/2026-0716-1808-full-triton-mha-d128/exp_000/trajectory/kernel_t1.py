import torch
import triton
import triton.language as tl
import math
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _attention_kernel(
    q_desc,
    k_desc,
    v_desc,
    o_desc,
    lse_ptr,
    S_length,
    lse_stride_row,
    scale,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    global_row = pid_bh * S_length + pid_m * 128
    
    q0 = q_desc.load([global_row, 0])
    q1 = q_desc.load([global_row, 64])
    
    acc_o0 = tl.zeros((128, 64), dtype=tl.float32)
    acc_o1 = tl.zeros((128, 64), dtype=tl.float32)
    
    acc_m = tl.full((128,), fill_value=float('-inf'), dtype=tl.float32)
    acc_sum = tl.zeros((128,), dtype=tl.float32)
    
    num_k_tiles = tl.cdiv(S_length, 128)
    
    for k_idx in range(num_k_tiles):
        global_col = k_idx * 128
        
        if global_col >= S_length:
            continue
            
        k0 = k_desc.load([global_col, 0])
        k1 = k_desc.load([global_col, 64])
        v0 = v_desc.load([global_col, 0])
        v1 = v_desc.load([global_col, 64])
        
        s = tl.dot(q0, k0.T) + tl.dot(q1, k1.T)
        s *= scale
        
        cols = tl.arange(0, 128)
        col_valid = global_col + cols < S_length
        s = tl.where(col_valid[None, :], s, -1e20)
        
        block_max = s.max(axis=1)
        m_new = tl.maximum(acc_m, block_max)
        alpha = tl.exp(acc_m - m_new)
        
        p = tl.exp(s - m_new[:, None])
        
        acc_sum = acc_sum * alpha + p.sum(axis=1)
        acc_m = m_new
        
        acc_o0 = acc_o0 * alpha[:, None] + tl.dot(p, v0)
        acc_o1 = acc_o1 * alpha[:, None] + tl.dot(p, v1)
        
    denom = acc_sum
    
    acc_o0 = acc_o0 / denom[:, None]
    acc_o1 = acc_o1 / denom[:, None]
    
    o_desc.store([global_row, 0], acc_o0)
    o_desc.store([global_row, 64], acc_o1)
    
    rows = tl.arange(0, 128)
    lse = acc_m + tl.log(denom)
    
    lse_ptr_curr = lse_ptr + (global_row + rows) * lse_stride_row
    row_valid = global_row + rows < S_length
    tl.store(lse_ptr_curr, lse, mask=row_valid)


def run(Q, K, V, O, LSE):
    """Compute Non-causal multi-head attention forward returning O and LSE."""
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    if S == 0:
        return
    
    scale = 1.0 / math.sqrt(D)
    
    Q = Q.reshape(B * H * S, D)
    K = K.reshape(B * H * S, D)
    V = V.reshape(B * H * S, D)
    O = O.reshape(B * H * S, D)
    LSE = LSE.reshape(B * H * S, 1)
    
    q_desc = TensorDescriptor.from_tensor(Q, [128, 64])
    k_desc = TensorDescriptor.from_tensor(K, [128, 64])
    v_desc = TensorDescriptor.from_tensor(V, [128, 64])
    o_desc = TensorDescriptor.from_tensor(O, [128, 64])
    
    grid = (triton.cdiv(S, 128), B * H)
    
    _attention_kernel[grid](
        q_desc, k_desc, v_desc, o_desc, LSE,
        S, LSE.stride(0), scale,
        num_warps=4, num_stages=1
    )