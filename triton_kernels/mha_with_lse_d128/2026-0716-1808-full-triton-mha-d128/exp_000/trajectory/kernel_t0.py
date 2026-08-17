import torch
import triton
import triton.language as tl
import math
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    lse_ptr,
    S_length,
    stride_q_row,
    stride_q_col,
    stride_k_row,
    stride_k_col,
    stride_v_row,
    stride_v_col,
    stride_o_row,
    stride_o_col,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    rows = tl.arange(0, 128)
    row_offset_q = pid_bh * S_length + pid_m * 128 + rows
    col_offset = tl.arange(0, 64)
    
    q0 = q_ptr + row_offset_q[:, None] * stride_q_row + col_offset[None, :] * stride_q_col
    q1 = q_ptr + row_offset_q[:, None] * stride_q_row + (col_offset[None, :] + 64) * stride_q_col
    
    q0 = tl.load(q0, mask=(row_offset_q[:, None] < pid_bh * S_length + S_length), other=0.0)
    q1 = tl.load(q1, mask=(row_offset_q[:, None] < pid_bh * S_length + S_length), other=0.0)
    
    acc_o0 = 0.0
    acc_o1 = 0.0
    acc_m = float('-inf')
    acc_sum = 0.0
    
    scale = 1.0 / math.sqrt(128)
    
    for k_idx in range(tl.cdiv(S_length, 128)):
        row_offset_k = pid_bh * S_length + k_idx * 128 + tl.arange(0, 128)
        col_offset_k = tl.arange(0, 64)
        
        k0 = k_ptr + row_offset_k[:, None] * stride_k_row + col_offset_k[None, :] * stride_k_col
        k1 = k_ptr + row_offset_k[:, None] * stride_k_row + (col_offset_k[None, :] + 64) * stride_k_col
        
        k0 = tl.load(k0, mask=(row_offset_k[:, None] < pid_bh * S_length + S_length), other=0.0)
        k1 = tl.load(k1, mask=(row_offset_k[:, None] < pid_bh * S_length + S_length), other=0.0)
        
        s = tl.dot(q0, k0.T) + tl.dot(q1, k1.T)
        s *= scale
        
        cols = tl.arange(0, 128)
        mask = k_idx * 128 + cols < S_length
        s = tl.where(mask[None, :], s, -1e20)
        
        row_max = tl.argmax(s, axis=1)
        m_new = tl.maximum(acc_m, row_max)
        alpha = tl.exp(acc_m - m_new)
        
        p = tl.exp(s - m_new[:, None])
        
        acc_sum = acc_sum * alpha + tl.sum(p, axis=1)
        acc_m = m_new
        
        acc_o0 = acc_o0 * alpha[:, None]
        acc_o1 = acc_o1 * alpha[:, None]
        
        v0 = v_ptr + row_offset_k[:, None] * stride_v_row + col_offset_k[None, :] * stride_v_col
        v1 = v_ptr + row_offset_k[:, None] * stride_v_row + (col_offset_k[None, :] + 64) * stride_v_col
        
        v0 = tl.load(v0, mask=(row_offset_k[:, None] < pid_bh * S_length + S_length), other=0.0)
        v1 = tl.load(v1, mask=(row_offset_k[:, None] < pid_bh * S_length + S_length), other=0.0)
        
        acc_o0 += tl.dot(p, v0)
        acc_o1 += tl.dot(p, v1)
        
    denom = acc_sum
    
    acc_o0 = acc_o0 / denom[:, None]
    acc_o1 = acc_o1 / denom[:, None]
    
    cols0 = tl.arange(0, 64)
    cols1 = tl.arange(64, 128)
    
    out_ptr0 = o_ptr + row_offset_q[:, None] * stride_o_row + cols0[None, :] * stride_o_col
    out_ptr1 = o_ptr + row_offset_q[:, None] * stride_o_row + cols1[None, :] * stride_o_col
    
    tl.store(out_ptr0, acc_o0, mask=(row_offset_q[:, None] < pid_bh * S_length + S_length))
    tl.store(out_ptr1, acc_o1, mask=(row_offset_q[:, None] < pid_bh * S_length + S_length))
    
    lse = acc_m + tl.log(acc_sum)
    
    lse_ptr_curr = lse_ptr + pid_bh * S_length + pid_m * 128 + rows
    tl.store(lse_ptr_curr, lse, mask=(row_offset_q < pid_bh * S_length + S_length))


def run(Q, K, V, O, LSE):
    """Compute Non-causal multi-head attention forward returning O and LSE."""
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    if S == 0:
        return
    
    scale = 1.0 / math.sqrt(D)
    
    row_q = torch.arange(0, B * H * S, device=Q.device)
    col_q = torch.arange(0, D, device=Q.device)
    Q = Q.reshape(B * H * S, D)
    K = K.reshape(B * H * S, D)
    V = V.reshape(B * H * S, D)
    
    q_desc = TensorDescriptor.from_tensor(Q, [128, 64])
    k_desc = TensorDescriptor.from_tensor(K, [128, 64])
    v_desc = TensorDescriptor.from_tensor(V, [128, 64])
    
    grid = (triton.cdiv(S, 128), B * H)
    
    _attention_kernel[grid](
        Q, K, V, O, LSE,
        S,
        D, 1, D, 1, D, 1, D, 1,
        scale,
        num_warps=4, num_stages=3
    )