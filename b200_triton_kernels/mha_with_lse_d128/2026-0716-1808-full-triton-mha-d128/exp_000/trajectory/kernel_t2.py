import torch
import triton
import triton.language as tl
import math


@triton.jit
def _attention_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr,
    total_rows, S_length, lse_stride, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, D: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    row_offset = pid_bh * S_length + pid_m * BLOCK_M
    
    q_desc = tl.make_tensor_descriptor(q_ptr, shape=[total_rows, D],
                                        strides=[D, 1], block_shape=[BLOCK_M, BLOCK_K], padding_option="zero")
    k_desc = tl.make_tensor_descriptor(k_ptr, shape=[total_rows, D],
                                        strides=[D, 1], block_shape=[BLOCK_N, BLOCK_K], padding_option="zero")
    v_desc = tl.make_tensor_descriptor(v_ptr, shape=[total_rows, D],
                                        strides=[D, 1], block_shape=[BLOCK_N, BLOCK_K], padding_option="zero")
    o_desc = tl.make_tensor_descriptor(o_ptr, shape=[total_rows, D],
                                        strides=[D, 1], block_shape=[BLOCK_M, BLOCK_K], padding_option="zero")
    
    q0 = q_desc.load([row_offset, 0])
    q1 = q_desc.load([row_offset, BLOCK_K])
    
    acc_o0 = tl.zeros((BLOCK_M, BLOCK_K), tl.float32)
    acc_o1 = tl.zeros((BLOCK_M, BLOCK_K), tl.float32)
    acc_m = tl.full((BLOCK_M,), float('-inf'), tl.float32)
    acc_sum = tl.zeros((BLOCK_M,), tl.float32)
    
    for k_tile in range(tl.cdiv(S_length, BLOCK_N)):
        col_offset = pid_bh * S_length + k_tile * BLOCK_N
        
        k0 = k_desc.load([col_offset, 0])
        k1 = k_desc.load([col_offset, BLOCK_K])
        v0 = v_desc.load([col_offset, 0])
        v1 = v_desc.load([col_offset, BLOCK_K])
        
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
        
        acc_o0 = acc_o0 * alpha[:, None] + tl.dot(p, v0)
        acc_o1 = acc_o1 * alpha[:, None] + tl.dot(p, v1)
        
    denom = acc_sum
    
    acc_o0 = acc_o0 / denom[:, None]
    acc_o1 = acc_o1 / denom[:, None]
    
    o_desc.store([row_offset, 0], acc_o0.to(tl.bfloat16))
    o_desc.store([row_offset, BLOCK_K], acc_o1.to(tl.bfloat16))
    
    rows = tl.arange(0, BLOCK_M)
    lse = acc_m + tl.log(denom)
    
    lse_ptr_curr = lse_ptr + (row_offset + rows) * lse_stride
    row_valid = row_offset + rows < (pid_bh + 1) * S_length
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
    
    total_rows = B * H * S
    
    grid = (triton.cdiv(S, 128), B * H)
    
    _attention_kernel[grid](
        Q, K, V, O, LSE,
        total_rows, S, LSE.stride(0), scale,
        BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, D=D,
        num_warps=4, num_stages=3
    )