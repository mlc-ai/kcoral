import torch
import triton
import triton.language as tl
import math


@triton.jit
def _mha_kernel_forward(
    ptr_q, ptr_k, ptr_v, ptr_o, ptr_lse,
    S_len, H, sqrt_d,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
):
    B = 4
    
    b_offset = b * H * S_len * D
    desc_q = tl.make_tensor_descriptor(
        ptr_q + b_offset, shape=[H, S_len, D], strides=[S_len * D, D, 1],
        block_shape=[1, BLOCK_M, D], padding_option="zero"
    )
    desc_k = tl.make_tensor_descriptor(
        ptr_k + b_offset, shape=[H, S_len, D], strides=[S_len * D, D, 1],
        block_shape=[1, BLOCK_N, D], padding_option="zero"
    )
    desc_v = tl.make_tensor_descriptor(
        ptr_v + b_offset, shape=[H, S_len, D], strides=[S_len * D, D, 1],
        block_shape=[1, BLOCK_N, D], padding_option="zero"
    )
    
    h_offset = h
    start_seq_q = pid_m * BLOCK_M
    n_blocks = triton.cdiv(S_len, BLOCK_N)
    
    q = desc_q.load([h_offset, start_seq_q, 0])
    
    desc_o = tl.make_tensor_descriptor(
        ptr_o + b_offset, shape=[H, S_len, D], strides=[S_len * D, D, 1],
        block_shape=[1, BLOCK_M, D]
    )
    
    s_dim = S_len * D
    lse_offset = b * H * S_len
    desc_lse = tl.make_tensor_descriptor(
        ptr_lse + lse_offset, shape=[H, S_len], strides=[S_len, 1],
        block_shape=[1, BLOCK_M], padding_option="zero"
    )
    
    o_acc = tl.zeros((BLOCK_M, D), tl.float32)
    m = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    l = tl.zeros((BLOCK_M,), dtype=tl.float32)
    
    for j in range(n_blocks):
        k = desc_k.load([h_offset, j * BLOCK_N, 0])
        v = desc_v.load([h_offset, j * BLOCK_N, 0])
        
        s_acc = tl.dot(q, k.T)
        s_acc /= sqrt_d
        
        col_offsets = j * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_k = col_offsets < S_len
        
        s_acc = tl.where(mask_k[None, :], s_acc, -float('inf'))
        
        m_curr = tl.max(s_acc, axis=1)
        m_new = tl.maximum(m, m_curr)
        scale = tl.exp(m - m_new)
        
        p = tl.exp(s_acc - m_new[:, None])
        l_curr = tl.sum(p, axis=1)
        l = l * scale + l_curr
        
        m = m_new
        o_acc = o_acc * scale[:, None]
        
        o_acc += tl.dot(p, v)
    
    seq_idx_q = start_seq_q + tl.arange(0, BLOCK_M)
    mask_m = seq_idx_q < S_len
    
    o_acc = o_acc / tl.where(mask_m[:, None], l[:, None], 1.0)
    lse = tl.where(mask_m, m + tl.log(l), float('-inf'))
    
    O_bf16 = o_acc.to(tl.bfloat16)
    desc_o.store([h_offset, start_seq_q, 0], O_bf16, mask=mask_m[:, None])
    desc_lse.store([h_offset, start_seq_q], lse, mask=mask_m)


BLOCK_M = 64
BLOCK_N = 64


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S_len, D = Q.shape
    grid = (triton.cdiv(S_len, BLOCK_M), H, B)
    
    sqrt_d = math.sqrt(D)
    
    _mha_kernel_forward[grid](
        Q, K, V, O, LSE,
        S_len, H, sqrt_d,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, D=D,
        num_warps=4, num_stages=3
    )