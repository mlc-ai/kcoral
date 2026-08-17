import torch
import triton
import triton.language as tl
import math


@triton.jit
def _mha_kernel_forward(
    ptr_q, ptr_k, ptr_v, ptr_o, ptr_lse,
    S_len, H, d_dim, s_dim,
    sqrt_d,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
):
    B = 4
    
    desc_q = tl.make_tensor_descriptor(
        ptr_q, shape=[B, H, S_len, D], strides=[H * S_len * D, S_len * D, D, 1],
        block_shape=[1, 1, BLOCK_M, D], padding_option="zero"
    )
    desc_k = tl.make_tensor_descriptor(
        ptr_k, shape=[B, H, S_len, D], strides=[H * S_len * D, S_len * D, D, 1],
        block_shape=[1, 1, BLOCK_N, D], padding_option="zero"
    )
    desc_v = tl.make_tensor_descriptor(
        ptr_v, shape=[B, H, S_len, D], strides=[H * S_len * D, S_len * D, D, 1],
        block_shape=[1, 1, BLOCK_N, D], padding_option="zero"
    )
    
    pid_m = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)
    
    start_seq_q = pid_m * BLOCK_M
    n_blocks = triton.cdiv(S_len, BLOCK_N)
    
    q = desc_q.load([b, h, start_seq_q, 0])
    
    o_acc = tl.zeros((BLOCK_M, D), tl.float32)
    m = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    l = tl.zeros((BLOCK_M,), dtype=tl.float32)
    
    mask_k = []
    for j in range(n_blocks):
        seq_idx = j * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_k.append(seq_idx < S_len)
    
    for j in range(n_blocks):
        k = desc_k.load([b, h, j * BLOCK_N, 0])
        v = desc_v.load([b, h, j * BLOCK_N, 0])
        
        s_acc = tl.dot(q, k.T)
        s_acc /= sqrt_d
        s_acc = tl.where(mask_k[j][:, None], s_acc, -float('inf'))
        
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
    
    o_acc = o_acc / l[:, None]
    lse = m + tl.log(l)
    
    h_dim = H * s_dim
    out_s = ptr_o + b * h_dim + h * s_dim
    col = tl.arange(0, D)
    out_ptr = out_s + seq_idx_q[:, None] * d_dim + col[None, :]
    
    O_bf16 = o_acc.to(tl.bfloat16)
    tl.store(out_ptr, O_bf16, mask=mask_m[:, None])
    
    lse_s = ptr_lse + b * (H * S_len) + h * S_len
    lse_ptr = lse_s + seq_idx_q
    tl.store(lse_ptr, lse, mask=mask_m)


BLOCK_M = 64
BLOCK_N = 64


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S_len, D = Q.shape
    grid = (triton.cdiv(S_len, BLOCK_M), H, B)
    
    s_dim = S_len * D
    d_dim = D
    sqrt_d = math.sqrt(D)
    
    _mha_kernel_forward[grid](
        Q, K, V, O, LSE,
        S_len, H, d_dim, s_dim,
        sqrt_d,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, D=D,
        num_warps=4, num_stages=4
    )