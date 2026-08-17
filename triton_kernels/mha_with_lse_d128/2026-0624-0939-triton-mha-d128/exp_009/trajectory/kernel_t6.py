import torch
import triton
import triton.language as tl
import math


def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


triton.set_allocator(alloc_fn)


@triton.jit
def _mha_kernel_forward(
    Q, K, V, O, LSE,
    B, H, S_len, sqrt_d,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)
    
    desc_q = tl.make_tensor_descriptor(
        Q, shape=[B, H, S_len, D], strides=[H * S_len * D, S_len * D, D, 1],
        block_shape=[1, 1, BLOCK_M, D], padding_option="zero"
    )
    desc_k = tl.make_tensor_descriptor(
        K, shape=[B, H, S_len, D], strides=[H * S_len * D, S_len * D, D, 1],
        block_shape=[1, 1, BLOCK_N, D], padding_option="zero"
    )
    desc_v = tl.make_tensor_descriptor(
        V, shape=[B, H, S_len, D], strides=[H * S_len * D, S_len * D, D, 1],
        block_shape=[1, 1, BLOCK_N, D], padding_option="zero"
    )
    
    start_seq_q = pid_m * BLOCK_M
    n_blocks = tl.cdiv(S_len, BLOCK_N)
    
    q = desc_q.load([b, h, start_seq_q, 0])
    
    o_acc = tl.zeros((BLOCK_M, D), tl.float32)
    m = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    l = tl.zeros((BLOCK_M,), dtype=tl.float32)
    
    for j in range(n_blocks):
        seq_idx = j * BLOCK_N
        
        k = desc_k.load([b, h, seq_idx, 0])
        v = desc_v.load([b, h, seq_idx, 0])
        
        s_acc = tl.dot(q, k.T)
        s_acc /= sqrt_d
        
        col_offsets = seq_idx + tl.arange(0, BLOCK_N)
        mask_k = col_offsets < S_len
        s_acc = tl.where(mask_k[None, :], s_acc, -float('inf'))
        
        m_curr = tl.max(s_acc, axis=1)
        m_new = tl.maximum(m, m_curr)
        scale = tl.exp(m - m_new)
        
        p = tl.exp(s_acc - m_new[:, None])
        l_curr = tl.sum(p, axis=1)
        l = l * scale + l_curr
        
        m = m_new
        
        o_acc = o_acc * scale[:, None] + tl.dot(p, v)
    
    seq_idx_q = start_seq_q + tl.arange(0, BLOCK_M)
    mask_m = seq_idx_q < S_len
    
    o_acc = o_acc / l[:, None]
    lse = m + tl.log(l)
    
    O_bf16 = o_acc.to(tl.bfloat16)
    
    out_s = O + b * H * S_len * D + h * S_len * D
    col = tl.arange(0, D)
    out_ptr = out_s + seq_idx_q[:, None] * D + col[None, :]
    tl.store(out_ptr, O_bf16, mask=mask_m[:, None])
    
    lse_s = LSE + b * H * S_len + h * S_len
    lse_ptr = lse_s + seq_idx_q
    tl.store(lse_ptr, lse, mask=mask_m)


BLOCK_M = 128
BLOCK_N = 64


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S_len, D = Q.shape
    grid = (triton.cdiv(S_len, BLOCK_M), H, B)
    
    sqrt_d = math.sqrt(D)
    
    _mha_kernel_forward[grid](
        Q, K, V, O, LSE,
        B, H, S_len, sqrt_d,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, D=D,
        num_warps=4, num_stages=4
    )