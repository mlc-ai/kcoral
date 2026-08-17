import torch
import triton
import triton.language as tl


@triton.jit
def _mha_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    seq_len, scale,
    stride_b, stride_h, stride_s,
    lse_stride_b, lse_stride_h,
    H: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    batch_head_idx = tl.program_id(0)
    i = tl.program_id(1)
    start_row = i * BLOCK_M
    
    if start_row >= seq_len:
        return
    
    b = batch_head_idx // H
    h = batch_head_idx % H
    
    rows_m = tl.arange(0, BLOCK_M)
    cols_d = tl.arange(0, 128)
    rows_n = tl.arange(0, BLOCK_N)
    
    q_base = Q_ptr + b * stride_b + h * stride_h + start_row * stride_s
    o_base = O_ptr + b * stride_b + h * stride_h + start_row * stride_s
    
    q_ptrs = q_base + rows_m[:, None] * stride_s + cols_d[None, :]
    Q = tl.load(q_ptrs, mask=(start_row + rows_m[:, None] < seq_len), other=0.0)
    Q = Q.to(tl.float32)
    
    O_acc = tl.zeros((BLOCK_M, 128), dtype=tl.float32)
    m = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    l = tl.zeros((BLOCK_M,), dtype=tl.float32)
    
    for j in range(0, i + 1):
        k_base = K_ptr + b * stride_b + h * stride_h + j * BLOCK_N * stride_s
        v_base = V_ptr + b * stride_b + h * stride_h + j * BLOCK_N * stride_s
        
        k_ptrs = k_base + rows_n[:, None] * stride_s + cols_d[None, :]
        K_j = tl.load(k_ptrs, mask=(j * BLOCK_N + rows_n[:, None] < seq_len), other=0.0)
        K_j = K_j.to(tl.float32)
        
        v_ptrs = v_base + rows_n[:, None] * stride_s + cols_d[None, :]
        V_j = tl.load(v_ptrs, mask=(j * BLOCK_N + rows_n[:, None] < seq_len), other=0.0)
        V_j = V_j.to(tl.float32)
        
        S_acc = tl.dot(Q, K_j.T)
        S_acc = S_acc * scale
        
        global_row = start_row + rows_m
        global_col = j * BLOCK_N + rows_n
        mask = (global_col[None, :] <= global_row[:, None]) & (global_col[None, :] < seq_len)
        S_acc = tl.where(mask, S_acc, -float('inf'))
        
        m_old = m
        rowmax = tl.max(S_acc, axis=1)
        m = tl.maximum(m, rowmax)
        exp_old = tl.exp(m_old - m)
        
        P = tl.exp(S_acc - m[:, None])
        l = l * exp_old + tl.sum(P, axis=1)
        
        O_acc = O_acc * exp_old[:, None] + tl.dot(P, V_j)
        
    valid = m > -float('inf')
    inv_l = 1.0 / l
    inv_l = tl.where(valid, inv_l, 0.0)
    O_acc *= inv_l[:, None]
    
    o_ptrs = o_base + rows_m[:, None] * stride_s + cols_d[None, :]
    tl.store(o_ptrs, O_acc.to(tl.bfloat16), mask=(start_row + rows_m[:, None] < seq_len))
    
    lse_base = LSE_ptr + b * lse_stride_b + h * lse_stride_h + start_row
    lse_ptrs = lse_base + rows_m
    lse_val = m + tl.log(l)
    lse_val = tl.where(valid, lse_val, 0.0)
    tl.store(lse_ptrs, lse_val, mask=(start_row + rows_m < seq_len))


def run(Q, K, V, O, LSE):
    """Compute Causal Multi-Head Attention forward pass and Log Sum Exp."""
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    seq_len = S
    
    num_batches = B * H
    num_tiles = triton.cdiv(seq_len, 128)
    
    grid = (num_batches, num_tiles)
    
    scale = 1.0 / float(D ** 0.5)
    
    _mha_fwd_kernel[grid](
        Q, K, V, O, LSE,
        seq_len, scale,
        stride_b=H*S*D, stride_h=S*D, stride_s=D,
        lse_stride_b=H*S, lse_stride_h=S,
        H=H,
        BLOCK_M=128,
        BLOCK_N=128,
        num_warps=4,
        num_stages=3,
    )