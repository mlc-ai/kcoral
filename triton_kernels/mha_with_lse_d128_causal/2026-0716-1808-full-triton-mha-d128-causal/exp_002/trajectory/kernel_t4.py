import torch
import triton
import triton.language as tl


@triton.jit
def _attention_forward_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    lse_ptr,
    S,
    HEAD_DIM: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)
    
    b_h = pid_b * 48 + pid_h
    BLOCK_N = BLOCK_S
    
    row_idx = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    valid_rows = row_idx < S
    
    out_acc = tl.zeros([BLOCK_S, BLOCK_N], dtype=tl.float32)
    
    m_i = tl.full([BLOCK_S], -float("inf"), dtype=tl.float32)
    D_i = tl.full([BLOCK_S], 0.0, dtype=tl.float32)
    
    scale = 1.0 / (HEAD_DIM ** 0.5)
    
    for k_tile in range(0, min(pid_s + 1, (S + BLOCK_S - 1) // BLOCK_S)):
        k_row = k_tile * BLOCK_S + tl.arange(0, BLOCK_S)
        
        m_i_k = tl.full([BLOCK_S], -float("inf"), dtype=tl.float32)
        D_i_k = tl.full([BLOCK_S], 0.0, dtype=tl.float32)
        acc_k = tl.zeros([BLOCK_S, BLOCK_N], dtype=tl.float32)
        
        for chunk in tl.range(0, HEAD_DIM, BLOCK_K, unroll_amount=2, num_stages=2):
            q = tl.load(q_ptr + b_h * S * HEAD_DIM + row_idx[:, None] * HEAD_DIM + tl.arange(0, BLOCK_K)[None, :] + chunk, other=0.0)
            k = tl.load(k_ptr + b_h * S * HEAD_DIM + k_row[:, None] * HEAD_DIM + tl.arange(0, BLOCK_K)[None, :] + chunk, other=0.0)
            
            p_chunk = tl.dot(q, k.T)
            
            col_idx = k_tile * BLOCK_S + tl.arange(0, BLOCK_S)
            mask = (row_idx[:, None] >= col_idx[None, :]) & (col_idx[None, :] < S)
            p_chunk = tl.where(mask, p_chunk * scale, -float('inf'))
            
            v = tl.load(v_ptr + b_h * S * HEAD_DIM + k_row[:, None] * HEAD_DIM + tl.arange(0, BLOCK_K)[None, :] + chunk, other=0.0)
            
            acc_k += tl.dot(p_chunk, v)
            
        curr_max_k = tl.maximum(m_i_k, tl.max(acc_k, axis=1))
        exp_diff_k = tl.exp(m_i_k - curr_max_k)
        D_i_k = D_i_k * exp_diff_k
        D_i_k += tl.sum(tl.exp(acc_k - curr_max_k[:, None]), axis=1)
        m_i_k = curr_max_k
        
        out_acc *= exp_diff_k[:, None, None]
        out_acc += acc_k
        
        curr_max = tl.maximum(m_i, m_i_k)
        exp_diff = tl.exp(m_i - curr_max)
        D_i = D_i * exp_diff
        D_i += D_i_k * exp_diff_k[:, None] * tl.exp(m_i_k - curr_max)
        m_i = curr_max

    final_out = out_acc.to(tl.bfloat16)
    
    c = o_ptr + b_h * S * HEAD_DIM + row_idx[:, None] * HEAD_DIM + tl.arange(0, HEAD_DIM)[None, :]
    
    tl.store(c, final_out, mask=valid_rows[:, None])
    
    lse = m_i + tl.log(D_i)
    
    lse_ptr_bh = lse_ptr + b_h * S
    lse_store = tl.where(valid_rows, lse, 0.0)
    tl.store(lse_ptr_bh + row_idx, lse_store, mask=valid_rows)


def run(Q, K, V, O, LSE):
    """
    Computes causal multi-head attention O and LSE given Q, K, V on the current device.
    Expected shapes:
      Q, K, V : [B, H, S, D]
      O       : [B, H, S, D]
      LSE     : [B, H, S]
    Where B=4, H=48, D=128.
    """
    torch.cuda.set_device(Q.device)
    
    S = Q.shape[2]
    
    block_size = 128
    grid = (
        4,   # B
        48,  # H
        triton.cdiv(S, block_size)
    )
    
    _attention_forward_kernel[grid](
        Q, K, V, O, LSE,
        S,
        HEAD_DIM=128,
        BLOCK_S=block_size,
        BLOCK_K=64,
        num_warps=4,
        num_stages=1,
    )