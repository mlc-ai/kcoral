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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_q_blk = tl.program_id(2)
    
    b_h = pid_b * 48 + pid_h
    
    q_row_start = pid_q_blk * BLOCK_M
    q_idx = q_row_start + tl.arange(0, BLOCK_M)
    valid_rows = q_idx < S
    
    stride_q_bh = S * HEAD_DIM
    stride_k_bh = S * HEAD_DIM
    stride_v_bh = S * HEAD_DIM
    stride_o_bh = S * HEAD_DIM
    
    out_acc_h0 = tl.zeros([BLOCK_M, 64], dtype=tl.float32)
    out_acc_h1 = tl.zeros([BLOCK_M, 64], dtype=tl.float32)
    
    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    D_i = tl.full([BLOCK_M], 0.0, dtype=tl.float32)
    
    q_h0 = tl.load(q_ptr + b_h * stride_q_bh + q_idx[:, None] * HEAD_DIM + tl.arange(0, 64)[None, :], mask=(q_idx[:, None] < S), other=0.0)
    q_h1 = tl.load(q_ptr + b_h * stride_q_bh + q_idx[:, None] * HEAD_DIM + 64 + tl.arange(0, 64)[None, :], mask=(q_idx[:, None] < S), other=0.0)
    
    max_k_tile = min((q_row_start + BLOCK_M + BLOCK_N - 1) // BLOCK_N, (S + BLOCK_N - 1) // BLOCK_N)
    
    for k_row_start in range(0, max_k_tile * BLOCK_N, BLOCK_N):
        k_idx = k_row_start + tl.arange(0, BLOCK_N)
        
        k_h0 = tl.load(k_ptr + b_h * stride_k_bh + k_idx[:, None] * HEAD_DIM + tl.arange(0, 64)[None, :], mask=(k_idx[:, None] < S), other=0.0)
        k_h1 = tl.load(k_ptr + b_h * stride_k_bh + k_idx[:, None] * HEAD_DIM + 64 + tl.arange(0, 64)[None, :], mask=(k_idx[:, None] < S), other=0.0)
        
        v_h0 = tl.load(v_ptr + b_h * stride_v_bh + k_idx[:, None] * HEAD_DIM + tl.arange(0, 64)[None, :], mask=(k_idx[:, None] < S), other=0.0)
        v_h1 = tl.load(v_ptr + b_h * stride_v_bh + k_idx[:, None] * HEAD_DIM + 64 + tl.arange(0, 64)[None, :], mask=(k_idx[:, None] < S), other=0.0)
        
        p = tl.dot(q_h0, k_h0.T)
        p = tl.dot(q_h1, k_h1.T, acc=p)
        
        p = p * (1.0 / (HEAD_DIM ** 0.5))
        
        mask = (q_idx[:, None] >= k_idx[None, :]) & (k_idx[None, :] < S)
        p = tl.where(mask, p, -float('inf'))
        
        curr_max = tl.maximum(m_i, tl.max(p, axis=1))
        safe_m_i = tl.where(m_i == -float('inf'), curr_max, m_i)
        exp_diff = tl.exp(safe_m_i - curr_max)
        D_i = D_i * exp_diff
        
        out_acc_h0 = out_acc_h0 * exp_diff[:, None]
        out_acc_h1 = out_acc_h1 * exp_diff[:, None]
        
        exp_p = tl.exp(p - curr_max[:, None])
        D_i += tl.sum(exp_p, axis=1)
        
        out_acc_h0 = tl.dot(exp_p.to(tl.bfloat16), v_h0, acc=out_acc_h0)
        out_acc_h1 = tl.dot(exp_p.to(tl.bfloat16), v_h1, acc=out_acc_h1)
        
        m_i = curr_max
        
    final_out_h0 = (out_acc_h0 / D_i[:, None]).to(tl.bfloat16)
    final_out_h1 = (out_acc_h1 / D_i[:, None]).to(tl.bfloat16)
    
    o_ptr_bh = o_ptr + b_h * stride_o_bh
    
    c0 = o_ptr_bh + q_idx[:, None] * HEAD_DIM + tl.arange(0, 64)[None, :]
    c1 = o_ptr_bh + q_idx[:, None] * HEAD_DIM + 64 + tl.arange(0, 64)[None, :]
    
    tl.store(c0, final_out_h0, mask=valid_rows[:, None])
    tl.store(c1, final_out_h1, mask=valid_rows[:, None])
    
    lse = m_i + tl.log(D_i)
    
    lse_ptr_bh = lse_ptr + b_h * S
    lse_store = tl.where(valid_rows, lse, 0.0)
    tl.store(lse_ptr_bh + q_idx, lse_store, mask=valid_rows)


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
    
    block_m = 128
    block_n = 128
    grid = (
        4,   
        48,  
        triton.cdiv(S, block_m)
    )
    
    _attention_forward_kernel[grid](
        Q, K, V, O, LSE,
        S,
        HEAD_DIM=128,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=8,
        num_stages=2,
    )