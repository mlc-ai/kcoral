import torch
import triton
import triton.language as tl


@triton.jit
def _mha_fwd(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    S_len, H,
    stride_b, stride_h, stride_s, stride_d,
    SCALE,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    b_idx = tl.program_id(2)
    h_idx = tl.program_id(1)
    q_idx = tl.program_id(0)
    seq_q = q_idx * BLOCK_Q
    
    # Precompute the Q tile coordinates covering the entire feature dimension
    q_row = seq_q + tl.arange(0, BLOCK_Q)
    d_col = tl.arange(0, 128)
    q_ptr = Q_ptr + b_idx * stride_b + h_idx * stride_h + \
            q_row[:, None] * stride_s + d_col[None, :] * stride_d
    Q_tile = tl.load(q_ptr, mask=(q_row[:, None] < S_len), other=0.0)
    
    old_max = tl.full((BLOCK_Q,), -float('inf'), dtype=tl.float32)
    old_l = tl.zeros((BLOCK_Q,), dtype=tl.float32)
    O_acc = tl.zeros((BLOCK_Q, 128), dtype=tl.float32)
    
    num_k_tiles = q_idx + 1
    for k_idx in range(num_k_tiles):
        seq_k = k_idx * BLOCK_K
        k_row = seq_k + tl.arange(0, BLOCK_K)
        
        k_ptr = K_ptr + b_idx * stride_b + h_idx * stride_h + \
                k_row[:, None] * stride_s + d_col[None, :] * stride_d
        K_tile = tl.load(k_ptr, mask=(k_row[:, None] < S_len), other=0.0)
        
        v_ptr = V_ptr + b_idx * stride_b + h_idx * stride_h + \
                k_row[:, None] * stride_s + d_col[None, :] * stride_d
        V_tile = tl.load(v_ptr, mask=(k_row[:, None] < S_len), other=0.0)
        
        S = tl.dot(Q_tile, K_tile.T) * SCALE
        
        row = tl.arange(0, BLOCK_Q)
        col = tl.arange(0, BLOCK_K)
        mask = (seq_k + col) <= (seq_q + row)
        mask = mask & (seq_k + col < S_len)
        S = tl.where(mask, S, -float('inf'))
        
        m_new = tl.max(S, axis=1)
        new_max = tl.maximum(old_max, m_new)
        corr = tl.exp(old_max - new_max)
        
        exp_S = tl.exp(S - new_max[:, None])
        sum_exp = tl.sum(exp_S, axis=1)
        
        old_l = old_l * corr + sum_exp
        old_max = new_max
        
        P = exp_S / old_l[:, None]
        
        O_acc = tl.dot(P.to(tl.bfloat16), V_tile, O_acc)
    
    lse_out = old_max + old_l * 1.4426950408889634
    
    q_row_seq = seq_q + tl.arange(0, BLOCK_Q)
    lse_ptr = LSE_ptr + b_idx * (h_idx * S_len + q_row_seq)
    tl.store(lse_ptr, lse_out, mask=(q_row_seq < S_len))
    
    out_ptr = O_ptr + b_idx * stride_b + h_idx * stride_h + \
              q_row[:, None] * stride_s + d_col[None, :] * stride_d
    tl.store(out_ptr, O_acc.to(tl.bfloat16), mask=(q_row[:, None] < S_len))


def run(Q, K, V, O, LSE):
    """Compute causal multi-head attention forward returning O and LSE."""
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    
    grid = (triton.cdiv(S, 64), H, B)
    
    stride_b = H * S * D
    stride_h = S * D
    stride_s = D
    stride_d = 1
    
    scale = 1.0 / (D ** 0.5)
    
    _mha_fwd[grid](
        Q, K, V, O, LSE,
        S, H,
        stride_b, stride_h, stride_s, stride_d,
        scale,
        BLOCK_Q=64,
        BLOCK_K=64,
        num_stages=3,
    )