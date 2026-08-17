import torch
import triton
import triton.language as tl


@triton.jit
def _mha_fwd(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    S_len, H, B,
    stride_b, stride_h, stride_s, stride_d,
    lse_stride_b, lse_stride_h,
    SCALE,
    D_half: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    b_idx = tl.program_id(2)
    h_idx = tl.program_id(1)
    q_idx = tl.program_id(0)
    seq_q = q_idx * BLOCK_Q
    
    row = tl.arange(0, 64)
    col = tl.arange(0, D_half)
    q_row = seq_q + row
    
    q_ptr_0 = Q_ptr + b_idx * stride_b + h_idx * stride_h + \
              q_row[:, None] * stride_s + col[None, :] * stride_d
    Q_0 = tl.load(q_ptr_0, mask=(q_row[:, None] < S_len), other=0.0)
    
    q_ptr_1 = Q_ptr + b_idx * stride_b + h_idx * stride_h + \
              q_row[:, None] * stride_s + (col[None, :] + D_half) * stride_d
    Q_1 = tl.load(q_ptr_1, mask=(q_row[:, None] < S_len), other=0.0)
    
    old_max = tl.full((64,), -float('inf'), dtype=tl.float32)
    old_l = tl.zeros((64,), dtype=tl.float32)
    
    O_acc_0 = tl.zeros((64, D_half), dtype=tl.float32)
    O_acc_1 = tl.zeros((64, D_half), dtype=tl.float32)
    
    for k_idx in range(q_idx + 1):
        seq_k = k_idx * BLOCK_K
        k_row = seq_k + row
        
        k_ptr_0 = K_ptr + b_idx * stride_b + h_idx * stride_h + \
                  k_row[:, None] * stride_s + col[None, :] * stride_d
        K_0 = tl.load(k_ptr_0, mask=(k_row[:, None] < S_len), other=0.0)
        
        k_ptr_1 = K_ptr + b_idx * stride_b + h_idx * stride_h + \
                  k_row[:, None] * stride_s + (col[None, :] + D_half) * stride_d
        K_1 = tl.load(k_ptr_1, mask=(k_row[:, None] < S_len), other=0.0)
        
        v_ptr_0 = V_ptr + b_idx * stride_b + h_idx * stride_h + \
                  k_row[:, None] * stride_s + col[None, :] * stride_d
        V_0 = tl.load(v_ptr_0, mask=(k_row[:, None] < S_len), other=0.0)
        
        v_ptr_1 = V_ptr + b_idx * stride_b + h_idx * stride_h + \
                  k_row[:, None] * stride_s + (col[None, :] + D_half) * stride_d
        V_1 = tl.load(v_ptr_1, mask=(k_row[:, None] < S_len), other=0.0)
        
        S = (tl.dot(Q_0, K_0.T) + tl.dot(Q_1, K_1.T)) * SCALE
        
        q_seq = seq_q + row
        k_seq = seq_k + row
        mask = (k_seq[None, :] <= q_seq[:, None]) & (k_seq[None, :] < S_len)
        
        S = tl.where(mask, S, -float('inf'))
        
        m = tl.max(S, axis=1)
        
        new_max = tl.maximum(old_max, m)
        corr = tl.exp(old_max - new_max)
        
        exp_S = tl.exp(S - new_max[:, None])
        sum_exp = tl.sum(exp_S, axis=1)
        
        old_l = old_l * corr + sum_exp
        old_max = new_max
        
        P = exp_S / old_l[:, None]
        
        O_acc_0 = tl.dot(P.to(tl.bfloat16), V_0, O_acc_0)
        O_acc_1 = tl.dot(P.to(tl.bfloat16), V_1, O_acc_1)
        
    lse_out = old_max + tl.log(old_l)
    lse_ptr = LSE_ptr + b_idx * lse_stride_b + h_idx * lse_stride_h + q_row
    tl.store(lse_ptr, lse_out, mask=(q_row < S_len))
    
    for block_n in range(2):
        D_dim = col + D_half * block_n
        o_ptr = O_ptr + b_idx * stride_b + h_idx * stride_h + \
                q_row[:, None] * stride_s + D_dim[None, :] * stride_d
        out_val = O_acc_0 if block_n == 0 else O_acc_1
        tl.store(o_ptr, out_val.to(tl.bfloat16), mask=(q_row[:, None] < S_len))


def run(Q, K, V, O, LSE):
    """Compute causal multi-head attention forward returning O and LSE."""
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    
    grid = (triton.cdiv(S, 64), H, B)
    
    stride_b = H * S * D
    stride_h = S * D
    stride_s = D
    stride_d = 1
    
    lse_stride_b = H * S
    lse_stride_h = S
    
    scale = 1.0 / (D ** 0.5)
    D_half = D // 2
    
    _mha_fwd[grid](
        Q, K, V, O, LSE,
        S, H, B,
        stride_b, stride_h, stride_s, stride_d,
        lse_stride_b, lse_stride_h,
        scale,
        D_half=D_half,
        BLOCK_Q=64,
        BLOCK_K=64,
        num_stages=3,
    )