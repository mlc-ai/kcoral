import torch
import triton
import triton.language as tl


@triton.jit
def _mha_fwd(
    Q, K, V, O, LSE,
    S_len,
    stride_b, stride_h, stride_s, stride_d,
    SCALE,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    b_idx = tl.program_id(2)
    h_idx = tl.program_id(1)
    q_idx = tl.program_id(0)
    seq_q = q_idx * 64
    
    base_ptr = Q + b_idx * stride_b + h_idx * stride_h
    
    Q_base_ptr = base_ptr
    K_base_ptr = base_ptr
    V_base_ptr = base_ptr
    O_base_ptr = base_ptr
    
    O_local = tl.zeros((2, 2, 32, 64), dtype=tl.bfloat16)
    
    old_max = tl.full((2, 64), -float('inf'), dtype=tl.float32)
    old_l = tl.zeros((2, 64), dtype=tl.float32)
    
    for k_idx in range(0, q_idx + 1):
        seq_k = k_idx * 64
        
        Q_0 = Q[..., seq_q : seq_q + 64, 0:64]
        Q_1 = Q[..., seq_q : seq_q + 64, 64:128]
        Q_local = tl.stack([Q_0, Q_1])
        Q_local = Q_local.reshape((2, 2, 32, 64))
        
        K_0 = K[..., seq_k : seq_k + 64, 0:64]
        K_1 = K[..., seq_k : seq_k + 64, 64:128]
        K_local = tl.stack([K_0, K_1])
        K_local = K_local.reshape((2, 2, 32, 64))
        
        S_local = Q_local[:, None, :, :] * K_local[None, :, :, :]
        S_local = S_local.sum(axis=-1)
        
        row = tl.arange(0, 64)
        col = tl.arange(0, 64)
        mask = (col <= row) & (col < S_len)
        S_local = S_local * mask * SCALE
        
        probs = tl.zeros((2, 2, 32), dtype=tl.bfloat16)
        for i in range(2):
            s = S_local[i, i, :]
            m_i = tl.max(s, axis=-1)
            old_max_i = old_max[i, :]
            new_max_i = tl.maximum(old_max_i, m_i)
            
            old_l_i = old_l[i, :]
            corr_i = tl.exp(old_max_i - new_max_i)
            new_l_i = old_l_i * corr_i + tl.sum(tl.exp(s - new_max_i), axis=-1)
            
            P_i = tl.exp(s - new_max_i) / new_l_i
            probs[i, i, :] = P_i
            
            old_max[i, :] = new_max_i
            old_l[i, :] = new_l_i
        
        for block_n in range(2):
            V_local = V[..., seq_k : seq_k + 64, block_n*64:(block_n+1)*64]
            V_local = V_local.reshape((2, 2, 32, 64))
            O_local[:, None, :, :] += S_local[:, :, :, None] * V_local[None, :, :, :]
            
    for i in range(2):
        lse_out = old_max[i, :] + old_l[i, :] * 1.4426950408889634
        lse_ptr = LSE + b_idx * (h_idx * S_len + seq_q) + tl.arange(0, 64)
        tl.store(lse_ptr, lse_out, mask=(seq_q + tl.arange(0, 64)) < S_len)
        
    for block_n in range(2):
        out_ptr = O_base_ptr + (seq_q + tl.arange(0, 64)) * stride_s + block_n * 64 + tl.arange(0, 64)
        out_val = O_local[:, block_n, :, :]
        mask = (seq_q + tl.arange(0, 64))[:, None] < S_len
        tl.store(out_ptr, out_val, mask=mask)


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
        S,
        stride_b, stride_h, stride_s, stride_d,
        scale,
        BLOCK_Q=64,
        BLOCK_K=64,
        num_stages=3,
    )