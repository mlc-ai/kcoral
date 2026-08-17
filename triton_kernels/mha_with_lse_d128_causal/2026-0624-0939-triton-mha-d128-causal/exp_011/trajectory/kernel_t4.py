import torch
import triton
import triton.language as tl


@triton.jit
def _mha_fwd_causal(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    S, H, B,
    stride_S, stride_LSE_S,
    SCALE,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    D: tl.constexpr = 128,
):
    i = tl.program_id(0)
    head_idx = tl.program_id(1)
    batch_idx = tl.program_id(2)
    
    base_offset = batch_idx * H * stride_S + head_idx * stride_S
    base_offset_lse = batch_idx * H * stride_LSE_S + head_idx * stride_LSE_S
    
    start_row = i * BLOCK_Q
    seq_offsets = start_row + tl.arange(0, BLOCK_Q)
    
    col_d0 = tl.arange(0, D//2)
    col_d1 = tl.arange(D//2, D)
    
    mask_q = (seq_offsets < S)[:, None]
    Q_0 = tl.load(Q_ptr + base_offset + seq_offsets[:, None] * D + col_d0[None, :], mask=mask_q, other=0.0)
    Q_1 = tl.load(Q_ptr + base_offset + seq_offsets[:, None] * D + col_d1[None, :], mask=mask_q, other=0.0)
    
    O_acc_0 = tl.zeros((BLOCK_Q, D//2), dtype=tl.float32)
    O_acc_1 = tl.zeros((BLOCK_Q, D//2), dtype=tl.float32)
    
    m = tl.full((BLOCK_Q,), -float('inf'), dtype=tl.float32)
    l = tl.full((BLOCK_Q,), 0.0, dtype=tl.float32)
    
    q_idx = start_row + tl.arange(0, BLOCK_Q)
    
    for j in range(i + 1):
        k_seq_offsets = j * BLOCK_K + tl.arange(0, BLOCK_K)
        
        mask_k = (k_seq_offsets < S)[:, None]
        K_0 = tl.load(K_ptr + base_offset + k_seq_offsets[:, None] * D + col_d0[None, :], mask=mask_k, other=0.0)
        K_1 = tl.load(K_ptr + base_offset + k_seq_offsets[:, None] * D + col_d1[None, :], mask=mask_k, other=0.0)
        V_0 = tl.load(V_ptr + base_offset + k_seq_offsets[:, None] * D + col_d0[None, :], mask=mask_k, other=0.0).to(tl.float32)
        V_1 = tl.load(V_ptr + base_offset + k_seq_offsets[:, None] * D + col_d1[None, :], mask=mask_k, other=0.0).to(tl.float32)
        
        acc = (tl.dot(Q_0, K_0.T) + tl.dot(Q_1, K_1.T)).to(tl.float32)
        acc = acc * SCALE
        
        valid_mask = (seq_offsets[:, None] < S) & (k_seq_offsets[None, :] < S)
        
        if j == i:
            k_idx = j * BLOCK_K + tl.arange(0, BLOCK_K)
            causal_mask = (q_idx[:, None] >= k_idx[None, :])
            acc = tl.where(causal_mask & valid_mask, acc, -1.0e20)
        else:
            acc = tl.where(valid_mask, acc, -1.0e20)
        
        m_old = m
        m_curr = tl.max(acc, axis=1)
        m = tl.maximum(m_old, m_curr)
        
        exp_scale = tl.exp(m_old - m)
        P = tl.exp(acc - m)
        
        l_curr = tl.sum(P, axis=1)
        l = l * exp_scale + l_curr
        
        O_acc_0 = O_acc_0 * exp_scale[:, None] + tl.dot(P, V_0)
        O_acc_1 = O_acc_1 * exp_scale[:, None] + tl.dot(P, V_1)
        
    inv_l = 1.0 / l
    O_0 = (O_acc_0 * inv_l[:, None]).to(tl.bfloat16)
    O_1 = (O_acc_1 * inv_l[:, None]).to(tl.bfloat16)
    
    L_val = m + tl.log(l)
    
    mask_o = (seq_offsets < S)[:, None]
    out_ptrs_0 = O_ptr + base_offset + seq_offsets[:, None] * D + col_d0[None, :]
    tl.store(out_ptrs_0, O_0, mask=mask_o)
    
    out_ptrs_1 = O_ptr + base_offset + seq_offsets[:, None] * D + col_d1[None, :]
    tl.store(out_ptrs_1, O_1, mask=mask_o)
    
    lse_ptr = LSE_ptr + base_offset_lse + seq_offsets
    tl.store(lse_ptr, L_val, mask=(seq_offsets < S))


def run(Q, K, V, O, LSE):
    """Compute Causal MHA Forward Pass with LSE output."""
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    stride_S = S * D
    stride_LSE_S = S
    
    scale = 1.0 / (D ** 0.5)
    
    grid = (triton.cdiv(S, 64), H, B)
    
    _mha_fwd_causal[grid](
        Q, K, V, O, LSE,
        S, H, B,
        stride_S, stride_LSE_S,
        scale,
        BLOCK_Q=64,
        BLOCK_K=64,
        num_warps=4,
        num_stages=3,
    )