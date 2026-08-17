import torch
import triton
import triton.language as tl


@triton.jit
def _mha_fwd_causal(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    S, H, B, D,
    SCALE,
    BLOCK_Q: tl.constexpr,
):
    i = tl.program_id(0)
    head_idx = tl.program_id(1)
    batch_idx = tl.program_id(2)
    
    # Compute base address offset for the specific batch and head
    base_q = batch_idx * H * S * D + head_idx * S * D
    batch_offset_lse = batch_idx * H * S + head_idx * S
    
    start_row = i * BLOCK_Q
    seq_offsets = start_row + tl.arange(0, BLOCK_Q)
    
    # Column indices for 128-dim tensors (split into two 64-dim blocks)
    col_q0 = tl.arange(0, 64)
    col_q1 = 64 + tl.arange(0, 64)
    
    # Load Q with explicit boundary mask
    mask_q = (seq_offsets[:, None] < S)
    
    ptrs_q0 = Q_ptr + base_q + seq_offsets[:, None] * D + col_q0[None, :]
    Q_0 = tl.load(ptrs_q0, mask=mask_q, other=0.0)
    
    ptrs_q1 = Q_ptr + base_q + seq_offsets[:, None] * D + col_q1[None, :]
    Q_1 = tl.load(ptrs_q1, mask=mask_q, other=0.0)
    
    O_acc_0 = tl.zeros((BLOCK_Q, 64), dtype=tl.float32)
    O_acc_1 = tl.zeros((BLOCK_Q, 64), dtype=tl.float32)
    
    m = tl.full((BLOCK_Q,), -float('inf'), dtype=tl.float32)
    l = tl.full((BLOCK_Q,), 0.0, dtype=tl.float32)
    
    ONLINE_SOFTMAX_NINF = -1.0e20
    
    for j in range(i + 1):
        k_seq_offsets = j * 64 + tl.arange(0, 64)
        mask_k = (k_seq_offsets[:, None] < S)
        
        ptrs_k0 = K_ptr + base_q + k_seq_offsets[:, None] * D + col_q0[None, :]
        K_0 = tl.load(ptrs_k0, mask=mask_k, other=0.0)
        
        ptrs_k1 = K_ptr + base_q + k_seq_offsets[:, None] * D + col_q1[None, :]
        K_1 = tl.load(ptrs_k1, mask=mask_k, other=0.0)
        
        ptrs_v0 = V_ptr + base_q + k_seq_offsets[:, None] * D + col_q0[None, :]
        V_0 = tl.load(ptrs_v0, mask=mask_k, other=0.0)
        
        ptrs_v1 = V_ptr + base_q + k_seq_offsets[:, None] * D + col_q1[None, :]
        V_1 = tl.load(ptrs_v1, mask=mask_k, other=0.0)
        
        acc = tl.dot(Q_0, K_0.T) + tl.dot(Q_1, K_1.T)
        acc = acc * SCALE
        
        if j == i:
            row = tl.arange(0, 64)
            col = tl.arange(0, 64)
            causal_mask = (row[:, None] >= col[None, :])
            acc = tl.where(causal_mask, acc, ONLINE_SOFTMAX_NINF)
        
        m_old = m
        m_curr = tl.max(acc, axis=1)
        m = tl.maximum(m_old, m_curr)
        
        exp_scale = tl.exp(m_old - m)
        P = tl.exp(acc - m)
        
        l_curr = tl.sum(P, axis=1)
        l = l * exp_scale + l_curr
        
        O_acc_0 = O_acc_0 * exp_scale[:, None]
        O_acc_1 = O_acc_1 * exp_scale[:, None]
        
        O_acc_0 = O_acc_0 + tl.dot(P, V_0)
        O_acc_1 = O_acc_1 + tl.dot(P, V_1)
        
    inv_l = 1.0 / l
    O_0 = (O_acc_0 * inv_l[:, None]).to(tl.bfloat16)
    O_1 = (O_acc_1 * inv_l[:, None]).to(tl.bfloat16)
    
    L_val = m + tl.log(l)
    
    mask_o = (seq_offsets[:, None] < S)
    out_ptrs_0 = O_ptr + base_q + seq_offsets[:, None] * D + col_q0[None, :]
    tl.store(out_ptrs_0, O_0, mask=mask_o)
    
    out_ptrs_1 = O_ptr + base_q + seq_offsets[:, None] * D + col_q1[None, :]
    tl.store(out_ptrs_1, O_1, mask=mask_o)
    
    lse_ptr = LSE_ptr + batch_offset_lse + seq_offsets
    tl.store(lse_ptr, L_val, mask=seq_offsets < S)


def run(Q, K, V, O, LSE):
    """Compute Causal MHA Forward Pass with LSE output."""
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    scale = 1.0 / (D ** 0.5)
    
    grid = (triton.cdiv(S, 64), H, B)
    
    _mha_fwd_causal[grid](
        Q, K, V, O, LSE,
        S, H, B, D,
        scale,
        BLOCK_Q=64,
        num_warps=4,
        num_stages=3,
    )