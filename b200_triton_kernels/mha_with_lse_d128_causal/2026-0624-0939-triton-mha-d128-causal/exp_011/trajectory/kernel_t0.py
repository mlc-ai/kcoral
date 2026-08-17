import torch
import triton
import triton.language as tl


@triton.jit
def _mha_fwd_causal(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    S, H, B,
    stride_Q2, stride_Q3,
    stride_K2, stride_K3,
    stride_V2, stride_V3,
    stride_O2, stride_O3,
    stride_LSE1, stride_LSE2,
    SCALE,
    BLOCK_Q: tl.constexpr,
):
    """
    Causal Multi-Head Attention Forward Kernel.
    Computes O = softmax(Q @ K^T / sqrt(D)) @ V and LSE = m + log(l)
    for a single (batch, head) pair and a single query tile.
    """
    i = tl.program_id(0)
    head_idx = tl.program_id(1)
    batch_idx = tl.program_id(2)
    
    start_row = i * BLOCK_Q
    seq_offsets = start_row + tl.arange(0, BLOCK_Q)
    
    base_s = S * stride_Q2
    batch_offset = (batch_idx * H + head_idx) * base_s
    seq_idx = batch_offset + seq_offsets * stride_Q2
    
    col_q0 = tl.arange(0, 64)
    col_q1 = 64 + tl.arange(0, 64)
    
    ptrs_q0 = Q_ptr + seq_idx[:, None] * stride_Q2 + col_q0[None, :] * stride_Q3
    Q_0 = tl.load(ptrs_q0)
    
    ptrs_q1 = Q_ptr + seq_idx[:, None] * stride_Q2 + col_q1[None, :] * stride_Q3
    Q_1 = tl.load(ptrs_q1)
    
    batch_offset_lse = batch_idx * H * stride_LSE1 + head_idx * stride_LSE1
    lse_ptr = LSE_ptr + batch_offset_lse + seq_offsets * stride_LSE2
    
    O_acc_0 = tl.zeros((BLOCK_Q, 64), dtype=tl.float32)
    O_acc_1 = tl.zeros((BLOCK_Q, 64), dtype=tl.float32)
    
    m = tl.full((BLOCK_Q,), -float('inf'), dtype=tl.float32)
    l = tl.full((BLOCK_Q,), 0.0, dtype=tl.float32)
    
    ONLINE_SOFTMAX_NINF = -1.0e20
    
    for j in range(i + 1):
        k_seq_offsets = j * 64 + tl.arange(0, 64)
        k_seq_idx = batch_offset + k_seq_offsets * stride_K2
        
        col_k0 = tl.arange(0, 64)
        col_k1 = 64 + tl.arange(0, 64)
        
        ptrs_k0 = K_ptr + k_seq_idx[:, None] * stride_K2 + col_k0[None, :] * stride_K3
        K_0 = tl.load(ptrs_k0, mask=k_seq_offsets[:, None] < S, other=0.0)
        
        ptrs_k1 = K_ptr + k_seq_idx[:, None] * stride_K2 + col_k1[None, :] * stride_K3
        K_1 = tl.load(ptrs_k1, mask=k_seq_offsets[:, None] < S, other=0.0)
        
        col_v0 = tl.arange(0, 64)
        col_v1 = 64 + tl.arange(0, 64)
        
        ptrs_v0 = V_ptr + k_seq_idx[:, None] * stride_V2 + col_v0[None, :] * stride_V3
        V_0 = tl.load(ptrs_v0, mask=k_seq_offsets[:, None] < S, other=0.0)
        
        ptrs_v1 = V_ptr + k_seq_idx[:, None] * stride_V2 + col_v1[None, :] * stride_V3
        V_1 = tl.load(ptrs_v1, mask=k_seq_offsets[:, None] < S, other=0.0)
        
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
    
    out_ptrs_0 = O_ptr + seq_idx[:, None] * stride_O2 + col_q0[None, :] * stride_O3
    tl.store(out_ptrs_0, O_0, mask=seq_offsets[:, None] < S)
    
    out_ptrs_1 = O_ptr + seq_idx[:, None] * stride_O2 + col_q1[None, :] * stride_O3
    tl.store(out_ptrs_1, O_1, mask=seq_offsets[:, None] < S)
    
    tl.store(lse_ptr, L_val, mask=seq_offsets < S)


def run(Q, K, V, O, LSE):
    """Compute Causal MHA Forward Pass with LSE output."""
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    stride_Q2 = Q.stride(2)
    stride_Q3 = Q.stride(3)
    stride_K2 = K.stride(2)
    stride_K3 = K.stride(3)
    stride_V2 = V.stride(2)
    stride_V3 = V.stride(3)
    stride_O2 = O.stride(2)
    stride_O3 = O.stride(3)
    stride_LSE1 = LSE.stride(1)
    stride_LSE2 = LSE.stride(2)
    
    scale = 1.0 / (D ** 0.5)
    
    grid = (triton.cdiv(S, 64), H, B)
    
    _mha_fwd_causal[grid](
        Q, K, V, O, LSE,
        S, H, B,
        stride_Q2, stride_Q3,
        stride_K2, stride_K3,
        stride_V2, stride_V3,
        stride_O2, stride_O3,
        stride_LSE1, stride_LSE2,
        scale,
        BLOCK_Q=64,
        num_warps=4,
        num_stages=3,
    )