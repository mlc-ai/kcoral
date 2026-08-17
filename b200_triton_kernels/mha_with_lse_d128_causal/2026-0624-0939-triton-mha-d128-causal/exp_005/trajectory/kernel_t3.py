import torch
import triton
import triton.language as tl


@triton.jit
def _attention_kernel(
    Q,
    K,
    V,
    O,
    LSE,
    S_len,
    H,
    B,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    b = tl.program_id(2)
    h = tl.program_id(1)
    i = tl.program_id(0)
    
    rows = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    
    q_row_idx = i * BLOCK_M + rows
    
    D_half = tl.constexpr(64)
    HEAD_DIM = tl.constexpr(128)
    
    d_indices_0 = tl.arange(0, D_half)
    d_indices_1 = tl.arange(0, D_half) + D_half
    
    base_offset = b * H * S_len * HEAD_DIM + h * S_len * HEAD_DIM
    
    q_ptrs_0 = Q + base_offset + q_row_idx[:, None] * HEAD_DIM + d_indices_0[None, :]
    q_ptrs_1 = Q + base_offset + q_row_idx[:, None] * HEAD_DIM + d_indices_1[None, :]
    
    Q_0 = tl.load(q_ptrs_0, mask=(q_row_idx[:, None] < S_len), other=0.0)
    Q_1 = tl.load(q_ptrs_1, mask=(q_row_idx[:, None] < S_len), other=0.0)
    
    m = tl.full((BLOCK_M,), -float('inf'), tl.float32)
    l = tl.full((BLOCK_M,), 0.0, tl.float32)
    O_acc_0 = tl.zeros((BLOCK_M, D_half), tl.float32)
    O_acc_1 = tl.zeros((BLOCK_M, D_half), tl.float32)
    
    for j in range(i + 1):
        k_col_idx = j * BLOCK_N + cols
        
        k_ptrs_0 = K + base_offset + k_col_idx[:, None] * HEAD_DIM + d_indices_0[None, :]
        k_ptrs_1 = K + base_offset + k_col_idx[:, None] * HEAD_DIM + d_indices_1[None, :]
        
        K_0 = tl.load(k_ptrs_0, mask=(k_col_idx[None, :] < S_len), other=0.0)
        K_1 = tl.load(k_ptrs_1, mask=(k_col_idx[None, :] < S_len), other=0.0)
        
        v_ptrs_0 = V + base_offset + k_col_idx[:, None] * HEAD_DIM + d_indices_0[None, :]
        v_ptrs_1 = V + base_offset + k_col_idx[:, None] * HEAD_DIM + d_indices_1[None, :]
        
        V_0 = tl.load(v_ptrs_0, mask=(k_col_idx[None, :] < S_len), other=0.0)
        V_1 = tl.load(v_ptrs_1, mask=(k_col_idx[None, :] < S_len), other=0.0)
        
        S = (tl.dot(Q_0, K_0.T) + tl.dot(Q_1, K_1.T)) * scale
        
        is_causal = q_row_idx[:, None] >= k_col_idx[None, :]
        S = tl.where(is_causal, S, -float('inf'))
        
        m_old = m
        m_local = tl.max(S, axis=1)
        m = tl.maximum(m_old, m_local)
        
        P = tl.exp(S - m[:, None])
        P = tl.where(m > -float('inf'), P, 0.0)
        
        l_local = tl.sum(P, axis=1)
        l = l * tl.exp(m_old - m) + l_local
        
        O_acc_0 = O_acc_0 * tl.exp(m_old - m)[:, None] + tl.dot(P, V_0)
        O_acc_1 = O_acc_1 * tl.exp(m_old - m)[:, None] + tl.dot(P, V_1)
    
    l_safe = tl.where(l > 0, l, 1.0)
    O_0 = (O_acc_0 / l_safe[:, None]).to(tl.bfloat16)
    O_1 = (O_acc_1 / l_safe[:, None]).to(tl.bfloat16)
    
    base_offset_O = b * H * S_len * HEAD_DIM + h * S_len * HEAD_DIM
    out_ptrs_0 = O + base_offset_O + q_row_idx[:, None] * HEAD_DIM + d_indices_0[None, :]
    out_ptrs_1 = O + base_offset_O + q_row_idx[:, None] * HEAD_DIM + d_indices_1[None, :]
    
    tl.store(out_ptrs_0, O_0, mask=(q_row_idx[:, None] < S_len))
    tl.store(out_ptrs_1, O_1, mask=(q_row_idx[:, None] < S_len))
    
    lse = m + tl.log(l_safe)
    
    lse_base = b * H * S_len + h * S_len
    lse_ptrs = LSE + lse_base + q_row_idx
    tl.store(lse_ptrs, lse, mask=(q_row_idx < S_len))


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S_len, D = Q.shape
    
    scale = 1.0 / (D ** 0.5)
    
    BLOCK_M = 64
    BLOCK_N = 64
    
    grid = (triton.cdiv(S_len, BLOCK_M), H, B)
    _attention_kernel[grid](
        Q, K, V, O, LSE,
        S_len, H, B,
        scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=3,
    )