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
    D,
    stride_q_s,
    stride_q_d,
    stride_k_s,
    stride_k_d,
    stride_v_s,
    stride_v_d,
    stride_o_s,
    stride_o_d,
    stride_lse_s,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    b = tl.program_id(2)
    h = tl.program_id(1)
    start_i = tl.program_id(0)
    
    base_offset = b * H * S_len * D + h * S_len * D
    
    query_rows = tl.arange(0, BLOCK_M)
    q_row_idx = start_i * BLOCK_M + query_rows
    d_indices = tl.arange(0, D)
    
    q_ptrs = Q + base_offset + q_row_idx[:, None] * stride_q_s + d_indices[None, :] * stride_q_d
    Q_tile = tl.load(q_ptrs, mask=(q_row_idx[:, None] < S_len), other=0.0)
    
    m = tl.full((BLOCK_M,), -float('inf'), tl.float32)
    l = tl.full((BLOCK_M,), 0.0, tl.float32)
    O_acc = tl.zeros((BLOCK_M, D), tl.float32)
    
    num_blocks = start_i + 1
    
    for j in range(num_blocks):
        key_cols = tl.arange(0, BLOCK_N)
        k_col_idx = j * BLOCK_N + key_cols
        
        k_ptrs = K + base_offset + k_col_idx[:, None] * stride_k_s + d_indices[None, :] * stride_k_d
        K_tile = tl.load(k_ptrs, mask=(k_col_idx[:, None] < S_len), other=0.0)
        
        v_ptrs = V + base_offset + k_col_idx[:, None] * stride_v_s + d_indices[None, :] * stride_v_d
        V_tile = tl.load(v_ptrs, mask=(k_col_idx[:, None] < S_len), other=0.0)
        
        S = tl.dot(Q_tile, K_tile.T) * scale
        
        is_causal = (q_row_idx[:, None]) >= (k_col_idx[None, :])
        S = tl.where(is_causal, S, -float('inf'))
        
        m_old = m
        m_local = tl.max(S, axis=1)
        m_new = tl.maximum(m_old, m_local)
        m = m_new
        
        P = tl.exp(S - m[:, None])
        
        l_local = tl.sum(P, axis=1)
        l = l * tl.exp(m_old - m) + l_local
        
        O_acc = O_acc * tl.exp(m_old - m)[:, None]
        O_acc = tl.dot(P, V_tile, acc=O_acc)
    
    l_safe = tl.where(l > 0, l, 1.0)
    O_acc_out = (O_acc / l_safe[:, None]).to(tl.bfloat16)
    
    out_ptrs = O + base_offset + q_row_idx[:, None] * stride_o_s + d_indices[None, :] * stride_o_d
    tl.store(out_ptrs, O_acc_out, mask=(q_row_idx[:, None] < S_len))
    
    lse = m + tl.log(l_safe)
    lse_ptrs = LSE + b * (h * S_len) + q_row_idx
    tl.store(lse_ptrs, lse, mask=(q_row_idx < S_len))


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S_len, D = Q.shape
    
    stride_q_s = Q.stride()[2]
    stride_q_d = Q.stride()[3]
    stride_k_s = K.stride()[2]
    stride_k_d = K.stride()[3]
    stride_v_s = V.stride()[2]
    stride_v_d = V.stride()[3]
    stride_o_s = O.stride()[2]
    stride_o_d = O.stride()[3]
    stride_lse_s = LSE.stride()[2]
    
    scale = 1.0 / (D ** 0.5)
    
    BLOCK_M = 64
    BLOCK_N = 64
    
    grid = (triton.cdiv(S_len, BLOCK_M), H, B)
    _attention_kernel[grid](
        Q, K, V, O, LSE,
        S_len, H, D,
        stride_q_s, stride_q_d,
        stride_k_s, stride_k_d,
        stride_v_s, stride_v_d,
        stride_o_s, stride_o_d,
        stride_lse_s,
        scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=2,
    )