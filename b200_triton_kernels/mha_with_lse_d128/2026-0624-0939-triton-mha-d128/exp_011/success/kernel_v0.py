import torch
import triton
import triton.language as tl


@triton.jit
def _mha_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    seq_len, head_dim,
    stride_S_Q, stride_D_Q,
    stride_S_K, stride_D_K,
    stride_S_V, stride_D_V,
    stride_S_O, stride_D_O,
    SCALE,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    m_block_idx = tl.program_id(1)
    
    start_m = m_block_idx * BLOCK_M
    num_n_blocks = tl.cdiv(seq_len, BLOCK_N)
    
    base_offset = batch_idx * seq_len * head_dim
    
    q_row_off = start_m + tl.arange(0, BLOCK_M)
    q_col_off = tl.arange(0, HEAD_DIM)
    q_off = q_row_off[:, None] * stride_S_Q + q_col_off[None, :] * stride_D_Q
    q = tl.load(Q_ptr + base_offset + q_off, mask=(q_row_off[:, None] < seq_len), other=0.0)
    
    m = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    l = tl.zeros((BLOCK_M,), dtype=tl.float32)
    O_acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    
    for j in range(num_n_blocks):
        start_n = j * BLOCK_N
        
        k_row_off = start_n + tl.arange(0, BLOCK_N)
        k_col_off = tl.arange(0, HEAD_DIM)
        k_off = k_row_off[:, None] * stride_S_K + k_col_off[None, :] * stride_D_K
        k = tl.load(K_ptr + base_offset + k_off, mask=(k_row_off[:, None] < seq_len), other=0.0)
        
        v_off = k_row_off[:, None] * stride_S_V + k_col_off[None, :] * stride_D_V
        v = tl.load(V_ptr + base_offset + v_off, mask=(k_row_off[:, None] < seq_len), other=0.0)
        
        S = tl.dot(q, k.T)
        S_scaled = S * SCALE
        
        m_old = m
        m_new = tl.maximum(m, tl.max(S_scaled, axis=1))
        
        P = tl.exp(S_scaled - m_new[:, None])
        
        valid_k = k_row_off < seq_len
        P = P * valid_k[None, :]
        
        l = l * tl.exp(m_old - m_new) + tl.sum(P, axis=1)
        m = m_new
        
        exp_scale = tl.exp(m_old - m_new)[:, None]
        O_acc = O_acc * exp_scale
        
        P_bf16 = P.to(tl.bfloat16)
        O_acc = tl.dot(P_bf16, v, O_acc)
        
    inv_l = 1.0 / tl.maximum(l, 1e-30)[:, None]
    O_out = (O_acc * inv_l).to(tl.bfloat16)
    
    o_row_off = start_m + tl.arange(0, BLOCK_M)
    o_col_off = tl.arange(0, HEAD_DIM)
    o_off = o_row_off[:, None] * stride_S_O + o_col_off[None, :] * stride_D_O
    valid_q = o_row_off < seq_len
    tl.store(O_ptr + base_offset + o_off, O_out, mask=valid_q[:, None])
    
    seq_idx = start_m + tl.arange(0, BLOCK_M)
    lse_val = m + tl.log(l)
    tl.store(LSE_ptr + batch_idx * seq_len + seq_idx, lse_val, mask=valid_q)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    if S == 0:
        return

    SCALE = 1.0 / (128.0 ** 0.5)
    
    q_stride_S = D
    q_stride_D = 1
    k_stride_S = D
    k_stride_D = 1
    v_stride_S = D
    v_stride_D = 1
    o_stride_S = D
    o_stride_D = 1
    
    grid = (B * H, triton.cdiv(S, 128))
    _mha_kernel[grid](
        Q, K, V, O, LSE, S, D,
        q_stride_S, q_stride_D,
        k_stride_S, k_stride_D,
        v_stride_S, v_stride_D,
        o_stride_S, o_stride_D,
        SCALE,
        BLOCK_M=128, BLOCK_N=64, HEAD_DIM=128,
        num_warps=8, num_stages=2
    )