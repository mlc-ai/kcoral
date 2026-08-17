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
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    b = tl.program_id(2)
    h = tl.program_id(1)
    i = tl.program_id(0)
    
    base_offset = b * H * S_len * D + h * S_len * D
    q_base = Q + base_offset
    k_base = K + base_offset
    v_base = V + base_offset
    o_base = O + base_offset
    lse_base = LSE + b * (h * S_len)
    
    rows = tl.arange(0, BLOCK_N)
    cols = tl.arange(0, HEAD_DIM)
    
    q_row_offsets = (i * BLOCK_N + rows[:, None]) * stride_q_s
    q_ptrs = q_base + q_row_offsets + cols[None, :] * stride_q_d
    Q_tile = tl.load(q_ptrs, mask=(i * BLOCK_N + rows[:, None]) < S_len, other=0.0)
    
    m = tl.full((BLOCK_N,), -float('inf'), tl.float32)
    l = tl.full((BLOCK_N,), 0.0, tl.float32)
    O_acc = tl.zeros((BLOCK_N, HEAD_DIM), tl.float32)
    
    for j in range(0, i + 1):
        k_row_offsets = (j * BLOCK_N + rows[:, None]) * stride_k_s
        k_ptrs = k_base + k_row_offsets + cols[None, :] * stride_k_d
        K_tile = tl.load(k_ptrs, mask=(j * BLOCK_N + rows[:, None]) < S_len, other=0.0)
        
        v_row_offsets = (j * BLOCK_N + rows[:, None]) * stride_v_s
        v_ptrs = v_base + v_row_offsets + cols[None, :] * stride_v_d
        V_tile = tl.load(v_ptrs, mask=(j * BLOCK_N + rows[:, None]) < S_len, other=0.0)
        
        S = tl.dot(Q_tile, K_tile.T) * scale
        
        is_causal = (i * BLOCK_N + rows[:, None]) >= (j * BLOCK_N + cols[None, :])
        S = tl.where(is_causal, S, -float('inf'))
        
        m_old = m
        m_local = tl.max(S, axis=1)
        m_new = tl.maximum(m_old, m_local)
        m = m_new
        
        P = tl.exp(S - m[:, None])
        
        l_local = tl.sum(P, axis=1)
        l = l * tl.exp(m_old - m) + l_local
        
        O_acc = O_acc * tl.exp(m_old - m)[:, None]
        O_acc = O_acc + tl.dot(P, V_tile)
        
    l_safe = tl.where(l > 0, l, 1.0)
    O_acc_out = O_acc / l_safe[:, None]
    
    out_ptrs = o_base + (i * BLOCK_N + rows[:, None]) * stride_o_s + cols[None, :] * stride_o_d
    tl.store(out_ptrs, O_acc_out, mask=(i * BLOCK_N + rows[:, None]) < S_len)
    
    lse = m + tl.log(l_safe)
    tl.store(lse_base + (i * BLOCK_N + rows), lse, mask=(i * BLOCK_N + rows) < S_len)


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
    
    grid = (triton.cdiv(S_len, 64), H, B)
    _attention_kernel[grid](
        Q, K, V, O, LSE,
        S_len, H, D,
        stride_q_s, stride_q_d,
        stride_k_s, stride_k_d,
        stride_v_s, stride_v_d,
        stride_o_s, stride_o_d,
        stride_lse_s,
        scale,
        BLOCK_N=64,
        HEAD_DIM=128,
        num_warps=4,
        num_stages=2,
    )